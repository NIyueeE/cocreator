"""
CoCreator CLI - Driving scenario event detection and causal inference.
"""

import asyncio
import base64
import concurrent.futures
import json
import os
import random
import shutil
import threading
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .config import load_config
from .schemas import AppConfig, DetectedEvent
from .pipeline.detector import EventDetector
from .pipeline.extractor import VideoFrameExtractor
from .pipeline.reasoner import CausalReasoner
from .pipeline.progress_tracker import ProgressTracker
from .providers.openai_compatible import OpenAICompatibleProvider

app = typer.Typer(
    name="cocreator", help="Driving scenario event detection and causal inference tool"
)

pack_app = typer.Typer(
    invoke_without_command=True,
    help="Pack causal chains into a training dataset or manage packed datasets.",
)
app.add_typer(pack_app, name="pack")

# Output directory for progress files
PROGRESS_FILE = "progress.json"


@app.command()
def detect(
    config: Path = typer.Option(
        "./config.yaml", "-c", "--config", help="YAML config file path"
    ),
    episode_ids: Optional[List[str]] = typer.Option(
        None, "--episode-id", help="Process specific episode(s). Repeat for multiple."
    ),
) -> None:
    """
    Detect driving events from position data.

    Reads action_info from the dataset, detects velocity anomalies,
    and writes one JSON file per event to {output}/events/.
    """
    config_obj = load_config(str(config))
    output = Path(config_obj.pipeline.output_dir)
    events_dir = output / "events"

    # Initialize detector
    detector = EventDetector(config_obj.pipeline)

    # Get episode directories
    dataset_path = Path(config_obj.pipeline.dataset_path)
    if not dataset_path.exists():
        typer.echo(f"Dataset path not found: {dataset_path}", err=True)
        raise typer.Exit(1)
    if episode_ids:
        # verify specified episodes exist
        for eid in episode_ids:
            if not (dataset_path / eid).is_dir():
                typer.echo(f"Episode not found: {eid}", err=True)
                raise typer.Exit(1)
    else:
        episode_ids = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    if not episode_ids:
        typer.echo("No episode directories found", err=True)
        raise typer.Exit(1)

    all_events = []
    max_workers = min(16, os.cpu_count() or 1)
    active_lock = threading.Lock()
    active_count = 0

    def _tracked_detect(ep_id: str) -> list:
        nonlocal active_count
        with active_lock:
            active_count += 1
        try:
            return detector.detect(ep_id)
        finally:
            with active_lock:
                active_count -= 1

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn(" • active "),
        TextColumn("{task.fields[active]}"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        transient=True,
    )
    task = progress.add_task("Detecting", total=len(episode_ids), active="0/0")

    with progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_tracked_detect, ep_id): ep_id for ep_id in episode_ids
            }
            for future in concurrent.futures.as_completed(futures):
                ep_id = futures[future]
                try:
                    events = future.result()
                    all_events.extend(events)
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{max_workers}",
                        info=f"{ep_id[:32]}… {len(events)} events",
                    )
                except Exception:
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{max_workers}",
                        info=f"{ep_id[:32]}… error",
                    )

    # Write individual JSON files (atomic: .tmp + rename)
    events_dir.mkdir(parents=True, exist_ok=True)
    for event in all_events:
        filename = f"{event.episode_id}_{event.frame_id}.json"
        filepath = events_dir / filename
        temppath = filepath.with_suffix(".tmp")
        with open(temppath, "w") as f:
            f.write(event.model_dump_json())
        temppath.replace(filepath)

    typer.secho(
        f"✓ Detected {len(all_events)} events → {events_dir}/", fg=typer.colors.GREEN
    )


@app.command()
def reason(
    config: Path = typer.Option(
        "./config.yaml", "-c", "--config", help="YAML config file path"
    ),
    event_ids: Optional[List[str]] = typer.Option(
        None,
        "--event-id",
        help="Process specific event(s) as episode_id:frame_id. Repeat for multiple.",
    ),
    resume: bool = typer.Option(True, help="Resume from previous progress"),
) -> None:
    """
    Perform causal reasoning on detected events.

    Reads events from a directory or a single JSON file, performs 2-stage VLM analysis,
    and writes one JSON file per causal chain to {output}/chains/.
    """
    config_obj = load_config(str(config))
    output = Path(config_obj.pipeline.output_dir)
    events_path = output / "events"
    chains_dir = output / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    # Load events (directory = batch, single file = one event)
    events_list = []
    if events_path.is_dir():
        for event_file in sorted(events_path.iterdir()):
            if event_file.suffix == ".json":
                with open(event_file) as f:
                    events_list.append(DetectedEvent.model_validate_json(f.read()))
    elif events_path.suffix == ".json":
        with open(events_path) as f:
            events_list.append(DetectedEvent.model_validate_json(f.read()))
    else:
        typer.echo(
            f"Events path must be a directory or .json file: {events_path}", err=True
        )
        raise typer.Exit(1)

    if event_ids:
        filter_set = set(event_ids)
        matched = [
            e for e in events_list if f"{e.episode_id}:{e.frame_id}" in filter_set
        ]
        if not matched:
            typer.echo("No matching events found for specified --event-id(s)", err=True)
            raise typer.Exit(1)
        typer.echo(
            f"Filtered: {len(matched)}/{len(events_list)} events match specified IDs"
        )
        events_list = matched

    typer.echo(f"Loaded {len(events_list)} events")

    # Initialize components
    progress_tracker = ProgressTracker(chains_dir / PROGRESS_FILE)
    extractor = VideoFrameExtractor(config_obj.pipeline.videos_path)

    # Create VLM provider
    async def create_and_run():
        provider = OpenAICompatibleProvider(config_obj.vlm, config_obj.rate_limit)

        reasoner = CausalReasoner(provider, extractor, config_obj.pipeline)

        # Filter unprocessed events
        if resume:
            remaining = [
                e
                for e in events_list
                if not progress_tracker.is_processed(e.episode_id, e.frame_id)
            ]
            typer.echo(f"Resuming: {len(remaining)} events remaining")
        else:
            remaining = events_list

        # Process events concurrently (respect rate_limit.concurrency)
        total = len(remaining)
        sem = asyncio.Semaphore(config_obj.rate_limit.concurrency)
        active_count = 0
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn(" • active "),
            TextColumn("{task.fields[active]}"),
            TimeElapsedColumn(),
            TextColumn("<"),
            TimeRemainingColumn(),
            transient=True,
        )
        task = progress.add_task("Reasoning", total=total, active="0/0")

        success_count = 0
        errors = []

        async def process_one(event: DetectedEvent) -> None:
            nonlocal active_count, success_count
            async with sem:
                active_count += 1
                progress.update(
                    task, active=f"{active_count}/{config_obj.rate_limit.concurrency}"
                )
                try:
                    chain = await reasoner.reason(event.episode_id, event)

                    # Write immediately (atomic: .tmp + rename)
                    filename = f"{chain.episode_id}_{chain.event_frame_id}.json"
                    filepath = chains_dir / filename
                    temppath = filepath.with_suffix(".tmp")
                    with open(temppath, "w") as f:
                        f.write(chain.model_dump_json())
                    temppath.replace(filepath)

                    await progress_tracker.mark_processed(
                        event.episode_id, event.frame_id
                    )
                    success_count += 1
                except Exception as e:
                    errors.append((event.episode_id, event.frame_id, e))
                finally:
                    active_count -= 1
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{config_obj.rate_limit.concurrency}",
                    )

        with progress:
            await asyncio.gather(
                *[process_one(e) for e in remaining],
                return_exceptions=True,
            )

        # Print errors after gather (avoids interleaved output)
        for ep_id, fr_id, e in errors:
            typer.echo(f"  Error [{ep_id}/{fr_id}]: {e}", err=True)

        typer.echo(f"  ✓ {success_count}/{total} completed")

        await provider.close()

    # Run async pipeline
    asyncio.run(create_and_run())

    typer.secho(f"✓ Generated chains → {chains_dir}/", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------


def _img_to_b64(path: Path) -> str:
    """Read an image file and return a base64 data URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _parse_frame_num(frame_id: str) -> int:
    """Extract frame number from frame_id (same logic as VideoFrameExtractor)."""
    for part in frame_id.split("_"):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Could not parse frame number from: {frame_id}")


def _build_dataset_report(dataset_dir: Path, samples_meta: Optional[dict] = None, max_samples: int = 7) -> str:
    """Generate an HTML report from the dataset structure with embedded images."""
    videos_dir = dataset_dir / "videos"
    if not videos_dir.exists():
        return "<html><body><p>No videos/ directory found in dataset.</p></body></html>"

    sample_ids = sorted(d.name for d in videos_dir.iterdir() if d.is_dir())
    total = len(sample_ids)

    # random sample
    k = min(max_samples, total)
    selected = random.sample(sample_ids, k)

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoCreator Dataset Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; padding: 24px; }
  h1 { font-size: 22px; color: #1a1a2e; margin-bottom: 4px; }
  .subtitle { font-size: 14px; color: #666; margin-bottom: 24px; }
  .chain { background: #fff; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); margin-bottom: 20px; padding: 20px; }
  .chain-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
  .chain-header h2 { font-size: 15px; color: #1a1a2e; }
  .chain-header .badge { font-size: 11px; background: #e8ecf4; color: #555; padding: 2px 10px; border-radius: 10px; }
  .frames { display: flex; flex-wrap: wrap; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
  .frame { flex: 0 0 auto; text-align: center; }
  .frame img { width: 140px; height: 105px; object-fit: cover; border-radius: 6px; border: 2px solid #ddd; display: block; }
  .frame .label { font-size: 10px; color: #888; margin-top: 2px; }
  .frame.history img { border-color: #4a90d9; }
  .frame.future img { border-color: #4caf50; }
  .event-marker { font-size: 12px; font-weight: 600; color: #e53935; margin: 8px 0; }
  .causal-text { font-size: 13.5px; line-height: 1.65; color: #333; background: #fafafa; padding: 12px 16px; border-radius: 8px; margin-top: 10px; max-height: 200px; overflow-y: auto; }
  .causal-text::before { content: "causal_text"; display: block; font-size: 11px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
</style>
</head>
<body>
<h1>CoCreator Dataset Report</h1>
""")
    if selected:
        first_dir = dataset_dir / "videos" / selected[0]
        n_frames = len(list(first_dir.glob("*.jpg")))
    else:
        n_frames = 0
    parts.append(
        f'<p class="subtitle">{total} total samples &middot; showing {len(selected)} samples &middot; {n_frames} frames per sample</p>'
    )

    for sid in selected:
        video_dir = dataset_dir / "videos" / sid
        num_frames = len(list(video_dir.glob("*.jpg")))
        causal_path = dataset_dir / "causal" / f"{sid}.txt"
        causal_text = (
            causal_path.read_text(encoding="utf-8") if causal_path.exists() else ""
        )

        # use actual event boundary from chain
        if not samples_meta or sid not in samples_meta:
            raise ValueError(f"Missing metadata for sample {sid}")
        mid = samples_meta[sid].get("event_idx")
        if mid is None:
            raise ValueError(f"Missing event_idx in metadata for sample {sid}")

        parts.append('<div class="chain">')
        parts.append(
            f'<div class="chain-header"><h2>Sample {sid}</h2><span class="badge">{num_frames} frames</span></div>'
        )
        parts.append('<div class="frames">')

        for fi in range(num_frames):
            cls = "history" if fi < mid else "future"
            is_event = fi == mid - 1
            img_path = video_dir / f"{fi + 1:02d}.jpg"
            src = _img_to_b64(img_path) if img_path.exists() else ""
            label = "EVENT" if is_event else f"{fi + 1:02d}"
            parts.append(
                f'<div class="frame {cls}"><img src="{src}" alt="frame {fi + 1}"><span class="label">{label}</span></div>'
            )

        parts.append("</div>")
        parts.append(f'<div class="causal-text">{causal_text}</div>')
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


@pack_app.callback()
def pack(
    ctx: typer.Context,
    config: Path = typer.Option(
        "./config.yaml", "-c", "--config", help="YAML config file path"
    ),
) -> None:
    """Pack causal chains into a training dataset or manage packed datasets."""
    config_obj = load_config(str(config))

    if ctx.invoked_subcommand is not None:
        ctx.obj = config_obj
        return

    output = Path(config_obj.pipeline.output_dir)
    chains_dir = output / "chains"
    dataset_dir = output / "dataset"
    videos_dir = dataset_dir / "videos"
    causal_dir = dataset_dir / "causal"
    video_source = Path(config_obj.pipeline.videos_path)

    chain_files = sorted(
        f for f in chains_dir.glob("*.json") if f.name != "progress.json"
    )
    if not chain_files:
        typer.echo("No chain files found", err=True)
        raise typer.Exit(1)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    causal_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    errors = []
    max_workers = min(16, os.cpu_count() or 1)
    active_lock = threading.Lock()
    active_count = 0

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn(" • active "),
        TextColumn("{task.fields[active]}"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        transient=True,
    )
    task = progress.add_task("Packing", total=len(chain_files), active="0/0")

    def pack_one(idx: int, chain_path: Path) -> tuple[Optional[dict], Optional[str]]:
        nonlocal active_count
        with active_lock:
            active_count += 1
        try:
            sample_id = f"{idx + 1:04d}"
            chain = json.loads(chain_path.read_text())

            episode_id = chain["episode_id"]
            frame_ids = chain["frame_ids"]
            event_frame_id = chain["event_frame_id"]
            causal_text = chain["causal_text"]
            event_num = _parse_frame_num(event_frame_id)
            event_idx = sum(1 for fid in frame_ids if _parse_frame_num(fid) < event_num)

            sample_video_dir = videos_dir / sample_id
            sample_video_dir.mkdir(parents=True, exist_ok=True)

            src_episode_dir = video_source / episode_id
            if not src_episode_dir.exists():
                return None, f"{sample_id}: episode dir not found {episode_id}"

            action_type = chain.get("action_type", "unknown")
            event_idx = sum(1 for fid in frame_ids if _parse_frame_num(fid) <= event_num)

            for fi, fid in enumerate(frame_ids):
                dst = sample_video_dir / f"{fi + 1:02d}.jpg"
                matches = sorted(src_episode_dir.glob(f"{fid}_*.jpg"))
                if not matches:
                    dst.write_bytes(b"")
                    continue
                shutil.copy2(matches[0], dst)

            (causal_dir / f"{sample_id}.txt").write_text(causal_text, encoding="utf-8")
            return {
                "id": sample_id,
                "episode_id": episode_id,
                "event_frame_id": chain["event_frame_id"],
                "action_type": action_type,
                "num_frames": len(frame_ids),
                "event_idx": event_idx,
            }, None
        finally:
            with active_lock:
                active_count -= 1

    with progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(pack_one, idx, cp): cp
                for idx, cp in enumerate(chain_files)
            }
            for future in concurrent.futures.as_completed(futures):
                result, error = future.result()
                if error:
                    errors.append(error)
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{max_workers}",
                        info=f"{error[:20]}… missing",
                    )
                else:
                    meta.append(result)
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{max_workers}",
                        info=f"{result['id']}… ✓",
                    )

    samples_meta = {m["id"]: m for m in meta}
    html = _build_dataset_report(dataset_dir, samples_meta=samples_meta)
    (dataset_dir / "review.html").write_text(html, encoding="utf-8")

    typer.secho(f"✓ Packed {len(meta)} samples → {dataset_dir}/", fg=typer.colors.GREEN)
    if errors:
        typer.echo(f"  {len(errors)} warnings:", err=True)
        for e in errors[:5]:
            typer.echo(f"    {e}", err=True)
    typer.echo(f"  review → {dataset_dir}/review.html")


@pack_app.command()
def review(
    ctx: typer.Context,
) -> None:
    """Regenerate review HTML report from an already-packed dataset."""
    config_obj: AppConfig = ctx.obj
    output = Path(config_obj.pipeline.output_dir)
    dataset_dir = output / "dataset"
    chains_dir = output / "chains"
    videos_dir = dataset_dir / "videos"

    if not dataset_dir.exists():
        typer.echo(f"Dataset directory not found: {dataset_dir}", err=True)
        raise typer.Exit(1)

    if not videos_dir.exists():
        typer.echo(f"Videos directory not found: {videos_dir}", err=True)
        raise typer.Exit(1)

    chain_files = sorted(
        f for f in chains_dir.glob("*.json") if f.name != "progress.json"
    )
    sample_ids = sorted(d.name for d in videos_dir.iterdir() if d.is_dir())

    samples_meta = {}
    for sid in sample_ids:
        idx = int(sid) - 1
        if idx >= len(chain_files):
            continue
        chain = json.loads(chain_files[idx].read_text())
        event_num = _parse_frame_num(chain["event_frame_id"])
        event_idx = sum(
            1 for fid in chain["frame_ids"] if _parse_frame_num(fid) <= event_num
        )
        samples_meta[sid] = {
            "id": sid,
            "episode_id": chain["episode_id"],
            "event_frame_id": chain["event_frame_id"],
            "action_type": chain.get("action_type", "unknown"),
            "num_frames": len(chain["frame_ids"]),
            "event_idx": event_idx,
        }

    html = _build_dataset_report(dataset_dir, samples_meta=samples_meta)
    (dataset_dir / "review.html").write_text(html, encoding="utf-8")
    typer.secho(f"✓ Review report → {dataset_dir}/review.html", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
