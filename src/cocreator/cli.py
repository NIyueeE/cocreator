"""
CoCreator CLI - Driving scenario event detection and causal inference.
"""

import asyncio
import base64
from pathlib import Path
from random import sample as random_sample
from typing import Optional

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import traceback
from .config import load_config
from .schemas import DetectedEvent, CausalChain
from .pipeline.detector import EventDetector
from .pipeline.extractor import VideoFrameExtractor
from .pipeline.reasoner import CausalReasoner
from .pipeline.progress_tracker import ProgressTracker
from .providers.openai_compatible import OpenAICompatibleProvider

app = typer.Typer(
    name="cocreator", help="Driving scenario event detection and causal inference tool"
)

# Output directory for progress files
PROGRESS_FILE = "progress.json"


@app.command()
def detect(
    config: Path = typer.Option(
        "./config.yaml", "-c", "--config", help="YAML config file path"
    ),
    output: Path = typer.Option(
        "./output", "-o", "--output", help="Output directory (events go to {output}/events/)"
    ),
    episode_id: Optional[str] = typer.Option(
        None, help="Process specific episode only"
    ),
) -> None:
    """
    Detect driving events from position data.

    Reads action_info from the dataset, detects velocity anomalies,
    and writes one JSON file per event to {output}/events/.
    """
    # Load config
    config_obj = load_config(str(config))

    events_dir = output / "events"

    # Initialize detector
    detector = EventDetector(config_obj.pipeline)

    # Get episode directories
    dataset_path = Path(config_obj.pipeline.dataset_path)
    if episode_id:
        episode_ids = [episode_id]
    else:
        episode_ids = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    # Process episodes with progress bar
    all_events = []
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn(" • "),
        TextColumn("{task.fields[info]}"),
        TimeElapsedColumn(),
        transient=True,
    )
    task = progress.add_task("Detecting", total=len(episode_ids), info="")

    with progress:
        for ep_id in episode_ids:
            try:
                events = detector.detect(ep_id)
                all_events.extend(events)
                progress.update(
                    task, advance=1, info=f"{ep_id[:32]}… {len(events)} events"
                )
            except Exception:
                progress.update(
                    task, advance=1, info=f"{ep_id[:32]}… error"
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
    events: Path = typer.Option(
        "./output/events", "-e", "--events", help="Input events directory or single event JSON file"
    ),
    output: Path = typer.Option(
        "./output", "-o", "--output", help="Output directory (chains go to {output}/chains/)"
    ),
    max_events: Optional[int] = typer.Option(None, help="Process only N events"),
    resume: bool = typer.Option(True, help="Resume from previous progress"),
) -> None:
    """
    Perform causal reasoning on detected events.

    Reads events from a directory or a single JSON file, performs 2-stage VLM analysis,
    and writes one JSON file per causal chain to {output}/chains/.
    """
    # Load config
    config_obj = load_config(str(config))

    chains_dir = output / "chains"
    chains_dir.mkdir(parents=True, exist_ok=True)

    # Load events (directory = batch, single file = one event)
    events_list = []
    if events.is_dir():
        for event_file in sorted(events.iterdir()):
            if event_file.suffix == ".json":
                with open(event_file) as f:
                    events_list.append(DetectedEvent.model_validate_json(f.read()))
    elif events.suffix == ".json":
        with open(events) as f:
            events_list.append(DetectedEvent.model_validate_json(f.read()))
    else:
        typer.echo(f"Events path must be a directory or .json file: {events}", err=True)
        raise typer.Exit(1)

    if max_events:
        events_list = events_list[:max_events]

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

        async def process_one(event: DetectedEvent) -> CausalChain:
            nonlocal active_count
            async with sem:
                active_count += 1
                progress.update(
                    task, active=f"{active_count}/{config_obj.rate_limit.concurrency}"
                )
                try:
                    chain = await reasoner.reason(event.episode_id, event)
                    await progress_tracker.mark_processed(
                        event.episode_id, event.frame_id
                    )
                    return chain
                finally:
                    active_count -= 1
                    progress.update(
                        task,
                        advance=1,
                        active=f"{active_count}/{config_obj.rate_limit.concurrency}",
                    )

        with progress:
            results = await asyncio.gather(
                *[process_one(e) for e in remaining],
                return_exceptions=True,
            )

        chains = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                typer.echo(
                    f"Error [{remaining[i].episode_id}/{remaining[i].frame_id}]: {r}",
                    err=True,
                )
                traceback.print_exc()
            else:
                chains.append(r)

        typer.echo(f"  ✓ {len(chains)}/{total} completed")

        await provider.close()
        return chains

    # Run async pipeline
    chains = asyncio.run(create_and_run())

    # Write individual JSON files (atomic: .tmp + rename)
    chains_dir.mkdir(parents=True, exist_ok=True)
    for chain in chains:
        filename = f"{chain.episode_id}_{chain.event_frame_id}.json"
        filepath = chains_dir / filename
        temppath = filepath.with_suffix(".tmp")
        with open(temppath, "w") as f:
            f.write(chain.model_dump_json())
        temppath.replace(filepath)

    typer.secho(
        f"✓ Generated {len(chains)} causal chains → {chains_dir}/", fg=typer.colors.GREEN
    )


@app.command()
def review(
    config: Path = typer.Option(
        "./config.yaml", "-c", "--config", help="YAML config file path"
    ),
    input: Path = typer.Option(
        "./output/chains", "-i", "--input", help="Input chains directory (random 1% sample) or single chain JSON file"
    ),
    count: int = typer.Option(
        10, "-n", "--count", help="Max chains in report (for directory mode)"
    ),
    output: Path = typer.Option(
        "./output", "-o", "--output", help="Output directory (report goes to {output}/report.html)"
    ),
) -> None:
    """
    Generate HTML report from causal chains.

    When input is a directory, randomly samples ~1% of chains.
    When input is a single JSON file, processes just that chain.
    """
    config_obj = load_config(str(config))
    videos_path = Path(config_obj.pipeline.videos_path)

    report_path = output / "report.html"

    # Load chains
    chains = []
    if input.is_dir():
        for chain_file in sorted(input.iterdir()):
            if chain_file.suffix == ".json":
                with open(chain_file) as f:
                    chains.append(CausalChain.model_validate_json(f.read()))
        total = len(chains)
        sample_count = max(1, int(total * 0.01))
        chains = random_sample(chains, min(sample_count, count))
        typer.echo(f"Loaded {total} chains, sampled {len(chains)} (1%)")
    elif input.suffix == ".json":
        with open(input) as f:
            chains.append(CausalChain.model_validate_json(f.read()))
    else:
        typer.echo(f"Input path must be a directory or .json file: {input}", err=True)
        raise typer.Exit(1)

    if not chains:
        typer.echo("No chains found in input file", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generating report with {len(chains)} chain(s)")

    # Generate HTML
    report_path.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html_report(chains, videos_path)

    with open(report_path, "w") as f:
        f.write(html)

    typer.secho(f"✓ Report saved → {report_path}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------


_HISTORY_COLOR = "#0064C8"  # blue
_FUTURE_COLOR = "#00B450"  # green


def _img_to_b64(path: Path) -> str:
    """Read an image file and return a base64 data URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _find_frame_file(video_dir: Path, frame_id: str) -> Optional[Path]:
    """Match a frame number to its image file."""
    for f in video_dir.iterdir():
        if f.suffix == ".jpg" and f.stem.startswith(frame_id):
            return f
    return None


def _build_html_report(chains: list[CausalChain], videos_path: Path) -> str:
    """Generate a standalone HTML report with embedded images."""
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoCreator Causal Analysis Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, 'Segoe UI', sans-serif; background: #f5f5f5; padding: 20px; }
  .chain { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); margin-bottom: 24px; padding: 24px; }
  .chain-header { margin-bottom: 16px; }
  .chain-header h2 { font-size: 18px; color: #333; }
  .chain-header .meta { font-size: 13px; color: #888; margin-top: 4px; }
  .frames { display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-end; margin-bottom: 16px; }
  .frame { display: flex; flex-direction: column; align-items: center; }
  .frame img { width: 160px; height: 120px; object-fit: cover; border-radius: 4px; border: 3px solid #ccc; }
  .frame .label { font-size: 11px; color: #666; margin-top: 4px; }
  .frame.history img { border-color: #0064C8; }
  .frame.future img { border-color: #00B450; }
  .event-marker { font-size: 14px; font-weight: bold; color: #c00; margin: 4px 0 8px; }
  .section-title { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 6px; }
  .analysis { font-size: 14px; line-height: 1.6; color: #333; white-space: pre-wrap; }
</style>
</head>
<body>
""")

    for i, chain in enumerate(chains, 1):
        video_dir = videos_path / chain.episode_id
        n_history = 7
        history_ids = chain.frame_ids[:n_history] if len(chain.frame_ids) > n_history else chain.frame_ids
        future_ids = chain.frame_ids[n_history:] if len(chain.frame_ids) > n_history else []

        parts.append(f'<div class="chain">')
        parts.append(f'<div class="chain-header"><h2>Chain {i}/{len(chains)}</h2>')
        parts.append(f'<div class="meta">Episode: {chain.episode_id} &mdash; Event Frame: {chain.event_frame_id}</div></div>')

        # History frames
        if history_ids:
            parts.append(f'<div class="section-title">History frames (before event)</div>')
            parts.append(f'<div class="frames">')
            for fid in history_ids:
                img_path = _find_frame_file(video_dir, fid)
                src = _img_to_b64(img_path) if img_path else ""
                parts.append(f'<div class="frame history"><img src="{src}" alt="{fid}"><span class="label">{fid}</span></div>')
            parts.append(f'</div>')

        # Event marker
        parts.append(f'<div class="event-marker">&#9660; EVENT</div>')

        # Future frames
        if future_ids:
            parts.append(f'<div class="section-title">Future frames (after event)</div>')
            parts.append(f'<div class="frames">')
            for fid in future_ids:
                img_path = _find_frame_file(video_dir, fid)
                src = _img_to_b64(img_path) if img_path else ""
                parts.append(f'<div class="frame future"><img src="{src}" alt="{fid}"><span class="label">{fid}</span></div>')
            parts.append(f'</div>')

        # Analysis
        parts.append(f'<div class="section-title">Analysis</div>')
        parts.append(f'<div class="analysis">{chain.causal_text}</div>')
        parts.append(f'</div>')

    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    app()
