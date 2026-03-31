"""
CoCreator CLI - Driving scenario event detection and causal inference.
"""
import asyncio
from pathlib import Path
from typing import Any, Optional

import typer

try:
    from tqdm.asyncio import tqdm as async_tqdm
except ImportError:
    async_tqdm: Any = None

from .config import load_config
from .schemas import DetectedEvent, CausalChain
from .pipeline.detector import EventDetector
from .pipeline.extractor import VideoFrameExtractor
from .pipeline.reasoner import CausalReasoner
from .pipeline.progress_tracker import ProgressTracker
from .providers.openai_compatible import OpenAICompatibleProvider

app = typer.Typer(
    name="cocreator",
    help="Driving scenario event detection and causal inference tool"
)

# Output directory for progress files
PROGRESS_FILE = "progress.json"


@app.command()
def detect(
    config: Path = typer.Option(..., "-c", "--config", help="YAML config file path"),
    output: Path = typer.Option(..., "-o", "--output", help="Output JSONL file path"),
    episode_id: Optional[str] = typer.Option(None, help="Process specific episode only"),
) -> None:
    """
    Detect driving events from position data.

    Reads action_info from the dataset, detects velocity anomalies,
    and outputs events to JSONL file.
    """
    # Load config
    config_obj = load_config(str(config))

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = EventDetector(config_obj.pipeline)

    # Get episode directories
    dataset_path = Path(config_obj.pipeline.dataset_path)
    if episode_id:
        episode_ids = [episode_id]
    else:
        episode_ids = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    # Process episodes
    all_events = []
    typer.echo(f"Processing {len(episode_ids)} episodes...")

    for ep_id in episode_ids:
        typer.echo(f"  Processing {ep_id}...")
        try:
            events = detector.detect(ep_id)
            all_events.extend(events)
            typer.echo(f"    Found {len(events)} events")
        except Exception as e:
            typer.echo(f"    Error: {e}", err=True)

    # Write JSONL output
    with open(output, "w") as f:
        for event in all_events:
            f.write(event.model_dump_json() + "\n")

    typer.secho(f"✓ Detected {len(all_events)} events → {output}", fg=typer.colors.GREEN)


@app.command()
def reason(
    config: Path = typer.Option(..., "-c", "--config", help="YAML config file path"),
    events: Path = typer.Option(..., "-e", "--events", help="Input events JSONL file"),
    output: Path = typer.Option(..., "-o", "--output", help="Output chains JSONL file"),
    max_events: Optional[int] = typer.Option(None, help="Process only N events"),
    resume: bool = typer.Option(True, help="Resume from previous progress"),
) -> None:
    """
    Perform causal reasoning on detected events.

    Reads events from JSONL, performs 2-stage VLM analysis,
    and outputs causal chains to JSONL file.
    """
    # Load config
    config_obj = load_config(str(config))

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Load events
    events_list = []
    with open(events, "r") as f:
        for line in f:
            if line.strip():
                events_list.append(DetectedEvent.model_validate_json(line))

    if max_events:
        events_list = events_list[:max_events]

    typer.echo(f"Loaded {len(events_list)} events")

    # Initialize components
    progress_tracker = ProgressTracker(output.parent / PROGRESS_FILE)
    extractor = VideoFrameExtractor(config_obj.pipeline.videos_path)

    # Create VLM provider
    async def create_and_run():
        provider = OpenAICompatibleProvider(
            config_obj.vlm,
            config_obj.rate_limit
        )

        reasoner = CausalReasoner(provider, extractor, config_obj.pipeline)

        # Filter unprocessed events
        if resume:
            remaining = [
                e for e in events_list
                if not progress_tracker.is_processed(e.episode_id, e.frame_id)
            ]
            typer.echo(f"Resuming: {len(remaining)} events remaining")
        else:
            remaining = events_list

        # Process events with progress bar
        chains = []
        if async_tqdm is not None:
            iterator = async_tqdm(remaining, desc="Reasoning")
        else:
            iterator = remaining

        for event in iterator:
            try:
                chain = await reasoner.reason(event.episode_id, event)
                chains.append(chain)
                progress_tracker.mark_processed_sync(event.episode_id, event.frame_id)
            except Exception as e:
                typer.echo(f"Error processing {event.episode_id}/{event.frame_id}: {e}", err=True)

        await provider.close()
        return chains

    # Run async pipeline
    chains = asyncio.run(create_and_run())

    # Write JSONL output
    with open(output, "w") as f:
        for chain in chains:
            f.write(chain.model_dump_json() + "\n")

    typer.secho(f"✓ Generated {len(chains)} causal chains → {output}", fg=typer.colors.GREEN)


@app.command()
def review(
    input: Path = typer.Option(..., "-i", "--input", help="Input chains JSONL file"),
    count: int = typer.Option(10, "-n", "--count", help="Number of chains to include"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output markdown file"),
) -> None:
    """
    Generate Markdown report from causal chains.

    Reads causal chains from JSONL and generates a human-readable
    markdown report for manual review.
    """
    # Load chains
    chains = []
    with open(input, "r") as f:
        for line in f:
            if line.strip():
                chains.append(CausalChain.model_validate_json(line))

    if not chains:
        typer.echo("No chains found in input file", err=True)
        raise typer.Exit(1)

    # Limit to count
    chains = chains[:count]

    # Generate markdown
    lines = [
        "# Causal Chain Report",
        "",
        f"Generated from {len(chains)} causal chains.",
        "",
    ]

    for i, chain in enumerate(chains, 1):
        lines.append(f"## Chain {i}: Episode {chain.episode_id} - Frame {chain.event_frame_id}")
        lines.append("")
        lines.append(f"**Confidence:** {chain.confidence * 100:.2f}%")
        lines.append("")
        lines.append("### Historical Analysis")
        lines.append(f"- Ego Status: {chain.historical_analysis.ego_status}")
        lines.append(f"- Predicted Action: {chain.historical_analysis.predicted_action}")
        if chain.historical_analysis.most_critical_object:
            obj = chain.historical_analysis.most_critical_object
            lines.append(f"- Critical Object: {obj.type} at {obj.location}")
        lines.append("")
        lines.append("### Future Confirmation")
        lines.append(f"- Actual Action: {chain.future_confirmation.actual_action}")
        lines.append(f"- Action Status: {chain.future_confirmation.action_status}")
        lines.append(f"- Related to History: {chain.future_confirmation.related_to_history}")
        lines.append("")
        lines.append("### Causal Link")
        lines.append(chain.causal_link)
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    # Output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(report)
        typer.secho(f"✓ Report saved → {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(report)


if __name__ == "__main__":
    app()