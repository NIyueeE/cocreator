import asyncio
import json
from pathlib import Path
from typing import Set
from dataclasses import dataclass, field


@dataclass
class ProgressTracker:
    """
    Tracks processed events for resume capability.

    Saves progress to progress.json to allow resuming
    long-running pipeline tasks after interruption.
    """

    progress_file: Path
    processed: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Load existing progress if file exists."""
        self.progress_file = Path(self.progress_file)
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                self.processed = set(data.get("processed", []))

    def _make_key(self, episode_id: str, frame_id: str) -> str:
        """Create unique key for an event."""
        return f"{episode_id}:{frame_id}"

    def is_processed(self, episode_id: str, frame_id: str) -> bool:
        """Check if an event has been processed."""
        key = self._make_key(episode_id, frame_id)
        return key in self.processed

    def mark_processed_sync(self, episode_id: str, frame_id: str) -> None:
        """Synchronous version for non-async code."""
        key = self._make_key(episode_id, frame_id)
        self.processed.add(key)
        self._save()

    async def mark_processed(self, episode_id: str, frame_id: str) -> None:
        """Async version that offloads file I/O to a thread pool."""
        key = self._make_key(episode_id, frame_id)
        self.processed.add(key)
        await asyncio.to_thread(self._save)

    def _save(self) -> None:
        """Save progress to JSON file (sync)."""
        data = {"processed": list(self.processed)}
        temp_file = self.progress_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f)
        temp_file.replace(self.progress_file)

    @property
    def total_processed(self) -> int:
        """Return count of processed events."""
        return len(self.processed)
