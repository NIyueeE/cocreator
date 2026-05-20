"""Tests for ProgressTracker.

Covers save/load, atomic write, and resume capability.
"""
import json
from pathlib import Path

from cocreator.pipeline.progress_tracker import ProgressTracker


class TestProgressTracker:
    def test_empty_tracker(self, tmp_path):
        tracker = ProgressTracker(tmp_path / "progress.json")
        assert tracker.total_processed == 0
        assert not tracker.is_processed("ep001", "frame_001")

    def test_mark_processed_sync(self, tmp_path):
        tracker = ProgressTracker(tmp_path / "progress.json")
        tracker.mark_processed_sync("ep001", "frame_001")
        assert tracker.is_processed("ep001", "frame_001")
        assert tracker.total_processed == 1

    def test_load_existing_progress(self, tmp_path):
        pf = tmp_path / "progress.json"
        pf.write_text(json.dumps({"processed": ["ep001:frame_001", "ep001:frame_002"]}))

        tracker = ProgressTracker(pf)
        assert tracker.is_processed("ep001", "frame_001")
        assert tracker.is_processed("ep001", "frame_002")
        assert not tracker.is_processed("ep001", "frame_003")
        assert tracker.total_processed == 2

    def test_atomic_write(self, tmp_path):
        """Progress file should only exist after successful write (via .tmp rename)."""
        pf = tmp_path / "progress.json"
        tracker = ProgressTracker(pf)
        tracker.mark_processed_sync("ep001", "frame_001")
        assert pf.exists()
        # .tmp should not remain after write
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
