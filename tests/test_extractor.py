"""Tests for FrameExtractor.

Covers temporal isolation, frame parsing, and edge cases.
"""
import pytest
from pathlib import Path

from cocreator.pipeline.extractor import VideoFrameExtractor


class TestVideoFrameExtractor:
    def test_history_frames_exclude_event(self, tmp_path):
        """History frames are strictly before the event frame."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(100):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        history = extractor.get_history_frames("ep001", "frame_0050", count=5)

        for fp in history:
            num = int(Path(fp).stem.split("_")[1])
            assert num < 50

    def test_future_frames_include_event(self, tmp_path):
        """Future frames include the event frame."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(100):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        future = extractor.get_future_frames("ep001", "frame_0050", count=5)

        for fp in future:
            num = int(Path(fp).stem.split("_")[1])
            assert num >= 50

    def test_parse_frame_num_various_formats(self):
        """_parse_frame_num should handle both naming conventions."""
        extractor = VideoFrameExtractor("/tmp")
        assert extractor._parse_frame_num("frame_0057") == 57
        assert extractor._parse_frame_num("0039_next_frame_position_at_current_camera") == 39

    def test_not_enough_history_warns(self, tmp_path):
        """When there aren't enough history frames, return what's available."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(5):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        history = extractor.get_history_frames("ep001", "frame_0003", count=10)
        assert len(history) == 3  # frames 0,1,2 (strictly before event frame 3)

    def test_not_enough_future_warns(self, tmp_path):
        """When there aren't enough future frames, return what's available."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(5):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        future = extractor.get_future_frames("ep001", "frame_0003", count=10)
        assert len(future) == 2  # frames 3,4 (includes event frame 3)

    def test_history_frames_chronological_order(self, tmp_path):
        """History frames must be in chronological order (oldest first)."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(100):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        history = extractor.get_history_frames("ep001", "frame_0050", count=10)

        nums = [int(Path(fp).stem.split("_")[1]) for fp in history]
        # Each frame must be older (smaller number) than the next one
        for i in range(len(nums) - 1):
            assert nums[i] < nums[i + 1], (
                f"History not chronological: {nums}"
            )
        assert nums[-1] < 50  # closest frame is right before event

    def test_future_frames_chronological_order(self, tmp_path):
        """Future frames must be in chronological order (oldest first)."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(100):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        future = extractor.get_future_frames("ep001", "frame_0050", count=10)

        nums = [int(Path(fp).stem.split("_")[1]) for fp in future]
        for i in range(len(nums) - 1):
            assert nums[i] < nums[i + 1], (
                f"Future not chronological: {nums}"
            )
        assert nums[0] == 50  # first frame is the event frame

    def test_history_frames_reverse_confirmation(self, tmp_path):
        """Confirm history frames are NOT reversed by checking last frame is the closest to the event."""
        video_dir = tmp_path / "ep001"
        video_dir.mkdir()
        for i in range(100, 200):
            (video_dir / f"frame_{i:04d}.jpg").write_text(f"img{i}")

        extractor = VideoFrameExtractor(str(tmp_path))
        history = extractor.get_history_frames("ep001", "frame_0150", count=5)

        nums = [int(Path(fp).stem.split("_")[1]) for fp in history]
        # Last frame should be frame_0149 (right before event)
        assert nums[-1] == 149, f"Last frame should be frame_0149, got frame_{nums[-1]:04d}"
        assert nums[0] == 145, f"First frame should be frame_0145, got frame_{nums[0]:04d}"

    def test_nonexistent_episode_raises(self, tmp_path):
        """Requesting a non-existent episode should raise ValueError."""
        extractor = VideoFrameExtractor(str(tmp_path))
        with pytest.raises(ValueError, match="Episode directory not found"):
            extractor.get_history_frames("nonexistent", "frame_0001", count=1)
