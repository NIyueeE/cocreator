"""Integration smoke tests for the CoCreator pipeline.

These tests verify the full pipeline works end-to-end with mocked VLM.
"""
import pytest
import os
from pathlib import Path

from cocreator.schemas import CausalChain, HistoricalAnalysis, FutureConfirmation, KeyObject, PipelineConfig
from cocreator.config import load_config
from cocreator.pipeline.detector import EventDetector
from cocreator.pipeline.extractor import VideoFrameExtractor
from cocreator.pipeline.progress_tracker import ProgressTracker


@pytest.mark.smoke
class TestEventDetectorIntegration:
    """Test EventDetector with mocked data."""

    def test_detector_with_mock_data(self, tmp_path):
        """Test detector on mocked position data."""
        # Create mock episode directory with position files
        episode_dir = tmp_path / "episode_001"
        episode_dir.mkdir()

        # Create position files (x, y, z per frame)
        # Format: *_position_*.txt with numpy array string
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # Moving forward
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Sudden jump - anomaly!
            [11.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
        ]

        for i, pos in enumerate(positions):
            # Pattern: 0001_next_frame_position_at_current_camera.txt
            frame_file = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            frame_file.write_text(f"[{pos[0]}  {pos[1]}  {pos[2]}]")

        config = PipelineConfig(
            dataset_path=str(tmp_path),
            anomaly_threshold=2.0
        )

        detector = EventDetector(config)
        events = detector.detect("episode_001")

        # Should detect anomaly at the sudden jump
        for event in events:
            assert event.episode_id == "episode_001"
            assert event.action_type in ["hard_brake", "acceleration"]


@pytest.mark.smoke
class TestProgressTrackerIntegration:
    """Test ProgressTracker file operations."""

    def test_progress_tracker_save_load(self, tmp_path):
        """Test progress tracker saves and loads correctly."""
        progress_file = tmp_path / "progress.json"

        tracker = ProgressTracker(progress_file)

        # Initially no events processed
        assert not tracker.is_processed("ep001", "frame_001")
        assert tracker.total_processed == 0

        # Mark some events as processed
        tracker.mark_processed_sync("ep001", "frame_001")
        tracker.mark_processed_sync("ep001", "frame_002")

        assert tracker.is_processed("ep001", "frame_001")
        assert tracker.is_processed("ep001", "frame_002")
        assert not tracker.is_processed("ep001", "frame_003")
        assert tracker.total_processed == 2

        # Create new tracker - should load previous progress
        tracker2 = ProgressTracker(progress_file)
        assert tracker2.is_processed("ep001", "frame_001")
        assert tracker2.is_processed("ep001", "frame_002")
        assert tracker2.total_processed == 2


@pytest.mark.smoke
class TestFrameExtractorIntegration:
    """Test FrameExtractor with mocked video files."""

    def test_history_frames_no_leakage(self, tmp_path):
        """Test that history frames never include event or later frames."""
        # Create mock video directory
        video_dir = tmp_path / "episode_001"
        video_dir.mkdir()

        # Create 100 frame files
        for i in range(100):
            frame_file = video_dir / f"frame_{i:04d}.jpg"
            frame_file.write_text(f"fake image {i}")

        extractor = VideoFrameExtractor(str(tmp_path))

        # Get history frames for event at frame 50
        history = extractor.get_history_frames("episode_001", "frame_0050", count=5)

        # Verify no frame >= 50
        for frame_path in history:
            frame_num = int(Path(frame_path).stem.split("_")[1])
            assert frame_num < 50, f"History frame {frame_num} >= event 50"

    def test_future_frames_no_leakage(self, tmp_path):
        """Test that future frames never include event or earlier frames."""
        # Create mock video directory
        video_dir = tmp_path / "episode_001"
        video_dir.mkdir()

        # Create 100 frame files
        for i in range(100):
            frame_file = video_dir / f"frame_{i:04d}.jpg"
            frame_file.write_text(f"fake image {i}")

        extractor = VideoFrameExtractor(str(tmp_path))

        # Get future frames for event at frame 50
        future = extractor.get_future_frames("episode_001", "frame_0050", count=5)

        # Verify no frame <= 50
        for frame_path in future:
            frame_num = int(Path(frame_path).stem.split("_")[1])
            assert frame_num > 50, f"Future frame {frame_num} <= event 50"


class TestCausalChainSerialization:
    """Test CausalChain JSON serialization."""

    def test_causal_chain_json_roundtrip(self):
        """Test CausalChain can be serialized and deserialized."""
        key_obj = KeyObject(type="pedestrian", location="crosswalk", threat_level="high")
        analysis = HistoricalAnalysis(
            ego_status="cruising",
            key_objects=[key_obj],
            most_critical_object=key_obj,
            predicted_action="brake",
            reasoning="Pedestrian detected"
        )
        confirmation = FutureConfirmation(
            actual_action="brake",
            action_status="completed",
            related_to_history=True
        )
        chain = CausalChain(
            episode_id="ep001",
            event_frame_id="frame_050",
            confidence=0.85,
            historical_analysis=analysis,
            future_confirmation=confirmation,
            causal_link="Pedestrian caused braking"
        )

        # Serialize to JSON
        json_str = chain.model_dump_json()

        # Deserialize back
        chain2 = CausalChain.model_validate_json(json_str)

        assert chain2.episode_id == "ep001"
        assert chain2.confidence == 0.85
        assert chain2.historical_analysis.ego_status == "cruising"
        assert chain2.future_confirmation.actual_action == "brake"
        assert chain2.causal_link == "Pedestrian caused braking"


class TestConfigLoader:
    """Test configuration loading."""

    def test_load_config_with_env_var(self, tmp_path):
        """Test config loading with environment variable substitution."""
        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vlm:
  base_url: "https://api.test.com"
  api_key: "${TEST_API_KEY}"
  model: "test-model"
pipeline:
  dataset_path: "/data/test"
  videos_path: "/videos/test"
""")

        # Set env var
        os.environ["TEST_API_KEY"] = "secret-key-123"

        try:
            config = load_config(str(config_file))

            assert config.vlm.base_url == "https://api.test.com"
            assert config.vlm.api_key == "secret-key-123"
            assert config.vlm.model == "test-model"
            assert config.pipeline.dataset_path == "/data/test"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_load_config_missing_env_var(self, tmp_path):
        """Test that missing env var raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vlm:
  api_key: "${NONEXISTENT_VAR_12345}"
""")

        with pytest.raises(ValueError, match="Environment variable not set"):
            load_config(str(config_file))