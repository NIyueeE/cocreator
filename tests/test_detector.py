"""Tests for EventDetector.

Covers anomaly detection, deduplication, and clustering.
"""
import numpy as np
import pytest
from pathlib import Path

from cocreator.schemas import PipelineConfig
from cocreator.pipeline.detector import EventDetector


class TestEventDetector:
    def test_no_events_for_smooth_driving(self, tmp_path):
        """Smooth constant velocity should produce no events."""
        episode_dir = tmp_path / "smooth"
        episode_dir.mkdir()
        for i in range(20):
            f = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            f.write_text(f"[{i*0.1}  0.0  0.0]")

        config = PipelineConfig(dataset_path=str(tmp_path), anomaly_threshold=3.0)
        detector = EventDetector(config)
        events = detector.detect("smooth")
        assert len(events) == 0

    def test_detects_acceleration_anomaly(self, tmp_path):
        """Sudden acceleration should be detected."""
        episode_dir = tmp_path / "accel"
        episode_dir.mkdir()
        # Many small steps with one big jump in the middle
        positions = [[i * 0.1, 0.0, 0.0] for i in range(15)]
        positions[10] = [50.0, 0.0, 0.0]  # sudden jump
        for i in range(11, 15):
            positions[i] = [50.0 + (i - 10) * 0.1, 0.0, 0.0]
        for i, pos in enumerate(positions):
            f = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            f.write_text(f"[{pos[0]}  {pos[1]}  {pos[2]}]")

        config = PipelineConfig(dataset_path=str(tmp_path), anomaly_threshold=1.0)
        detector = EventDetector(config)
        events = detector.detect("accel")
        assert len(events) >= 1
        assert all(e.episode_id == "accel" for e in events)

    def test_insufficient_data_returns_empty(self, tmp_path):
        """Fewer than 3 positions should return no events."""
        episode_dir = tmp_path / "short"
        episode_dir.mkdir()
        for i in range(2):
            f = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            f.write_text(f"[{i}.0  0.0  0.0]")

        config = PipelineConfig(dataset_path=str(tmp_path))
        detector = EventDetector(config)
        assert detector.detect("short") == []

    def test_steering_event_detected(self, tmp_path):
        """Sharp direction change should produce a steering event."""
        episode_dir = tmp_path / "steer"
        episode_dir.mkdir()
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 5.0, 0.0],  # sharp turn
            [2.0, 10.0, 0.0],
        ]
        for i, pos in enumerate(positions):
            f = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            f.write_text(f"[{pos[0]}  {pos[1]}  {pos[2]}]")

        config = PipelineConfig(dataset_path=str(tmp_path), steering_threshold=10.0)
        detector = EventDetector(config)
        events = detector.detect("steer")
        assert len(events) >= 1

    def test_event_deduplication(self, tmp_path):
        """Events within min_interval should be deduplicated."""
        episode_dir = tmp_path / "dedup"
        episode_dir.mkdir()
        positions = [[0.0, 0.0, 0.0]]
        for i in range(1, 20):
            # Alternating normal / jump to produce many candidate events
            if i % 3 == 0:
                positions.append([positions[-1][0] + 5.0, 0.0, 0.0])
            else:
                positions.append([positions[-1][0] + 0.1, 0.0, 0.0])
        for i, pos in enumerate(positions):
            f = episode_dir / f"{i:04d}_next_frame_position_at_current_camera.txt"
            f.write_text(f"[{pos[0]}  {pos[1]}  {pos[2]}]")

        config = PipelineConfig(
            dataset_path=str(tmp_path),
            anomaly_threshold=2.0,
            min_event_interval=3,
        )
        detector = EventDetector(config)
        events = detector.detect("dedup")
        # Without dedup we'd have many events; dedup should reduce count
        assert len(events) < 5
