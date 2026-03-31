"""Velocity-based anomaly detection for driving scenarios.

Detects anomalous events from trajectory/velocity data.
"""

import re
from pathlib import Path

import numpy as np

from ..schemas import DetectedEvent, PipelineConfig


class EventDetector:
    """
    Detects driving events based on velocity anomalies.

    Reads position data from action_info directory and detects
    significant speed changes that indicate driving maneuvers.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.threshold = config.anomaly_threshold

    def detect(self, episode_id: str) -> list[DetectedEvent]:
        """
        Detect events in a single episode.

        Args:
            episode_id: Episode identifier (e.g., "061613_s20-147_1704411595.0_1704411615.0")

        Returns:
            List of DetectedEvent objects
        """
        # Load position data
        positions = self._load_positions(episode_id)

        # Compute velocities and accelerations
        velocities = self._compute_velocities(positions)
        accelerations = self._compute_accelerations(velocities)

        # Detect anomalies
        events = self._detect_anomalies(episode_id, positions, velocities, accelerations)

        return events

    def _load_positions(self, episode_id: str) -> np.ndarray:
        """
        Load position vectors from action_info/{episode_id}/*_position_*.txt

        Each file contains one frame's position as numpy array string.
        Returns shape (n_frames, 3)
        """
        episode_dir = Path(self.config.dataset_path) / episode_id
        if not episode_dir.exists():
            raise ValueError(f"Episode directory not found: {episode_dir}")

        # Get all position files and sort by frame number
        # Pattern: XXXX_next_frame_position_at_current_camera.txt
        pos_files = sorted(
            episode_dir.glob("*_position_*.txt"),
            key=lambda f: int(f.stem.split("_")[0]),
        )

        positions = []
        for pos_file in pos_files:
            with open(pos_file, "r") as f:
                line = f.read().strip()
                if line:
                    # Parse numpy array string format: [-0.00091541  0.00032911  0.05035767]
                    coords = re.findall(r"-?\d+\.?\d*", line)
                    if len(coords) >= 3:
                        positions.append([float(coords[0]), float(coords[1]), float(coords[2])])

        return np.array(positions)

    def _compute_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocity magnitude at each step."""
        if len(positions) < 2:
            return np.array([])

        diffs = np.diff(positions, axis=0)
        velocities = np.linalg.norm(diffs, axis=1)
        return velocities

    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration (change in velocity)."""
        if len(velocities) < 2:
            return np.array([])
        return np.diff(velocities)

    def _detect_anomalies(
        self,
        episode_id: str,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> list[DetectedEvent]:
        """
        Detect anomalies based on acceleration threshold.

        Returns list of DetectedEvent with frame_id, action_type, confidence.
        """
        if len(accelerations) == 0:
            return []

        # Compute adaptive threshold
        mu = np.mean(velocities)
        sigma = np.std(velocities)
        tau = mu + self.threshold * sigma

        events = []
        n = len(accelerations)

        # Get frame IDs from positions
        episode_dir = Path(self.config.dataset_path) / episode_id
        pos_files = sorted(
            episode_dir.glob("*_position_*.txt"),
            key=lambda f: int(f.stem.split("_")[0]),
        )

        for i in range(n):
            acc = accelerations[i]
            abs_acc = abs(acc)

            if abs_acc > tau:
                # Determine action type based on acceleration sign
                if acc < 0:
                    action_type = "hard_brake"
                else:
                    action_type = "acceleration"

                # Calculate confidence based on how far above threshold
                confidence = min(0.99, abs_acc / tau)

                # Get frame_id (accelerations[i] corresponds to transition from i to i+1)
                if i + 1 < len(pos_files):
                    frame_id = pos_files[i + 1].stem  # e.g., "0015_next_frame_position_at_current_camera"
                else:
                    frame_id = f"frame_{i + 1:04d}"

                events.append(
                    DetectedEvent(
                        episode_id=episode_id,
                        frame_id=frame_id,
                        action_type=action_type,
                        confidence=float(confidence),
                    )
                )

        return events


__all__ = ["EventDetector"]