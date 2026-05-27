"""Velocity and direction-based anomaly detection for driving scenarios.

Detects anomalous events from trajectory, velocity, and directional data.
Implements event deduplication and clustering for robust detection.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..schemas import DetectedEvent, PipelineConfig


@dataclass
class RawEvent:
    """Internal representation of a detected event before clustering."""

    episode_id: str
    frame_idx: int
    frame_id: str
    action_type: str
    event_type: str  # 'acceleration', 'braking', 'steering'


class EventDetector:
    """
    Detects driving events based on velocity, acceleration, and steering anomalies.

    Reads position data from action_info directory and detects:
    - Hard braking (negative acceleration)
    - Acceleration (positive acceleration)
    - Steering (direction changes)

    Implements event deduplication and clustering for robust detection.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.threshold = config.anomaly_threshold
        self.min_interval = config.min_event_interval
        self.steering_threshold = config.steering_threshold
        self.min_steering_speed = config.min_steering_speed
        self.merge_adjacent = config.merge_adjacent_events

    def detect(self, episode_id: str) -> list[DetectedEvent]:
        """
        Detect events in a single episode.

        Args:
            episode_id: Episode identifier

        Returns:
            List of DetectedEvent objects
        """
        # Load position data
        positions, frame_ids = self._load_positions(episode_id)

        if len(positions) < 3:
            return []

        # Compute motion features
        velocities = self._compute_velocities(positions)
        accelerations = self._compute_accelerations(velocities)
        direction_changes = self._compute_direction_changes(positions)

        # Detect all candidate events
        raw_events = self._detect_all_events(
            episode_id, velocities, accelerations, direction_changes, frame_ids
        )

        # Apply deduplication and clustering
        if self.merge_adjacent:
            events = self._cluster_events(raw_events)
        else:
            events = self._deduplicate_events(raw_events)

        return events

    def _load_positions(self, episode_id: str) -> tuple[np.ndarray, list[str]]:
        """
        Load position vectors and frame IDs.

        Returns:
            Tuple of (positions array, frame_id list)
        """
        episode_dir = Path(self.config.dataset_path) / episode_id
        if not episode_dir.exists():
            raise ValueError(f"Episode directory not found: {episode_dir}")

        pos_files = sorted(
            episode_dir.glob("*_position_*.txt"),
            key=lambda f: int(f.stem.split("_")[0]),
        )

        positions = []
        frame_ids = []
        for pos_file in pos_files:
            with open(pos_file, "r") as f:
                line = f.read().strip()
                if line:
                    # Parse: strip brackets, split by whitespace, convert to float
                    # Handles both decimal (0.015) and scientific notation (1.23e-02)
                    tokens = line.strip("[]").strip().split()
                    if len(tokens) >= 3:
                        positions.append(
                            [float(tokens[0]), float(tokens[1]), float(tokens[2])]
                        )
                        frame_ids.append(pos_file.stem)

        return np.array(positions), frame_ids

    def _compute_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute speed magnitude.

        Position files store ego-motion (frame-to-frame displacement vectors).
        Speed is the magnitude of each displacement vector directly,
        not the diff of positions.
        """
        if len(positions) < 1:
            return np.array([])
        return np.linalg.norm(positions, axis=1)

    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration (change in velocity)."""
        if len(velocities) < 2:
            return np.array([])
        return np.diff(velocities)

    def _compute_direction_changes(self, positions: np.ndarray) -> np.ndarray:
        """
        Detect steering via trajectory curvature analysis.

        Instead of computing noisy frame-to-frame direction angles, this method
        fits a straight line to a sliding window of XY positions and measures
        the deviation from linearity. Real turns produce systematic curvature
        (large residual/path ratio); random noise cancels out over the window.

        Returns:
            Array of curvature-derived angles in degrees (0 = straight).
        """
        if len(positions) < 3:
            return np.array([])

        W = 5  # half-window: fits over 2W+1 = 11 frames

        xy = positions[:, :2]
        n = len(xy)
        result = np.zeros(n - 2)

        for i in range(W, n - W):
            segment = xy[i - W : i + W + 1]
            t = np.arange(len(segment), dtype=float)

            # Fit line: x = a0 + a1*t, y = b0 + b1*t
            A = np.column_stack([np.ones_like(t), t])
            cx, *_ = np.linalg.lstsq(A, segment[:, 0], rcond=None)
            cy, *_ = np.linalg.lstsq(A, segment[:, 1], rcond=None)

            # RMS residual from the fitted straight line
            x_fit = cx[0] + cx[1] * t
            y_fit = cy[0] + cy[1] * t
            residuals = np.sqrt(
                (segment[:, 0] - x_fit) ** 2 + (segment[:, 1] - y_fit) ** 2
            )
            rms_residual = np.sqrt(np.mean(residuals ** 2))

            # Total path length travelled in this window
            deltas = np.diff(segment, axis=0)
            path_len = np.sum(np.linalg.norm(deltas, axis=1))

            # Normalized curvature: rms_residual / path_len
            # A straight line → ratio ≈ 0, a sharp turn → ratio ≫ 0
            ratio = rms_residual / max(path_len, 1e-8)

            # Convert ratio to an angle-like measure (degrees)
            # For a circular arc: chord deviation ~ chord_length * (1 - cos(θ/2))
            # Approximate: angle ≈ 2 * arcsin(ratio)
            angle = np.degrees(2 * np.arcsin(np.clip(ratio, 0, 1)))

            # Map to position i-1 (matching output indexing: direction_changes[i] → frame_idx = i+1)
            result[i - 1] = angle

        return result

    def _detect_all_events(
        self,
        episode_id: str,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        direction_changes: np.ndarray,
        frame_ids: list[str],
    ) -> list[RawEvent]:
        """
        Detect all candidate events (acceleration, braking, steering).

        Returns:
            List of RawEvent objects sorted by frame index
        """
        events = []

        # Compute adaptive threshold from accelerations
        if len(accelerations) > 0:
            acc_mu = np.mean(accelerations)
            acc_sigma = np.std(accelerations)
            acc_threshold = abs(acc_mu) + self.threshold * acc_sigma
        else:
            acc_threshold = float("inf")

        # Detect acceleration/braking events
        for i in range(len(accelerations)):
            acc = accelerations[i]
            abs_acc = abs(acc)

            if abs_acc > acc_threshold:
                # Determine action type based on acceleration sign
                if acc < 0:
                    action_type = "hard_brake"
                    event_type = "braking"
                else:
                    action_type = "acceleration"
                    event_type = "acceleration"

                # Frame index corresponds to i+1 (acceleration at transition)
                frame_idx = i + 1
                frame_id = (
                    frame_ids[frame_idx]
                    if frame_idx < len(frame_ids)
                    else f"frame_{frame_idx:04d}"
                )

                events.append(
                    RawEvent(
                        episode_id=episode_id,
                        frame_idx=frame_idx,
                        frame_id=frame_id,
                        action_type=action_type,
                        event_type=event_type,
                    )
                )

        # Detect steering events
        for i in range(len(direction_changes)):
            angle = direction_changes[i]

            if angle > self.steering_threshold:
                # Frame index corresponds to i+1 (direction change at transition)
                frame_idx = i + 1

                # Skip steering events at very low speed (noise-dominated)
                if frame_idx < len(velocities) and velocities[frame_idx] < self.min_steering_speed:
                    continue

                frame_id = (
                    frame_ids[frame_idx]
                    if frame_idx < len(frame_ids)
                    else f"frame_{frame_idx:04d}"
                )

                events.append(
                    RawEvent(
                        episode_id=episode_id,
                        frame_idx=frame_idx,
                        frame_id=frame_id,
                        action_type="steering",
                        event_type="steering",
                    )
                )

        # Sort by frame index
        events.sort(key=lambda e: e.frame_idx)
        return events

    def _deduplicate_events(self, events: list[RawEvent]) -> list[DetectedEvent]:
        """
        Remove events that are too close to each other (within min_interval).
        """
        if not events:
            return []

        filtered = []
        last_event_idx = -self.min_interval - 1

        for event in events:
            if event.frame_idx - last_event_idx >= self.min_interval:
                filtered.append(event)
                last_event_idx = event.frame_idx

        return [
            DetectedEvent(
                episode_id=event.episode_id,
                frame_id=event.frame_id,
                action_type=event.action_type,
            )
            for event in filtered
        ]

    def _cluster_events(self, events: list[RawEvent]) -> list[DetectedEvent]:
        """
        Cluster adjacent events into merged events.

        Events within min_interval frames are merged into a single event:
        - action_type: prioritized as hard_brake > steering > acceleration
        - frame_id: first event in cluster
        """
        if not events:
            return []

        clusters = []
        current_cluster = [events[0]]

        for event in events[1:]:
            if event.frame_idx - current_cluster[-1].frame_idx <= self.min_interval:
                current_cluster.append(event)
            else:
                clusters.append(current_cluster)
                current_cluster = [event]
        clusters.append(current_cluster)

        # Merge each cluster
        merged_events = []
        for cluster in clusters:
            merged = self._merge_cluster(cluster)
            merged_events.append(merged)

        return merged_events

    def _merge_cluster(self, cluster: list[RawEvent]) -> DetectedEvent:
        """Merge a cluster of events into a single DetectedEvent."""
        # Priority: hard_brake > steering > acceleration
        priority = {"hard_brake": 3, "steering": 2, "acceleration": 1}

        # frame_id = first frame in the cluster (marks the start of the maneuver)
        # action_type = highest priority action in the cluster (most significant event)
        first_event = cluster[0]
        best_event = max(cluster, key=lambda e: priority.get(e.action_type, 0))

        return DetectedEvent(
            episode_id=first_event.episode_id,
            frame_id=first_event.frame_id,
            action_type=best_event.action_type,
        )


__all__ = ["EventDetector"]
