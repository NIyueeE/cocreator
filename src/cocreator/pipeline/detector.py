"""Velocity and direction-based anomaly detection for driving scenarios.

Detects anomalous events from trajectory, velocity, and directional data.
Implements event deduplication and clustering for robust detection.
"""

import re
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
    confidence: float
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
                    coords = re.findall(r"-?\d+\.?\d*", line)
                    if len(coords) >= 3:
                        positions.append([float(coords[0]), float(coords[1]), float(coords[2])])
                        frame_ids.append(pos_file.stem)

        return np.array(positions), frame_ids

    def _compute_velocities(self, positions: np.ndarray) -> np.ndarray:
        """Compute velocity magnitude at each step."""
        if len(positions) < 2:
            return np.array([])
        diffs = np.diff(positions, axis=0)
        return np.linalg.norm(diffs, axis=1)

    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration (change in velocity)."""
        if len(velocities) < 2:
            return np.array([])
        return np.diff(velocities)

    def _compute_direction_changes(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute direction changes (steering angles) between consecutive segments.

        Returns:
            Array of angular changes in degrees
        """
        if len(positions) < 3:
            return np.array([])

        # Compute direction vectors (velocity vectors)
        directions = np.diff(positions, axis=0)  # (n-1, 3)

        # Compute angles between consecutive direction vectors
        angles = []
        for i in range(len(directions) - 1):
            v1 = directions[i]
            v2 = directions[i + 1]

            # Compute angle between vectors
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                # Vehicle is stationary, no meaningful direction
                angles.append(0.0)
                continue

            # Cosine of angle: cos(θ) = (v1·v2) / (|v1||v2|)
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

        return np.array(angles)

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

        # Compute adaptive threshold for acceleration
        if len(velocities) > 0:
            mu = np.mean(velocities)
            sigma = np.std(velocities)
            acc_threshold = mu + self.threshold * sigma
        else:
            acc_threshold = float('inf')

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

                confidence = min(0.99, abs_acc / acc_threshold)

                # Frame index corresponds to i+1 (acceleration at transition)
                frame_idx = i + 1
                frame_id = frame_ids[frame_idx] if frame_idx < len(frame_ids) else f"frame_{frame_idx:04d}"

                events.append(RawEvent(
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    frame_id=frame_id,
                    action_type=action_type,
                    confidence=float(confidence),
                    event_type=event_type,
                ))

        # Detect steering events
        for i in range(len(direction_changes)):
            angle = direction_changes[i]

            if angle > self.steering_threshold:
                confidence = min(0.99, angle / (self.steering_threshold * 2))

                # Frame index corresponds to i+1 (direction change at transition)
                frame_idx = i + 1
                frame_id = frame_ids[frame_idx] if frame_idx < len(frame_ids) else f"frame_{frame_idx:04d}"

                events.append(RawEvent(
                    episode_id=episode_id,
                    frame_idx=frame_idx,
                    frame_id=frame_id,
                    action_type="steering",
                    confidence=float(confidence),
                    event_type="steering",
                ))

        # Sort by frame index
        events.sort(key=lambda e: e.frame_idx)
        return events

    def _deduplicate_events(self, events: list[RawEvent]) -> list[DetectedEvent]:
        """
        Remove events that are too close to each other (within min_interval).
        Keep the event with highest confidence in each group.
        """
        if not events:
            return []

        filtered = []
        last_event_idx = -self.min_interval - 1

        for event in events:
            if event.frame_idx - last_event_idx >= self.min_interval:
                filtered.append(event)
                last_event_idx = event.frame_idx
            else:
                # Too close to previous event, keep the one with higher confidence
                if filtered and event.confidence > filtered[-1].confidence:
                    filtered[-1] = event

        return [
            DetectedEvent(
                episode_id=event.episode_id,
                frame_id=event.frame_id,
                action_type=event.action_type,
                confidence=event.confidence,
            )
            for event in filtered
        ]

    def _cluster_events(self, events: list[RawEvent]) -> list[DetectedEvent]:
        """
        Cluster adjacent events into merged events.

        Events within min_interval frames are merged into a single event:
        - action_type: prioritized as hard_brake > steering > acceleration
        - confidence: maximum of all events in cluster
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

        # Select action type with highest priority
        best_event = max(cluster, key=lambda e: (priority.get(e.action_type, 0), e.confidence))

        # Compute max confidence in cluster
        max_confidence = max(e.confidence for e in cluster)

        # Use first event's frame_id
        first_event = cluster[0]

        return DetectedEvent(
            episode_id=first_event.episode_id,
            frame_id=first_event.frame_id,
            action_type=best_event.action_type,
            confidence=max_confidence,
        )


__all__ = ["EventDetector"]
