"""Frame extraction for temporal context.

Extracts historical and future frames around events for analysis.
Enforces strict temporal isolation between historical and future frames.
"""

from pathlib import Path
from typing import Protocol

__all__ = ["FrameExtractor", "VideoFrameExtractor"]


class FrameExtractor(Protocol):
    """
    Protocol for frame extraction with temporal isolation.

    Historical frames (before event) and future frames (after event)
    must be strictly separated - no data leakage allowed.
    """

    def get_history_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        """
        Get frames BEFORE the event frame.

        Args:
            episode_id: Episode identifier
            event_frame_id: The event frame (e.g., "frame_0057")
            count: Number of history frames to retrieve

        Returns:
            List of absolute file paths to JPEG images

        Raises:
            ValueError: If requested frames don't exist
        """
        ...

    def get_future_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        """
        Get frames AFTER the event frame.

        Args:
            episode_id: Episode identifier
            event_frame_id: The event frame (e.g., "frame_0057")
            count: Number of future frames to retrieve

        Returns:
            List of absolute file paths to JPEG images

        Raises:
            ValueError: If requested frames don't exist
        """
        ...


class VideoFrameExtractor(FrameExtractor):
    """
    Concrete implementation for video-based frame extraction.

    Videos are stored in: {videos_path}/{episode_id}/frame_{id}.jpg
    """

    def __init__(self, videos_path: str):
        self.videos_path = Path(videos_path)

    def _get_frame_path(self, episode_id: str, frame_id: str) -> Path:
        """Get path for a specific frame."""
        return self.videos_path / episode_id / f"{frame_id}.jpg"

    def get_history_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        """
        Get 'count' frames BEFORE the event frame.

        Example: If event_frame_id="frame_0057" and count=3:
        Returns frames: frame_0051, frame_0053, frame_0055
        (odd frames before 57)

        Raises:
            ValueError: If not enough history frames exist
        """
        # Parse event frame number
        prefix, num_str = event_frame_id.rsplit("_", 1)
        event_num = int(num_str)

        history_frames = []
        frame_num = event_num - 1  # Start from frame before event

        while len(history_frames) < count and frame_num >= 0:
            frame_id = f"{prefix}_{frame_num:04d}"
            frame_path = self._get_frame_path(episode_id, frame_id)

            if frame_path.exists():
                history_frames.append(str(frame_path))
            # Skip even frames to get every other frame (like the spec shows)
            frame_num -= 2

        if len(history_frames) < count:
            # Fall back to consecutive frames if odd frames don't exist
            frame_num = event_num - 1
            while len(history_frames) < count and frame_num >= 0:
                frame_id = f"{prefix}_{frame_num:04d}"
                frame_path = self._get_frame_path(episode_id, frame_id)
                # Check if this frame is not the event frame and not already collected
                if frame_id != event_frame_id and str(frame_path) not in history_frames:
                    if frame_path.exists():
                        history_frames.append(str(frame_path))
                frame_num -= 1

        if len(history_frames) < count:
            raise ValueError(
                f"Not enough history frames for episode {episode_id}: "
                f"requested {count}, got {len(history_frames)}"
            )

        history_frames.reverse()  # Oldest first
        self._validate_no_leakage(episode_id, event_frame_id, history_frames, is_history=True)
        return history_frames

    def get_future_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        """
        Get 'count' frames AFTER the event frame.

        Example: If event_frame_id="frame_0057" and count=2:
        Returns frames: frame_0059, frame_0061
        (odd frames after 57)

        Raises:
            ValueError: If not enough future frames exist
        """
        # Parse event frame number
        prefix, num_str = event_frame_id.rsplit("_", 1)
        event_num = int(num_str)

        future_frames = []
        frame_num = event_num + 1  # Start from frame after event

        while len(future_frames) < count:
            frame_id = f"{prefix}_{frame_num:04d}"
            frame_path = self._get_frame_path(episode_id, frame_id)

            if frame_path.exists():
                future_frames.append(str(frame_path))
            else:
                # Frame doesn't exist, try next one
                pass

            frame_num += 2  # Skip even frames to get every other frame

        if len(future_frames) < count:
            # Fall back to consecutive frames if odd frames don't exist
            frame_num = event_num + 1
            while len(future_frames) < count:
                frame_id = f"{prefix}_{frame_num:04d}"
                frame_path = self._get_frame_path(episode_id, frame_id)
                if frame_id != event_frame_id and str(frame_path) not in future_frames:
                    if frame_path.exists():
                        future_frames.append(str(frame_path))
                frame_num += 1

        if len(future_frames) < count:
            raise ValueError(
                f"Not enough future frames for episode {episode_id}: "
                f"requested {count}, got {len(future_frames)}"
            )

        self._validate_no_leakage(episode_id, event_frame_id, future_frames, is_history=False)
        return future_frames

    def _validate_no_leakage(
        self, episode_id: str, event_frame_id: str, frames: list[str], is_history: bool
    ) -> None:
        """
        Verify that frames don't contain the event frame.

        This is a safety check - should never trigger if implementation is correct.
        """
        prefix, num_str = event_frame_id.rsplit("_", 1)
        event_num = int(num_str)

        for frame_path in frames:
            frame_name = Path(frame_path).stem
            prefix2, num_str2 = frame_name.rsplit("_", 1)
            frame_num = int(num_str2)

            if is_history:
                assert frame_num < event_num, (
                    f"History frame {frame_num} >= event {event_num}. "
                    f"Data isolation violation!"
                )
            else:
                assert frame_num > event_num, (
                    f"Future frame {frame_num} <= event {event_num}. "
                    f"Data isolation violation!"
                )