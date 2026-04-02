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

    def _parse_frame_num(self, frame_id: str) -> int:
        """Extract frame number from frame_id.

        Handles: "frame_0057" -> 57
                "0039_next_frame_position_at_current_camera" -> 39
        """
        for part in frame_id.split("_"):
            if part.isdigit():
                return int(part)
        raise ValueError(f"Could not parse frame number from: {frame_id}")

    def _list_episode_frames(self, episode_id: str) -> list[tuple[int, Path]]:
        episode_dir = self.videos_path / episode_id
        frames = []
        for file_path in episode_dir.iterdir():
            if file_path.suffix == ".jpg":
                for part in file_path.stem.split("_"):
                    if part.isdigit():
                        frames.append((int(part), file_path))
                        break
        frames.sort(key=lambda x: x[0])
        return frames

    def get_history_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        event_num = self._parse_frame_num(event_frame_id)
        all_frames = self._list_episode_frames(episode_id)
        history = [(num, path) for num, path in all_frames if num < event_num]
        if len(history) < count:
            raise ValueError(f"Not enough history frames: requested {count}, got {len(history)}")
        result = [str(path) for _, path in history[-count:]]
        result.reverse()
        self._validate_no_leakage(episode_id, event_frame_id, result, is_history=True)
        return result

    def get_future_frames(self, episode_id: str, event_frame_id: str, count: int) -> list[str]:
        event_num = self._parse_frame_num(event_frame_id)
        all_frames = self._list_episode_frames(episode_id)
        future = [(num, path) for num, path in all_frames if num > event_num]
        if len(future) < count:
            raise ValueError(f"Not enough future frames: requested {count}, got {len(future)}")
        result = [str(path) for _, path in future[:count]]
        self._validate_no_leakage(episode_id, event_frame_id, result, is_history=False)
        return result

    def _validate_no_leakage(
        self, episode_id: str, event_frame_id: str, frames: list[str], is_history: bool
    ) -> None:
        """
        Verify that frames don't contain the event frame.

        This is a safety check - should never trigger if implementation is correct.
        """
        event_num = self._parse_frame_num(event_frame_id)

        for frame_path in frames:
            frame_num = self._parse_frame_num(Path(frame_path).stem)

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