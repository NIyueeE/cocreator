"""VLM-based causal reasoning for driving events.

Two-stage analysis: history analysis + future confirmation.
Strict data isolation enforced between stages.
"""

import asyncio
import json
from pathlib import Path

from ..schemas import (
    DetectedEvent,
    CausalChain,
    HistoricalAnalysis,
    FutureConfirmation,
    PipelineConfig,
)
from ..providers.openai_compatible import OpenAICompatibleProvider
from .extractor import VideoFrameExtractor


_HISTORY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "history_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "description_text": {
                    "type": "string",
                    "description": (
                        "Complete narrative description of the driving scene. "
                        "Describe road layout, traffic participants, spatial relationships, "
                        "and current driving state in a coherent paragraph."
                    ),
                },
                "predict_action": {
                    "type": "string",
                    "enum": ["hard_brake", "steering", "acceleration", "maintain"],
                },
            },
            "required": ["description_text", "predict_action"],
            "additionalProperties": False,
        },
    },
}

IDENTITY = (
    "You are the driver of this vehicle, actively operating it in real traffic. "
    "You are not a passenger or external observer. "
    "Images from user are your visual perception — your own eyes looking at the road ahead. "
    "This is your real-time first-person view of the driving environment. "
    "Always respond as the driver, in first person, based on what you see."
)

SYSTEM_PROMPT_HISTORY = (
    f"{IDENTITY}\n\n"
    "Describe your current driving scene in detail, including road layout, "
    "traffic participants, spatial relationships, and your driving state. "
    "Then predict what action you will take next."
)

SYSTEM_PROMPT_FUTURE = (
    f"{IDENTITY}\n\n"
    "Based on the images, describe the driving event. "
    "Respond in order: first the initial driving scene in rich detail"
    " (road layout, traffic participants, spatial relationships), "
    "then what you intended to do and why, "
    "then what actually happened and what action you took, "
    "then why you drove that way. "
    "Be thorough and specific — cover each part with a complete description."
)

SYSTEM_PROMPT_SIMPLE = (
    f"{IDENTITY}\n\n"
    "Based on the images, describe the driving event. "
    "Respond in order: first the initial driving scene in rich detail"
    " (road layout, traffic participants, spatial relationships), "
    "then what actually happened and what action you took, "
    "then why you drove that way. "
    "Be thorough and specific — cover each part with a complete description."
)

_SYSTEM_MESSAGE_HISTORY = {"role": "system", "content": SYSTEM_PROMPT_HISTORY}
_SYSTEM_MESSAGE_FUTURE = {"role": "system", "content": SYSTEM_PROMPT_FUTURE}
_SYSTEM_MESSAGE_SIMPLE = {"role": "system", "content": SYSTEM_PROMPT_SIMPLE}


class CausalReasoner:
    """
    Two-stage VLM causal reasoning with strict data isolation.

    Stage 1 (History): Analyze frames BEFORE the event → predict action
    Stage 2 (Future): Analyze frames AFTER the event → confirm action

    These stages CANNOT share information - strict isolation enforced.
    """

    def __init__(
        self,
        provider: OpenAICompatibleProvider,
        extractor: VideoFrameExtractor,
        pipeline_config: PipelineConfig,
    ):
        self.provider = provider
        self.extractor = extractor
        self.pipeline_config = pipeline_config

    async def reason(self, episode_id: str, event: DetectedEvent) -> CausalChain:
        """
        Perform 2-stage causal reasoning for a detected event.

        Args:
            episode_id: Episode identifier
            event: The detected event

        Returns:
            CausalChain with historical analysis, future confirmation, and causal link
        """
        # Get frames with strict temporal isolation
        history_paths = self.extractor.get_history_frames(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            count=self.pipeline_config.history_frames,
        )
        future_paths = self.extractor.get_future_frames(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            count=self.pipeline_config.future_frames,
        )

        # Stage 1: Historical Analysis (frames BEFORE event)
        history_result = await self._analyze_history(history_paths, episode_id, event)

        # Stage 2 + baseline run concurrently (independent of each other)
        future_task = asyncio.create_task(
            self._confirm_future(future_paths, episode_id, event, history_result)
        )
        simple_task = asyncio.create_task(
            self._analyze_simple(history_paths + future_paths, episode_id, event)
        )

        _, causal_text = await future_task
        simple_text = await simple_task

        # Build final output
        frame_ids = self._build_frame_ids(history_paths, future_paths)

        return CausalChain(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            frame_ids=frame_ids,
            action_type=event.action_type,
            causal_text=causal_text,
            simple_text=simple_text,
        )

    async def _analyze_history(
        self, frame_paths: list[str], episode_id: str, event: DetectedEvent
    ) -> HistoricalAnalysis:
        """
        Stage 1: Analyze historical frames (BEFORE event).

        STRICT: This method only looks at frames before event_frame_id.
        """
        response = await self.provider.chat_with_images(
            image_paths=frame_paths,
            messages=[_SYSTEM_MESSAGE_HISTORY, {"role": "user", "content": ""}],
            response_format=_HISTORY_SCHEMA,
            temperature=0.2,
            max_tokens=16384,
        )

        result = json.loads(response)

        return HistoricalAnalysis(
            description_text=result.get("description_text", ""),
            predict_action=result.get("predict_action", "maintain"),
        )

    async def _confirm_future(
        self,
        frame_paths: list[str],
        episode_id: str,
        event: DetectedEvent,
        history_result: HistoricalAnalysis,
    ) -> tuple[FutureConfirmation, str]:
        """
        Stage 2: Deep causal understanding from future frames.

        STRICT: This method only looks at frames after event_frame_id.
        Does NOT receive history frames or the actual event frame.

        Multi-turn: assistant replies with Stage 1 analysis, then user asks
        for causal understanding combining prediction + actual outcome.
        Returns (FutureConfirmation, causal_text) where causal_text is VLM-produced.
        """
        stage1_assistant = {
            "role": "assistant",
            "content": (
                f"{self._format_history_as_assistant(history_result)}\n"
                f"However, the vehicle actually performed a {event.action_type}."
            ),
        }

        response = await self.provider.chat_with_images(
            image_paths=frame_paths,
            messages=[_SYSTEM_MESSAGE_FUTURE, stage1_assistant, {"role": "user", "content": ""}],
            temperature=0.2,
            max_tokens=16384,
        )

        causal_text = response.strip()
        return FutureConfirmation(causal_text=causal_text), causal_text

    def _format_history_as_assistant(self, history: HistoricalAnalysis) -> str:
        """Format Stage 1 analysis as a first-person assistant response."""
        return (
            f"{history.description_text}\n"
            f"I intend to {history.predict_action}."
        )

    async def _analyze_simple(
        self, frame_paths: list[str], episode_id: str, event: DetectedEvent
    ) -> str:
        """
        Baseline: single-call analysis with ALL frames (no two-stage isolation).

        Takes all frames (history + future) in one call and produces a unified
        description of the driving scene and what happened.
        """
        response = await self.provider.chat_with_images(
            image_paths=frame_paths,
            messages=[_SYSTEM_MESSAGE_SIMPLE, {"role": "user", "content": ""}],
            temperature=0.2,
            max_tokens=16384,
        )

        return response.strip()

    def _build_frame_ids(
        self, history_paths: list[str], future_paths: list[str]
    ) -> list[str]:
        """Extract frame numbers from file paths in chronological order."""
        nums = []
        for p in history_paths + future_paths:
            num = self.extractor._parse_frame_num(Path(p).stem)
            nums.append(f"{num:04d}")
        return nums


__all__ = ["CausalReasoner", "IDENTITY", "SYSTEM_PROMPT_HISTORY", "SYSTEM_PROMPT_FUTURE", "SYSTEM_PROMPT_SIMPLE"]
