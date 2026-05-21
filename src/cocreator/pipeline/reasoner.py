"""VLM-based causal reasoning for driving events.

Two-stage analysis: history analysis + future confirmation.
Strict data isolation enforced between stages.
"""

import json
from pathlib import Path

from ..schemas import (
    DetectedEvent,
    CausalChain,
    HistoricalAnalysis,
    FutureConfirmation,
    KeyObject,
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
                "ego_status": {
                    "type": "string",
                    "enum": [
                        "cruising",
                        "accelerating",
                        "braking",
                        "turning",
                        "stopped",
                        "lane_changing",
                    ],
                },
                "key_objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "pedestrian",
                                    "vehicle",
                                    "traffic_light",
                                    "cyclist",
                                    "animal",
                                    "other",
                                ],
                            },
                            "location": {
                                "type": "string",
                                "enum": [
                                    "ahead",
                                    "behind",
                                    "left",
                                    "right",
                                    "intersection",
                                    "crosswalk",
                                    "road_edge",
                                ],
                            },
                            "threat_level": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                        "required": ["type", "location", "threat_level"],
                        "additionalProperties": False,
                    },
                },
                "most_critical_object": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "pedestrian",
                                        "vehicle",
                                        "traffic_light",
                                        "cyclist",
                                        "animal",
                                        "other",
                                    ],
                                },
                                "location": {
                                    "type": "string",
                                    "enum": [
                                        "ahead",
                                        "behind",
                                        "left",
                                        "right",
                                        "intersection",
                                        "crosswalk",
                                        "road_edge",
                                    ],
                                },
                                "threat_level": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                            },
                            "required": ["type", "location", "threat_level"],
                            "additionalProperties": False,
                        },
                        {"type": "null"},
                    ],
                },
                "predicted_action": {
                    "type": "string",
                    "enum": [
                        "brake",
                        "accelerate",
                        "lane_change_left",
                        "lane_change_right",
                        "maintain_speed",
                        "stop",
                    ],
                },
                "reasoning": {"type": "string"},
            },
            "required": [
                "ego_status",
                "key_objects",
                "most_critical_object",
                "predicted_action",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}

_FUTURE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "future_confirmation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "causal_text": {
                    "type": "string",
                    "description": "Complete causal description combining what was predicted and what actually happened.",
                },
                "actual_action": {
                    "type": "string",
                    "enum": [
                        "brake",
                        "accelerate",
                        "lane_change_left",
                        "lane_change_right",
                        "maintain_speed",
                        "stop",
                        "none",
                    ],
                },
                "action_status": {
                    "type": "string",
                    "enum": ["completed", "in_progress", "aborted", "none"],
                },
                "related_to_history": {"type": "boolean"},
                "verification_notes": {"type": "string"},
            },
            "required": [
                "causal_text",
                "actual_action",
                "action_status",
                "related_to_history",
                "verification_notes",
            ],
            "additionalProperties": False,
        },
    },
}

_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are the driver of this vehicle. You see the road ahead, the traffic around you, "
        "and you make decisions based on what you observe. Your task is to explain what you saw, "
        "what you did, and why."
    ),
}


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

        # Stage 2: Future Confirmation (frames AFTER event) + VLM-produced causal_text
        future_result, causal_text = await self._confirm_future(
            future_paths, episode_id, event, history_result
        )

        # Build final output
        frame_ids = self._build_frame_ids(history_paths, future_paths)

        return CausalChain(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            frame_ids=frame_ids,
            causal_text=causal_text,
        )

    async def _analyze_history(
        self, frame_paths: list[str], episode_id: str, event: DetectedEvent
    ) -> HistoricalAnalysis:
        """
        Stage 1: Analyze historical frames (BEFORE event).

        STRICT: This method only looks at frames before event_frame_id.
        """
        user_message = {
            "role": "user",
            "content": "Look at the road ahead. What do you see and what do you do?",
        }

        response = await self.provider.chat_with_images(
            image_paths=frame_paths,
            messages=[_SYSTEM_MESSAGE, user_message],
            response_format=_HISTORY_SCHEMA,
        )

        result = json.loads(response)
        key_objects = [
            KeyObject(
                type=o.get("type", "unknown"),
                location=o.get("location", "unknown"),
                threat_level=o.get("threat_level", "low"),
            )
            for o in result.get("key_objects", [])
        ]

        most_critical = result.get("most_critical_object")
        if most_critical:
            most_critical = KeyObject(
                type=most_critical.get("type", "unknown"),
                location=most_critical.get("location", "unknown"),
                threat_level=most_critical.get("threat_level", "low"),
            )

        return HistoricalAnalysis(
            ego_status=result.get("ego_status", "unknown"),
            key_objects=key_objects,
            most_critical_object=most_critical,
            predicted_action=result.get("predicted_action", event.action_type),
            reasoning=result.get("reasoning", ""),
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
            "content": self._format_history_as_assistant(history_result),
        }
        stage2_user = {
            "role": "user",
            "content": (
                "Now look at the future frames. What actually happened? "
                "Compare with your earlier understanding and write a complete "
                "causal description of the driving scene."
            ),
        }

        response = await self.provider.chat_with_images(
            image_paths=frame_paths,
            messages=[_SYSTEM_MESSAGE, stage1_assistant, stage2_user],
            response_format=_FUTURE_SCHEMA,
        )

        result = json.loads(response)
        causal_text = result.get("causal_text", "")
        return FutureConfirmation(
            actual_action=result.get("actual_action", "none"),
            action_status=result.get("action_status", "unknown"),
            related_to_history=result.get("related_to_history", False),
            verification_notes=result.get("verification_notes", ""),
        ), causal_text

    def _format_history_as_assistant(self, history: HistoricalAnalysis) -> str:
        """Format Stage 1 analysis as a first-person assistant response."""
        parts = [f"I see the ego vehicle is {history.ego_status}."]
        if history.key_objects:
            obj_descs = [
                f"{o.type} at {o.location} ({o.threat_level} threat)"
                for o in history.key_objects
            ]
            parts.append("Key objects: " + ", ".join(obj_descs) + ".")
        if history.most_critical_object:
            o = history.most_critical_object
            parts.append(f"The most critical object is {o.type} at {o.location}.")
        parts.append(f"I predict I will {history.predicted_action}.")
        return " ".join(parts)

    def _build_frame_ids(
        self, history_paths: list[str], future_paths: list[str]
    ) -> list[str]:
        """Extract frame numbers from file paths in chronological order."""
        nums = []
        for p in history_paths + future_paths:
            num = self.extractor._parse_frame_num(Path(p).stem)
            nums.append(f"{num:04d}")
        return nums


__all__ = ["CausalReasoner"]
