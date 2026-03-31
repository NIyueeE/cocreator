"""VLM-based causal reasoning for driving events.

Two-stage analysis: history analysis + future confirmation.
Strict data isolation enforced between stages.
"""

import json
from jinja2 import Environment, FileSystemLoader
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

        # Load Jinja2 templates
        prompts_dir = Path(__file__).parent.parent / "prompts"
        env = Environment(loader=FileSystemLoader(prompts_dir))
        self.history_template = env.get_template("history_analysis.j2")
        self.future_template = env.get_template("future_confirmation.j2")

    async def reason(self, episode_id: str, event: DetectedEvent) -> CausalChain:
        """
        Perform 2-stage causal reasoning for a detected event.

        Args:
            episode_id: Episode identifier
            event: The detected event

        Returns:
            CausalChain with historical analysis, future confirmation, and causal link
        """
        # Stage 1: Historical Analysis (frames BEFORE event)
        history_result = await self._analyze_history(episode_id, event)

        # Stage 2: Future Confirmation (frames AFTER event)
        future_result = await self._confirm_future(episode_id, event, history_result)

        # Compute overall confidence
        confidence = self._compute_confidence(history_result, future_result)

        # Generate causal link text
        causal_link = self._generate_causal_link(history_result, future_result)

        return CausalChain(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            confidence=confidence,
            historical_analysis=history_result,
            future_confirmation=future_result,
            causal_link=causal_link,
        )

    async def _analyze_history(
        self, episode_id: str, event: DetectedEvent
    ) -> HistoricalAnalysis:
        """
        Stage 1: Analyze historical frames (BEFORE event).

        STRICT: This method only looks at frames before event_frame_id.
        """
        # Get frames BEFORE event (strict isolation)
        history_frames = self.extractor.get_history_frames(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            count=self.pipeline_config.frames_per_segment
            * self.pipeline_config.history_segments,
        )

        # Build prompt
        prompt = self.history_template.render(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            predicted_action=event.action_type,
        )

        # System message
        system_message = {
            "role": "system",
            "content": "You are a driving scenario analyst. Analyze the provided frames and respond with ONLY valid JSON.",
        }

        # User message with prompt
        user_message = {
            "role": "user",
            "content": prompt + "\n\nPlease analyze the following frames:",
        }

        # Call VLM with images
        response = await self.provider.chat_with_images(
            image_paths=history_frames,
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        result = json.loads(response)

        # Build HistoricalAnalysis
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
        episode_id: str,
        event: DetectedEvent,
        history_result: HistoricalAnalysis,
    ) -> FutureConfirmation:
        """
        Stage 2: Confirm with future frames (AFTER event).

        STRICT: This method only looks at frames after event_frame_id.
        Does NOT receive history frames or the actual event frame.
        """
        # Get frames AFTER event (strict isolation)
        future_frames = self.extractor.get_future_frames(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            count=self.pipeline_config.frames_per_segment
            * self.pipeline_config.future_segments,
        )

        # Build prompt with history summary
        prompt = self.future_template.render(
            episode_id=episode_id,
            event_frame_id=event.frame_id,
            predicted_action=history_result.predicted_action,
            historical_reasoning=history_result.reasoning,
        )

        # System message
        system_message = {
            "role": "system",
            "content": "You are a driving scenario verifier. Analyze the provided frames and respond with ONLY valid JSON.",
        }

        # User message with prompt
        user_message = {
            "role": "user",
            "content": prompt + "\n\nPlease verify the following frames:",
        }

        # Call VLM with images
        response = await self.provider.chat_with_images(
            image_paths=future_frames,
            messages=[system_message, user_message],
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        result = json.loads(response)

        return FutureConfirmation(
            actual_action=result.get("actual_action", "none"),
            action_status=result.get("action_status", "unknown"),
            related_to_history=result.get("related_to_history", False),
            verification_notes=result.get("verification_notes", ""),
        )

    def _compute_confidence(
        self,
        history: HistoricalAnalysis,
        future: FutureConfirmation,
    ) -> float:
        """Compute overall confidence from both stages."""
        # Base confidence from history
        history_confidence = 0.7 if history.predicted_action else 0.5

        # Adjustment based on future confirmation
        if future.related_to_history:
            future_confidence = 0.3
        else:
            future_confidence = -0.2  # Penalty for mismatch

        return min(0.99, max(0.0, history_confidence + future_confidence))

    def _generate_causal_link(
        self,
        history: HistoricalAnalysis,
        future: FutureConfirmation,
    ) -> str:
        """Generate natural language causal link summary."""
        critical_obj = history.most_critical_object
        if critical_obj:
            threat_level = critical_obj.threat_level
            obj_desc = f"{critical_obj.type} at {critical_obj.location}"
        else:
            threat_level = "unknown"
            obj_desc = "the driving scenario"

        action_desc = future.actual_action.replace("_", " ")

        if future.related_to_history:
            return (
                f"Because {obj_desc} {threat_level} threat level, "
                f"the ego vehicle predicted {history.predicted_action} "
                f"and actually executed {action_desc}."
            )
        else:
            return (
                f"The ego vehicle predicted {history.predicted_action} due to {obj_desc}, "
                f"but actually executed {action_desc} - prediction mismatch."
            )


__all__ = ["CausalReasoner"]