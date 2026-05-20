"""Tests for CausalReasoner.

Covers format helpers and multi-turn Stage 2 flow.
"""
from cocreator.schemas import HistoricalAnalysis, FutureConfirmation, KeyObject, DetectedEvent


class TestCausalReasonerLogic:
    def test_format_history_as_assistant(self):
        """Assistant reply should be first-person with key objects."""
        from cocreator.pipeline.reasoner import CausalReasoner
        from unittest.mock import MagicMock

        reasoner = CausalReasoner(
            provider=MagicMock(),
            extractor=MagicMock(),
            pipeline_config=MagicMock(),
        )
        obj = KeyObject(type="pedestrian", location="crosswalk", threat_level="high")
        history = HistoricalAnalysis(
            ego_status="cruising",
            key_objects=[obj],
            most_critical_object=obj,
            predicted_action="brake",
            reasoning="Pedestrian detected on crosswalk.",
        )
        text = reasoner._format_history_as_assistant(history)
        assert "I see the ego vehicle is cruising" in text
        assert "pedestrian at crosswalk" in text
        assert "I predict I will brake" in text
        # reasoning should NOT be in assistant text (snake_case noise)
        assert "Pedestrian detected" not in text
