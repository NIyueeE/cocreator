"""Tests for CausalReasoner.

Covers format helpers and multi-turn Stage 2 flow.
"""
from cocreator.schemas import HistoricalAnalysis, FutureConfirmation, DetectedEvent


class TestCausalReasonerLogic:
    def test_format_history_as_assistant(self):
        """Assistant reply should combine description and prediction."""
        from cocreator.pipeline.reasoner import CausalReasoner
        from unittest.mock import MagicMock

        reasoner = CausalReasoner(
            provider=MagicMock(),
            extractor=MagicMock(),
            pipeline_config=MagicMock(),
        )
        history = HistoricalAnalysis(
            description_text="I am driving in the left lane approaching an intersection. "
                             "A pedestrian is waiting at the crosswalk ahead.",
            predict_action="hard_brake",
        )
        text = reasoner._format_history_as_assistant(history)
        assert "pedestrian" in text
        assert "intend to hard_brake" in text
