"""Tests for cocreator schemas."""

import pytest
from pydantic import ValidationError

from cocreator.schemas import (
    AppConfig,
    CausalChain,
    DetectedEvent,
    FutureConfirmation,
    HistoricalAnalysis,
    PipelineConfig,
    RateLimitConfig,
    VLMConfig,
)


def test_vlm_config_defaults():
    config = VLMConfig()
    assert config.base_url == "https://api.siliconflow.cn"
    assert config.model == "Qwen/Qwen3.5-397B-A17B"


def test_vlm_config_custom():
    config = VLMConfig(base_url="https://custom.api.com", api_key="test-key")
    assert config.base_url == "https://custom.api.com"
    assert config.api_key == "test-key"


def test_rate_limit_defaults():
    config = RateLimitConfig()
    assert config.rpm == 500
    assert config.concurrency == 20


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.anomaly_threshold == 2.0
    assert config.history_frames == 7


def test_pipeline_config_custom():
    config = PipelineConfig(dataset_path="/data/test", videos_path="/videos/test")
    assert config.dataset_path == "/data/test"
    assert config.videos_path == "/videos/test"


def test_detected_event_creation():
    event = DetectedEvent(
        episode_id="ep001", frame_id="frame_100", action_type="hard_brake"
    )
    assert event.episode_id == "ep001"
    assert event.action_type == "hard_brake"


def test_detected_event_json_serialization():
    event = DetectedEvent(
        episode_id="ep001", frame_id="frame_100", action_type="hard_brake"
    )
    json_str = event.model_dump_json()
    assert "ep001" in json_str
    assert "hard_brake" in json_str


def test_detected_event_no_extra_fields():
    with pytest.raises(ValidationError):
        DetectedEvent(
            episode_id="ep001", frame_id="frame_100", unknown_field="value"
        )


def test_historical_analysis_creation():
    analysis = HistoricalAnalysis(
        description_text="I am driving in the left lane approaching an intersection. "
                         "A pedestrian is waiting at the crosswalk ahead.",
        predict_action="hard_brake",
    )
    assert "pedestrian" in analysis.description_text
    assert analysis.predict_action == "hard_brake"


def test_future_confirmation_creation():
    confirmation = FutureConfirmation(
        causal_text="The ego vehicle braked due to a pedestrian at crosswalk.",
    )
    assert confirmation.causal_text


def test_causal_chain_creation():
    chain = CausalChain(
        episode_id="ep001",
        event_frame_id="frame_100",
        frame_ids=["0090", "0095", "0105", "0110"],
        action_type="hard_brake",
        causal_text="The ego vehicle was cruising. It predicted brake.",
    )
    assert chain.episode_id == "ep001"
    assert chain.frame_ids == ["0090", "0095", "0105", "0110"]
    assert chain.action_type == "hard_brake"
    assert chain.causal_text


def test_causal_chain_json_serialization():
    chain = CausalChain(
        episode_id="ep001",
        event_frame_id="frame_100",
        frame_ids=["0090", "0095"],
        action_type="acceleration",
        causal_text="The ego vehicle was cruising.",
    )
    json_str = chain.model_dump_json()
    assert "ep001" in json_str
    assert "action_type" in json_str
    assert "cruising" in json_str
    assert "0090" in json_str


def test_app_config_defaults():
    config = AppConfig()
    assert config.vlm.base_url == "https://api.siliconflow.cn"
    assert config.pipeline.anomaly_threshold == 2.0