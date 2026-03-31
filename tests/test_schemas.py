"""Tests for cocreator schemas."""

import pytest
from pydantic import ValidationError

from cocreator.schemas import (
    AppConfig,
    CausalChain,
    DetectedEvent,
    FutureConfirmation,
    HistoricalAnalysis,
    KeyObject,
    PipelineConfig,
    RateLimitConfig,
    VLMConfig,
)


def test_vlm_config_defaults():
    config = VLMConfig()
    assert config.base_url == "https://api.siliconflow.cn"
    assert config.model == "Qwen/Qwen2.5-72B-Instruct"


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
    assert config.history_segments == 3


def test_pipeline_config_custom():
    config = PipelineConfig(dataset_path="/data/test", videos_path="/videos/test")
    assert config.dataset_path == "/data/test"
    assert config.videos_path == "/videos/test"


def test_detected_event_creation():
    event = DetectedEvent(
        episode_id="ep001", frame_id="frame_100", action_type="hard_brake", confidence=0.95
    )
    assert event.episode_id == "ep001"
    assert event.action_type == "hard_brake"
    assert event.confidence == 0.95


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


def test_key_object_creation():
    obj = KeyObject(type="pedestrian", location="crosswalk", threat_level="high")
    assert obj.type == "pedestrian"
    assert obj.threat_level == "high"


def test_historical_analysis_creation():
    obj = KeyObject(type="pedestrian", location="crosswalk", threat_level="high")
    analysis = HistoricalAnalysis(
        ego_status="cruising",
        key_objects=[obj],
        most_critical_object=obj,
        predicted_action="brake",
        reasoning="Pedestrian detected at crosswalk",
    )
    assert analysis.ego_status == "cruising"
    assert len(analysis.key_objects) == 1
    assert analysis.predicted_action == "brake"


def test_future_confirmation_creation():
    confirmation = FutureConfirmation(
        actual_action="brake",
        action_status="completed",
        related_to_history=True,
    )
    assert confirmation.actual_action == "brake"
    assert confirmation.related_to_history is True


def test_causal_chain_creation():
    obj = KeyObject(type="vehicle", location="ahead", threat_level="high")
    analysis = HistoricalAnalysis(
        ego_status="cruising", key_objects=[obj], predicted_action="brake"
    )
    confirmation = FutureConfirmation(
        actual_action="brake", action_status="completed", related_to_history=True
    )
    chain = CausalChain(
        episode_id="ep001",
        event_frame_id="frame_100",
        confidence=0.85,
        historical_analysis=analysis,
        future_confirmation=confirmation,
        causal_link="Vehicle ahead caused braking",
    )
    assert chain.episode_id == "ep001"
    assert chain.confidence == 0.85
    assert chain.causal_link == "Vehicle ahead caused braking"


def test_causal_chain_json_serialization():
    obj = KeyObject(type="vehicle", location="ahead", threat_level="high")
    analysis = HistoricalAnalysis(
        ego_status="cruising", key_objects=[obj], predicted_action="brake"
    )
    confirmation = FutureConfirmation(
        actual_action="brake", action_status="completed", related_to_history=True
    )
    chain = CausalChain(
        episode_id="ep001",
        event_frame_id="frame_100",
        confidence=0.85,
        historical_analysis=analysis,
        future_confirmation=confirmation,
        causal_link="Vehicle ahead caused braking",
    )
    json_str = chain.model_dump_json()
    assert "ep001" in json_str
    assert "cruising" in json_str
    assert "brake" in json_str


def test_app_config_defaults():
    config = AppConfig()
    assert config.vlm.base_url == "https://api.siliconflow.cn"
    assert config.pipeline.anomaly_threshold == 2.0