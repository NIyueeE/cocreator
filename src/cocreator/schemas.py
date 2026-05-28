"""Pydantic data models for cocreator pipeline.

Defines all data structures used across the event detection,
causal reasoning, and report generation stages.
"""

from typing import Optional

from pydantic import BaseModel


class VLMConfig(BaseModel):
    """VLM API configuration settings.

    Attributes:
        base_url: Base URL for the VLM API endpoint.
            Note: no /v1 suffix - the provider appends it.
        api_key: API key for authentication.
        model: Model identifier for the VLM.
        timeout: Request timeout in seconds.
        enable_thinking: Enable thinking mode for supported models (e.g., Qwen).
    """

    base_url: str = "https://api.siliconflow.cn"
    api_key: str = ""
    model: str = "Qwen/Qwen3.5-397B-A17B"
    timeout: float = 120.0
    enable_thinking: bool = False


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for API requests.

    Attributes:
        rpm: Requests per minute limit.
        tpm: Tokens per minute limit.
        concurrency: Maximum concurrent requests.
    """

    rpm: int = 500
    tpm: int = 2000000
    concurrency: int = 3


class PipelineConfig(BaseModel):
    """Pipeline-wide configuration settings.

    Attributes:
        dataset_path: Path to the dataset directory.
        videos_path: Path to the videos directory.
        output_dir: Directory for pipeline output files.
        history_frames: Number of historical frames before event for VLM analysis.
        future_frames: Number of future frames after event for VLM confirmation.
        anomaly_threshold: Threshold for anomaly detection.
        retry_max_attempts: Maximum retry attempts on failure.
        retry_backoff_factor: Backoff multiplier for retries.
        min_event_interval: Minimum frame interval between events (deduplication).
        steering_threshold: Angular threshold (degrees) for steering detection.
        merge_adjacent_events: Whether to merge adjacent events within interval.
    """

    dataset_path: str = ""
    videos_path: str = ""
    output_dir: str = "./output"
    history_frames: int = 7
    future_frames: int = 11
    anomaly_threshold: float = 2.0
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    min_event_interval: int = 5
    steering_threshold: float = 13.0
    min_steering_speed: float = 0.5
    merge_adjacent_events: bool = True


class AppConfig(BaseModel):
    """Composite application configuration.

    Attributes:
        vlm: VLM API configuration.
        rate_limit: Rate limiting configuration.
        pipeline: Pipeline configuration.
    """

    vlm: VLMConfig = VLMConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    pipeline: PipelineConfig = PipelineConfig()


class DetectedEvent(BaseModel):
    """Event detection output from the anomaly detector.

    Attributes:
        episode_id: Unique identifier for the driving episode.
        frame_id: Frame identifier where the event was detected.
        action_type: Type of detected action (e.g., "hard_brake", "acceleration").
    """

    episode_id: str
    frame_id: str
    action_type: str

    model_config = {"extra": "forbid"}


class KeyObject(BaseModel):
    """Object of interest identified during historical analysis.

    Attributes:
        type: Object type (e.g., "pedestrian", "vehicle", "traffic_light").
        location: Spatial location (e.g., "crosswalk", "intersection").
        threat_level: Threat assessment (e.g., "high", "medium", "low").
    """

    type: str
    location: str
    threat_level: str


class HistoricalAnalysis(BaseModel):
    """Stage 1 output: Analysis of historical frames leading to event.

    Attributes:
        description_text: Complete narrative description of the driving scene,
            including road layout, traffic participants, spatial relationships,
            and current vehicle state.
        predict_action: Predicted action the ego vehicle will take.
    """

    description_text: str
    predict_action: str


class FutureConfirmation(BaseModel):
    """Stage 2 output: Causal description from future frames.

    Attributes:
        causal_text: Complete causal description of what happened and why.
    """

    causal_text: str = ""


class CausalChain(BaseModel):
    """Final output: Causal chain for an event, ready for fine-tuning.

    Attributes:
        episode_id: Unique identifier for the driving episode.
        event_frame_id: Frame identifier of the detected event.
        frame_ids: All frame numbers used for analysis (history + future, chronological).
        action_type: Type of detected action (e.g., "hard_brake", "steering").
        causal_text: VLM-produced causal description combining prediction and outcome.
        simple_text: Baseline single-call VLM description (no two-stage isolation).
    """

    episode_id: str
    event_frame_id: str
    frame_ids: list[str]
    action_type: str
    causal_text: str
    simple_text: str = ""

    model_config = {"extra": "forbid"}
