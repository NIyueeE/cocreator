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
    """

    base_url: str = "https://api.siliconflow.cn"
    api_key: str = ""
    model: str = "Qwen/Qwen3.5-397B-A17B"
    timeout: float = 120.0


class RateLimitConfig(BaseModel):
    """Rate limiting configuration for API requests.

    Attributes:
        rpm: Requests per minute limit.
        tpm: Tokens per minute limit.
        concurrency: Maximum concurrent requests.
    """

    rpm: int = 500
    tpm: int = 2000000
    concurrency: int = 20


class PipelineConfig(BaseModel):
    """Pipeline-wide configuration settings.

    Attributes:
        dataset_path: Path to the dataset directory.
        videos_path: Path to the videos directory.
        output_dir: Directory for pipeline output files.
        history_segments: Number of past segments to analyze.
        future_segments: Number of future segments to confirm.
        frames_per_segment: Number of frames per segment.
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
    history_segments: int = 3
    future_segments: int = 2
    frames_per_segment: int = 5
    anomaly_threshold: float = 2.0
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    min_event_interval: int = 5
    steering_threshold: float = 15.0
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
        confidence: Confidence score for the detection.
    """

    episode_id: str
    frame_id: str
    action_type: str
    confidence: float = 1.0

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
        ego_status: Ego vehicle status (e.g., "cruising", "turning", "stopped").
        key_objects: List of objects of interest in the scene.
        most_critical_object: The most critical object if identified.
        predicted_action: Predicted action the ego vehicle will take.
        reasoning: Natural language reasoning for the analysis.
    """

    ego_status: str
    key_objects: list[KeyObject]
    most_critical_object: Optional[KeyObject] = None
    predicted_action: str
    reasoning: str = ""


class FutureConfirmation(BaseModel):
    """Stage 2 output: Confirmation from future frames.

    Attributes:
        actual_action: The actual action observed in future frames.
        action_status: Status of the action (e.g., "completed", "in_progress", "aborted").
        related_to_history: Whether the action confirms the historical prediction.
        verification_notes: Additional notes on verification.
    """

    actual_action: str
    action_status: str
    related_to_history: bool
    verification_notes: str = ""


class CausalChain(BaseModel):
    """Final output: Complete causal chain for an event.

    Attributes:
        episode_id: Unique identifier for the driving episode.
        event_frame_id: Frame identifier of the detected event.
        confidence: Overall confidence score for the causal chain.
        historical_analysis: Stage 1 analysis results.
        future_confirmation: Stage 2 confirmation results.
        causal_link: Natural language summary of the causal relationship.
    """

    episode_id: str
    event_frame_id: str
    confidence: float
    historical_analysis: HistoricalAnalysis
    future_confirmation: FutureConfirmation
    causal_link: str

    model_config = {"extra": "forbid"}
