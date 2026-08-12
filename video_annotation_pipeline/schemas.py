"""Pydantic schemas for episode inputs and normalized annotations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CameraRole = Literal["main", "overhead", "wrist", "wrist_left", "wrist_right", "other"]


class CameraSpec(BaseModel):
    name: str = Field(min_length=1)
    role: CameraRole
    path: str = Field(min_length=1)
    time_offset_sec: float = 0.0


class EpisodeMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    episode_id: str = Field(min_length=1)
    source: str = Field(default="unknown", min_length=1)
    task_hint: Optional[str] = None
    cameras: list[CameraSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def camera_names_are_unique(self) -> "EpisodeMetadata":
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("camera names must be unique inside an episode")
        return self

    def resolved_camera_path(self, episode_dir: Path, camera: CameraSpec) -> Path:
        path = Path(camera.path)
        return path if path.is_absolute() else episode_dir / path


class SubtaskAnnotation(BaseModel):
    subtask_id: int = Field(ge=0)
    start_time_sec: float = Field(ge=0)
    end_time_sec: float = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    action: str = Field(min_length=1)
    fine_action: str = Field(default="other", min_length=1)
    object: Optional[str] = None
    instruction: str = Field(min_length=1)
    training_eligible: bool = True
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("object")
    @classmethod
    def normalize_empty_object(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


class AnnotationMetadata(BaseModel):
    model: str
    model_path: Optional[str] = None
    prompt_version: str
    sampling_fps: float = Field(gt=0)
    annotation_mode: str = "single_pass"
    retry_count: int = Field(default=0, ge=0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: Optional[str] = None
    runtime_sec: Optional[float] = Field(default=None, ge=0)


class EpisodeAnnotation(BaseModel):
    schema_version: str = "0.1"
    episode_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    duration_sec: float = Field(gt=0)
    video_level_instruction: str = Field(min_length=1)
    subtasks: list[SubtaskAnnotation] = Field(min_length=1)
    annotation_metadata: AnnotationMetadata
    episode_quality_flags: list[str] = Field(default_factory=list)


class ModelSubtask(BaseModel):
    """Lenient model-facing schema before normalization and repair."""

    model_config = ConfigDict(extra="allow")

    start_time_sec: float
    end_time_sec: float
    action: str
    fine_action: Optional[str] = None
    object: Optional[str] = None
    instruction: str


class ModelAnnotation(BaseModel):
    model_config = ConfigDict(extra="allow")

    video_level_instruction: str
    subtasks: list[ModelSubtask]


class VideoInfo(BaseModel):
    camera_name: str
    camera_role: str
    path: str
    codec_name: Optional[str] = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    duration_sec: float = Field(gt=0)
    time_offset_sec: float = 0.0


class SampleManifestEntry(BaseModel):
    sample_index: int = Field(ge=0)
    original_time_sec: float = Field(ge=0)
    image_path: str
    camera_roles: list[str]
    source_times_sec: dict[str, float]


class ValidationReport(BaseModel):
    episode_id: str
    valid: bool
    quality_flags: list[str]
    retry_reasons: list[str]
    coverage_ratio: float = Field(ge=0)
    normalized_annotation: Optional[dict[str, Any]] = None
