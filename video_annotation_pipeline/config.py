"""Typed YAML configuration for the annotation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    schema_version: str = "0.1"
    output_dir: str = "outputs/video_annotation"


class BackendConfig(BaseModel):
    type: Literal["mock", "qwen_local", "openai_compatible"] = "mock"
    model: str = "Qwen3-VL-32B-Instruct"
    model_path: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = "VLM_API_KEY"
    timeout_sec: int = Field(default=300, ge=1)
    max_retries: int = Field(default=2, ge=0, le=10)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_new_tokens: int = Field(default=1024, ge=64)
    attn_implementation: Optional[str] = None


class SamplingConfig(BaseModel):
    fps: float = Field(default=2.0, gt=0)
    max_frames: int = Field(default=128, ge=1)
    resize_long_side: int = Field(default=448, ge=64)
    multi_view_layout: Literal["grid"] = "grid"


class SegmentationConfig(BaseModel):
    min_subtask_duration_sec: float = Field(default=3.0, gt=0)
    max_subtasks_per_minute: float = Field(default=30.0, gt=0)
    sustained_pause_sec: float = Field(default=1.5, ge=0)


class ValidationConfig(BaseModel):
    continuity_tolerance_sec: float = Field(default=0.2, ge=0)
    max_uncovered_gap_sec: float = Field(default=1.0, ge=0)
    max_multiview_duration_diff_sec: float = Field(default=0.5, ge=0)
    minimum_video_duration_sec: float = Field(default=3.0, gt=0)
    reject_idle_only: bool = True
    warn_other_ratio: float = Field(default=0.3, ge=0, le=1)
    minimum_coverage_ratio: float = Field(default=0.7, ge=0, le=1)


class OutputConfig(BaseModel):
    save_raw_response: bool = True
    save_sampled_frames: bool = True
    save_prompt: bool = True
    generate_html: bool = True


class PipelineConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompt_version: str = "lingbot_v0.1"


def load_config(path: Path | str) -> PipelineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    return PipelineConfig.model_validate(data)


def write_config_snapshot(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    )
