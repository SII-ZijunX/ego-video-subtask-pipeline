"""Backend contract shared by mock, local-Qwen, and HTTP implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..schemas import EpisodeMetadata, SampleManifestEntry


@dataclass
class BackendResponse:
    text: str
    model: str
    model_path: str | None = None
    usage: dict[str, Any] | None = None


class VideoAnnotatorBackend(ABC):
    @abstractmethod
    def annotate(
        self,
        frames: list[SampleManifestEntry],
        timestamps: list[float],
        camera_roles: list[str],
        metadata: EpisodeMetadata,
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        raise NotImplementedError
