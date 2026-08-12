"""Construct a backend without importing GPU libraries on CPU-only commands."""

from __future__ import annotations

from .base import VideoAnnotatorBackend
from .mock import MockBackend
from ..config import BackendConfig


def create_backend(config: BackendConfig) -> VideoAnnotatorBackend:
    if config.type == "mock":
        return MockBackend()
    if config.type == "qwen_local":
        from .qwen_local import QwenLocalBackend
        return QwenLocalBackend(config)
    if config.type == "openai_compatible":
        from .openai_compatible import OpenAICompatibleBackend
        return OpenAICompatibleBackend(config)
    raise ValueError(f"unsupported backend type: {config.type}")
