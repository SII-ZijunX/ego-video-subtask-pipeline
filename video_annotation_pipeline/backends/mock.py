"""Deterministic backend for CPU-only integration tests."""

from __future__ import annotations

import json

from .base import BackendResponse, VideoAnnotatorBackend
from ..schemas import EpisodeMetadata, SampleManifestEntry


class MockBackend(VideoAnnotatorBackend):
    def annotate(
        self,
        frames: list[SampleManifestEntry],
        timestamps: list[float],
        camera_roles: list[str],
        metadata: EpisodeMetadata,
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        duration = max(timestamps) if timestamps else 1.0
        # Sample timestamps are centers; extend the mock segment to a stable
        # approximation of the full duration for validator coverage tests.
        if len(timestamps) > 1:
            duration += (timestamps[1] - timestamps[0]) / 2
        task = metadata.task_hint or "Move the visible object."
        payload = {
            "video_level_instruction": task,
            "subtasks": [{
                "start_time_sec": 0.0,
                "end_time_sec": round(max(0.5, duration), 3),
                "action": "move",
                "object": "visible object",
                "instruction": task,
            }],
        }
        return BackendResponse(text=json.dumps(payload), model="mock-v0.1")
