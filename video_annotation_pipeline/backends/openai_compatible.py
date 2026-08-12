"""Dependency-light OpenAI-compatible multi-image backend."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from .base import BackendResponse, VideoAnnotatorBackend
from ..config import BackendConfig
from ..prompts import response_json_schema
from ..schemas import EpisodeMetadata, SampleManifestEntry


class OpenAICompatibleBackend(VideoAnnotatorBackend):
    def __init__(self, config: BackendConfig):
        if not config.base_url:
            raise ValueError("backend.base_url is required for openai_compatible")
        if not os.getenv(config.api_key_env):
            raise ValueError(
                f"environment variable {config.api_key_env} is required for openai_compatible"
            )
        self.config = config

    def annotate(
        self,
        frames: list[SampleManifestEntry],
        timestamps: list[float],
        camera_roles: list[str],
        metadata: EpisodeMetadata,
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for frame in frames:
            encoded = base64.b64encode(open(frame.image_path, "rb").read()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_new_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "video_annotation",
                    "strict": True,
                    "schema": response_json_schema(),
                },
            },
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv(self.config.api_key_env, '')}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc}") from exc
        return BackendResponse(
            text=result["choices"][0]["message"]["content"],
            model=str(result.get("model") or self.config.model),
            usage=result.get("usage"),
        )
