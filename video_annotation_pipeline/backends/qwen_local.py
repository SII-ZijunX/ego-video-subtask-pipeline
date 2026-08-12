"""In-process Qwen3-VL backend using already-downloaded local weights."""

from __future__ import annotations

import time

from .base import BackendResponse, VideoAnnotatorBackend
from ..config import BackendConfig
from ..schemas import EpisodeMetadata, SampleManifestEntry


class QwenLocalBackend(VideoAnnotatorBackend):
    def __init__(self, config: BackendConfig):
        self.config = config
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.config.model_path:
            raise ValueError(
                "backend.model_path is required for qwen_local; set it in your config"
            )
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        kwargs = {"dtype": "auto", "device_map": "auto"}
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        print(f"[qwen] loading {self.config.model_path}", flush=True)
        started = time.time()
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_path, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.config.model_path)
        print(f"[qwen] loaded in {time.time() - started:.1f}s", flush=True)

    def annotate(
        self,
        frames: list[SampleManifestEntry],
        timestamps: list[float],
        camera_roles: list[str],
        metadata: EpisodeMetadata,
        system_prompt: str,
        user_prompt: str,
    ) -> BackendResponse:
        import torch
        from qwen_vl_utils import process_vision_info

        self._load()
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for frame in frames:
            content.extend([
                {"type": "text", "text": f"Sample {frame.sample_index}, t={frame.original_time_sec:.3f}s"},
                {"type": "image", "image": frame.image_path},
            ])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._model.device)
        generation_kwargs = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.config.temperature})
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_kwargs)
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        response = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return BackendResponse(
            text=response,
            model=self.config.model,
            model_path=self.config.model_path,
            usage={"input_tokens": int(inputs.input_ids.shape[-1]), "sampled_frames": len(frames)},
        )
