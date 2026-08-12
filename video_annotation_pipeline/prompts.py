"""Versioned prompts derived from annotation.md, section 10."""

from __future__ import annotations

import json

from .schemas import EpisodeMetadata, SampleManifestEntry
from .vocabulary import ACTION_VOCABULARY


SYSTEM_PROMPT = """You are an expert annotator for robot manipulation and egocentric operation videos.

Your task is to identify the overall task, segment the video into temporally contiguous manipulation subtasks, assign one coarse action label plus one visually grounded fine action verb, identify the primary manipulated object, and write one concise imperative instruction per subtask.

Allowed action labels (use exactly one):
move, fold, pour, unfold, push, wipe, pull, stir, rotate, cut, open, press, close, attach, detach, transit, idle, other.

fine_action is open vocabulary: use one specific lowercase visual verb such as peel, wash, inspect, type, or hold. Never replace a visible action with "other" in fine_action.

Segmentation rules:
- Treat approaching, grasping, carrying, and releasing the same object as one move subtask.
- Create a boundary only when the manipulated object changes, the atomic action changes, or a sustained pause starts a new sub-goal.
- Do not create boundaries for minor grasp adjustments, brief hesitation, camera motion, or temporary occlusion.
- move means relocating an object; transit means empty-hand motion; idle means hands remain stationary.
- Subtasks must be ordered, non-overlapping, and cover the meaningful visible activity.
- Object names are open vocabulary. Do not infer invisible objects, goals, identities, or state changes.
- Use timestamps shown in the sampled frames. Return valid JSON only, without markdown fences.
"""


def build_user_prompt(
    metadata: EpisodeMetadata,
    duration_sec: float,
    samples: list[SampleManifestEntry],
    retry_errors: list[str] | None = None,
    previous_response: str | None = None,
    simplify: bool = False,
) -> str:
    timeline = [
        {
            "sample_index": sample.sample_index,
            "time_sec": sample.original_time_sec,
            "camera_roles": sample.camera_roles,
        }
        for sample in samples
    ]
    prompt = f"""Episode ID: {metadata.episode_id}
Video duration: {duration_sec:.3f} seconds.
Source: {metadata.source}
Optional task hint: {metadata.task_hint or 'none'}
Sample timeline: {json.dumps(timeline, ensure_ascii=False)}

Return exactly this JSON shape:
{{
  "video_level_instruction": "concise overall imperative instruction",
  "subtasks": [
    {{
      "start_time_sec": 0.0,
      "end_time_sec": 1.0,
      "action": "one allowed label",
      "fine_action": "one specific visible action verb",
      "object": "primary object or null",
      "instruction": "concise imperative instruction"
    }}
  ]
}}

Boundaries must be chosen from or interpolated between visible sample timestamps. Use "other" for the coarse action only when no allowed action fits, and still provide a specific fine_action.
"""
    if retry_errors:
        prompt += (
            "\nThe previous response failed validation. Correct these issues: "
            + ", ".join(retry_errors)
            + ". Return a complete corrected JSON object.\n"
        )
    if previous_response:
        prompt += "Previous response:\n" + previous_response[:12000]
    if simplify:
        prompt += (
            "\nFinal retry: reduce complexity. Return only the required JSON keys, use the fewest "
            "subtasks justified by visible object/action changes, and do not add explanations.\n"
        )
    return prompt


def response_json_schema() -> dict:
    return {
        "type": "object",
        "required": ["video_level_instruction", "subtasks"],
        "properties": {
            "video_level_instruction": {"type": "string", "minLength": 1},
            "subtasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["start_time_sec", "end_time_sec", "action", "fine_action", "instruction"],
                    "properties": {
                        "start_time_sec": {"type": "number", "minimum": 0},
                        "end_time_sec": {"type": "number", "minimum": 0},
                        "action": {"type": "string", "enum": ACTION_VOCABULARY},
                        "fine_action": {"type": "string", "minLength": 1},
                        "object": {"type": ["string", "null"]},
                        "instruction": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }
