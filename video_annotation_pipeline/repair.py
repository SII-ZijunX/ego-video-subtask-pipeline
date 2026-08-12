"""Conservative automatic repairs allowed by annotation.md."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import PipelineConfig
from .vocabulary import canonicalize_token, normalize_action, normalize_object


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_and_repair(
    raw: dict[str, Any], duration_sec: float, fps: float, config: PipelineConfig
) -> tuple[dict[str, Any], list[str]]:
    """Normalize model output without guessing semantics or moving large boundaries."""
    data = deepcopy(raw)
    repair_flags: list[str] = []
    source_subtasks = data.get("subtasks")
    if not isinstance(source_subtasks, list):
        source_subtasks = []

    normalized: list[dict[str, Any]] = []
    for source_index, item in enumerate(source_subtasks):
        if not isinstance(item, dict):
            repair_flags.append("invalid_subtask_type")
            continue
        start = min(duration_sec, max(0.0, _as_float(item.get("start_time_sec"))))
        end = min(duration_sec, max(0.0, _as_float(item.get("end_time_sec"))))
        if start != _as_float(item.get("start_time_sec")) or end != _as_float(item.get("end_time_sec")):
            repair_flags.append("time_clamped")
        raw_action = canonicalize_token(item.get("action"))
        action, action_flags = normalize_action(raw_action)
        fine_action = canonicalize_token(item.get("fine_action"))
        if not fine_action or fine_action in {"unknown", "n/a"}:
            fine_action = raw_action or action
        if action_flags == ["out_of_vocabulary_action"] and fine_action not in {"", "other", "unknown", "n/a"}:
            action_flags = []
            repair_flags.append("coarse_action_inferred")
        instruction = str(item.get("instruction") or "").strip()
        normalized.append({
            "source_index": source_index,
            "start_time_sec": round(start, 3),
            "end_time_sec": round(end, 3),
            "action": action,
            "fine_action": fine_action,
            "object": normalize_object(item.get("object")),
            "instruction": instruction,
            "quality_flags": list(dict.fromkeys(action_flags)),
        })
        repair_flags.extend(action_flags)

    sorted_rows = sorted(normalized, key=lambda row: (row["start_time_sec"], row["end_time_sec"]))
    if sorted_rows != normalized:
        repair_flags.append("subtasks_sorted")

    # Remove exact/contained semantic duplicates, but never merge different actions.
    deduplicated: list[dict[str, Any]] = []
    for row in sorted_rows:
        duplicate = any(
            row["start_time_sec"] >= prev["start_time_sec"]
            and row["end_time_sec"] <= prev["end_time_sec"]
            and row["action"] == prev["action"]
            and row["object"] == prev["object"]
            for prev in deduplicated
        )
        if duplicate:
            repair_flags.append("contained_duplicate_removed")
            continue
        deduplicated.append(row)

    tolerance = config.validation.continuity_tolerance_sec
    for previous, current in zip(deduplicated, deduplicated[1:]):
        gap = current["start_time_sec"] - previous["end_time_sec"]
        if 0 < gap <= tolerance:
            boundary = round((current["start_time_sec"] + previous["end_time_sec"]) / 2, 3)
            previous["end_time_sec"] = boundary
            current["start_time_sec"] = boundary
            repair_flags.append("small_gap_repaired")

    # The specification explicitly treats grasp/carry/release of the same
    # object as one interaction.  Adjacent rows with the same normalized
    # action and object can therefore be merged safely when there is no
    # sustained pause; unlike semantic actions are never merged here.
    merged_rows: list[dict[str, Any]] = []
    for row in deduplicated:
        if merged_rows:
            previous = merged_rows[-1]
            gap = row["start_time_sec"] - previous["end_time_sec"]
            same_interaction = (
                previous["action"] == row["action"]
                and previous["object"] == row["object"]
                and gap <= config.segmentation.sustained_pause_sec
            )
            if same_interaction:
                previous["end_time_sec"] = max(previous["end_time_sec"], row["end_time_sec"])
                first_instruction = previous["instruction"].strip()
                second_instruction = row["instruction"].strip()
                if second_instruction and second_instruction.lower() != first_instruction.lower():
                    if first_instruction:
                        second_instruction = second_instruction[0].lower() + second_instruction[1:]
                        previous["instruction"] = first_instruction.rstrip(".") + ", then " + second_instruction
                    else:
                        previous["instruction"] = second_instruction
                previous["quality_flags"] = list(dict.fromkeys(
                    previous["quality_flags"] + row["quality_flags"]
                ))
                repair_flags.append("same_interaction_merged")
                continue
        merged_rows.append(row)

    final_rows: list[dict[str, Any]] = []
    for subtask_id, row in enumerate(merged_rows):
        quality_flags = list(dict.fromkeys(row.pop("quality_flags")))
        row.pop("source_index", None)
        row.update({
            "subtask_id": subtask_id,
            "start_frame": max(0, int(round(row["start_time_sec"] * fps))),
            "end_frame": max(0, int(round(row["end_time_sec"] * fps))),
            "training_eligible": not quality_flags,
            "quality_flags": quality_flags,
        })
        final_rows.append(row)

    normalized_annotation = {
        "video_level_instruction": str(data.get("video_level_instruction") or "").strip(),
        "subtasks": final_rows,
    }
    return normalized_annotation, list(dict.fromkeys(repair_flags))
