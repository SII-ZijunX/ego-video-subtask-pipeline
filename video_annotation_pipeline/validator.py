"""Schema, temporal, and semantic checks for normalized annotations."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .config import PipelineConfig
from .vocabulary import ACTION_SET, object_is_required


RETRY_FLAGS = frozenset({
    "vlm_parse_failure",
    "missing_video_level_instruction",
    "empty_subtasks",
    "overlapping_segments",
    "idle_only_episode",
    "low_temporal_coverage",
    "other_dominant",
})

HARD_INVALID_FLAGS = frozenset({
    "invalid_action",
    "missing_object",
    "empty_instruction",
    "invalid_time_range",
    "overlapping_segments",
    "missing_video_level_instruction",
    "empty_subtasks",
})


def validate_annotation(
    annotation: dict[str, Any], duration_sec: float, config: PipelineConfig
) -> tuple[list[str], list[str], float]:
    flags: list[str] = []
    subtasks = annotation.get("subtasks") or []
    if not str(annotation.get("video_level_instruction") or "").strip():
        flags.append("missing_video_level_instruction")
    if not subtasks:
        flags.append("empty_subtasks")
        return flags, [flag for flag in flags if flag in RETRY_FLAGS], 0.0

    valid_intervals: list[tuple[float, float]] = []
    previous_end: float | None = None
    action_counts: Counter[str] = Counter()
    ambiguous_other_count = 0
    for row in subtasks:
        action = str(row.get("action") or "")
        action_counts[action] += 1
        fine_action = str(row.get("fine_action") or "").strip().lower()
        if action == "other" and fine_action in {"", "other", "unknown", "n/a"}:
            ambiguous_other_count += 1
        row_flags = row.setdefault("quality_flags", [])
        flags.extend(str(flag) for flag in row_flags)
        start = float(row.get("start_time_sec", 0))
        end = float(row.get("end_time_sec", 0))

        if action not in ACTION_SET:
            row_flags.append("invalid_action")
            flags.append("invalid_action")
        if object_is_required(action) and not row.get("object"):
            row_flags.append("missing_object")
            flags.append("missing_object")
        if not str(row.get("instruction") or "").strip():
            row_flags.append("empty_instruction")
            flags.append("empty_instruction")
        if start < 0 or end > duration_sec + 1e-6 or end <= start:
            row_flags.append("invalid_time_range")
            flags.append("invalid_time_range")
        segment_duration = max(0.0, end - start)
        if segment_duration > 0:
            valid_intervals.append((max(0.0, start), min(duration_sec, end)))
        if segment_duration < config.segmentation.min_subtask_duration_sec:
            row_flags.append("too_short_segment")
            flags.append("too_short_segment")
        if previous_end is not None:
            if start < previous_end - config.validation.continuity_tolerance_sec:
                row_flags.append("overlapping_segments")
                flags.append("overlapping_segments")
            elif start - previous_end > config.validation.max_uncovered_gap_sec:
                row_flags.append("large_temporal_gap")
                flags.append("large_temporal_gap")
        previous_end = max(previous_end or 0.0, end)
        row["quality_flags"] = list(dict.fromkeys(row_flags))
        if row["quality_flags"]:
            row["training_eligible"] = False

    max_subtasks = max(1, int(config.segmentation.max_subtasks_per_minute * duration_sec / 60.0))
    if len(subtasks) > max_subtasks:
        flags.append("too_many_segments")
    if config.validation.reject_idle_only and action_counts and set(action_counts) == {"idle"}:
        flags.append("idle_only_episode")
    if ambiguous_other_count / len(subtasks) > config.validation.warn_other_ratio:
        flags.append("other_dominant")

    if subtasks:
        leading_gap = max(0.0, float(subtasks[0].get("start_time_sec", 0)))
        trailing_gap = max(0.0, duration_sec - float(subtasks[-1].get("end_time_sec", 0)))
        if leading_gap > config.validation.max_uncovered_gap_sec:
            flags.append("large_temporal_gap")
        if trailing_gap > config.validation.max_uncovered_gap_sec:
            flags.append("large_temporal_gap")

    covered = 0.0
    merged_intervals: list[list[float]] = []
    for start, end in sorted(valid_intervals):
        if end <= start:
            continue
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
    covered = sum(end - start for start, end in merged_intervals)
    coverage_ratio = min(1.0, covered / duration_sec) if duration_sec > 0 else 0.0
    if coverage_ratio < config.validation.minimum_coverage_ratio:
        flags.append("low_temporal_coverage")

    unique_flags = list(dict.fromkeys(flags))
    retry_reasons = [flag for flag in unique_flags if flag in RETRY_FLAGS]
    return unique_flags, retry_reasons, round(coverage_ratio, 4)


def annotation_is_valid(flags: list[str], retry_reasons: list[str]) -> bool:
    return not retry_reasons and not (set(flags) & HARD_INVALID_FLAGS)
