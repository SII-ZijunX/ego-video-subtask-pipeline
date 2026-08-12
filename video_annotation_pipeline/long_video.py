"""End-to-end Ego4D long-video subtask segmentation workflow.

The long source clip is covered by overlapping *analysis windows*.  These
windows are inference chunks, not training segments.  The existing
``video_annotation_pipeline`` predicts one or more visual subtasks per window;
this tool maps them back to source time, reconciles overlap, exports the final
short clips, and builds an auditable review page.

Narration is deliberately not used for selection, prompting, segmentation, or
label generation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_EGO4D_ROOT = os.getenv("EGO4D_ROOT")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ffprobe_duration(path: Path) -> float | None:
    completed = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    if completed.returncode:
        return None
    try:
        return round(float(completed.stdout.strip()), 3)
    except ValueError:
        return None


def stable_score(seed: int, clip_uid: str) -> str:
    return hashlib.sha1(f"{seed}:{clip_uid}".encode()).hexdigest()


def safe_identifier(value: str) -> str:
    """Return a stable filesystem/episode-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned or hashlib.sha1(value.encode()).hexdigest()[:16]


def analysis_windows(duration_sec: float, window_sec: float, overlap_sec: float) -> list[tuple[float, float]]:
    """Return deterministic windows that cover [0, duration] without gaps."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if window_sec <= 0 or not 0 <= overlap_sec < window_sec:
        raise ValueError("require window_sec > 0 and 0 <= overlap_sec < window_sec")
    if duration_sec <= window_sec:
        return [(0.0, round(duration_sec, 3))]
    stride = window_sec - overlap_sec
    starts: list[float] = []
    current = 0.0
    while current + window_sec < duration_sec - 1e-6:
        starts.append(round(current, 3))
        current += stride
    final_start = max(0.0, duration_sec - window_sec)
    if not starts or abs(final_start - starts[-1]) > 1e-3:
        starts.append(round(final_start, 3))
    return [(start, round(min(duration_sec, start + window_sec), 3)) for start in starts]


def select_source_videos(
    root: Path,
    count: int,
    min_duration_sec: float,
    max_duration_sec: float,
    seed: int,
) -> list[dict[str, Any]]:
    metadata_path = root / "ego4d.json"
    clips_dir = root / "v2" / "clips"
    if not metadata_path.is_file() or not clips_dir.is_dir():
        raise FileNotFoundError(f"incomplete Ego4D mount under {root}")
    metadata = read_json(metadata_path)
    videos = {str(row.get("video_uid")): row for row in metadata.get("videos", [])}
    candidates: list[dict[str, Any]] = []
    for clip in metadata.get("clips", []):
        clip_uid = str(clip.get("clip_uid") or "")
        video_uid = str(clip.get("video_uid") or "")
        if not clip_uid or not video_uid:
            continue
        source_path = clips_dir / f"{clip_uid}.mp4"
        if not source_path.is_file():
            continue
        clip_start = float(clip.get("video_start_sec") or 0.0)
        clip_end = float(clip.get("video_end_sec") or 0.0)
        duration = clip_end - clip_start
        if not min_duration_sec <= duration <= max_duration_sec:
            continue
        video_meta = videos.get(video_uid, {})
        scenarios = [str(value) for value in (video_meta.get("scenarios") or []) if value]
        candidates.append({
            "dataset": "ego4d_v2",
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "source_clip_path": str(source_path.resolve()),
            "duration_sec": round(duration, 3),
            "clip_video_start_sec": round(clip_start, 3),
            "clip_video_end_sec": round(clip_end, 3),
            "scenario": scenarios[0] if scenarios else "unknown",
            "scenarios": scenarios,
            "selection_score": stable_score(seed, clip_uid),
            "label_source": "qwen_visual_only",
            "narration_used": False,
        })
    candidates.sort(key=lambda row: (row["selection_score"], row["clip_uid"]))
    selected: list[dict[str, Any]] = []
    used_scenarios: set[str] = set()
    for require_new_scenario in (True, False):
        for row in candidates:
            if row in selected:
                continue
            scenario = str(row["scenario"])
            if require_new_scenario and scenario in used_scenarios:
                continue
            selected.append(row)
            used_scenarios.add(scenario)
            if len(selected) == count:
                return selected
    raise RuntimeError(
        f"only found {len(selected)} eligible local clips in {min_duration_sec}-{max_duration_sec}s"
    )


def load_source_manifest(
    manifest_path: Path,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    """Load generic MP4 sources using the long-video output contract."""
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError(f"source manifest is empty: {manifest_path}")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for index, raw in enumerate(rows):
        path_value = str(raw.get("source_clip_path") or raw.get("path") or "").strip()
        if not path_value:
            errors.append(f"row {index}: missing source_clip_path")
            continue
        source_path = Path(path_value).expanduser().resolve()
        if not source_path.is_file():
            errors.append(f"row {index}: file not found: {source_path}")
            continue
        duration = ffprobe_duration(source_path)
        if duration is None or duration <= 0:
            errors.append(f"row {index}: ffprobe failed: {source_path}")
            continue
        if not min_duration_sec <= duration <= max_duration_sec:
            errors.append(
                f"row {index}: duration {duration}s outside "
                f"[{min_duration_sec}, {max_duration_sec}]"
            )
            continue
        dataset = safe_identifier(str(raw.get("dataset") or "generic_mp4"))
        supplied_uid = str(raw.get("clip_uid") or "").strip()
        clip_uid = safe_identifier(
            supplied_uid
            or f"{dataset}-{source_path.stem}-{hashlib.sha1(str(source_path).encode()).hexdigest()[:8]}"
        )
        if clip_uid in seen:
            errors.append(f"row {index}: duplicate clip_uid: {clip_uid}")
            continue
        seen.add(clip_uid)
        scenario = str(raw.get("scenario") or raw.get("task_hint") or "unknown")
        selected.append({
            **raw,
            "dataset": dataset,
            "clip_uid": clip_uid,
            "video_uid": str(raw.get("video_uid") or clip_uid),
            "source_clip_path": str(source_path),
            "duration_sec": duration,
            "clip_video_start_sec": 0.0,
            "clip_video_end_sec": duration,
            "scenario": scenario,
            "scenarios": list(raw.get("scenarios") or ([scenario] if scenario else [])),
            "label_source": "qwen_visual_only",
            "narration_used": False,
        })
    if errors:
        raise ValueError("invalid source manifest:\n" + "\n".join(errors))
    return selected


def export_analysis_window(
    row: dict[str, Any], pilot_dir: Path, overwrite: bool
) -> dict[str, Any]:
    started = time.time()
    clip_uid = str(row["clip_uid"])
    episode_id = str(row["episode_id"])
    target = pilot_dir / "analysis_clips" / clip_uid / f"{episode_id}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {**row, "analysis_clip_path": str(target.resolve()), "export_ok": False, "error": None}
    if overwrite or not target.is_file():
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(row['window_start_sec']):.3f}",
            "-i", str(row["source_clip_path"]),
            "-t", f"{float(row['window_duration_sec']):.3f}",
            "-map", "0:v:0", "-an", "-vf", "scale=960:-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ]
        completed = run(command)
        result["ffmpeg_returncode"] = completed.returncode
        result["ffmpeg_stderr_tail"] = completed.stderr[-1000:]
        if completed.returncode:
            result["error"] = "ffmpeg_failed"
            return result
    duration = ffprobe_duration(target)
    result["exported_duration_sec"] = duration
    result["export_elapsed_sec"] = round(time.time() - started, 3)
    tolerance = max(0.35, 2.0 / 30.0)
    result["export_ok"] = duration is not None and abs(duration - float(row["window_duration_sec"])) <= tolerance
    if not result["export_ok"]:
        result["error"] = "duration_mismatch"
        return result
    episode_dir = pilot_dir / "dataset" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_metadata = {
        "episode_id": episode_id,
        "source": f"{row.get('dataset', 'generic_mp4')}_long_window",
        "dataset": row.get("dataset", "generic_mp4"),
        "task_hint": row.get("task_hint"),
        "cameras": [{
            "name": "main", "role": "main", "path": str(target.resolve()),
            "time_offset_sec": 0.0,
        }],
        "clip_uid": clip_uid,
        "video_uid": row.get("video_uid"),
        "source_clip_path": row["source_clip_path"],
        "source_duration_sec": row["source_duration_sec"],
        "source_window_start_sec": row["window_start_sec"],
        "source_window_end_sec": row["window_end_sec"],
        "analysis_window_index": row["window_index"],
        "analysis_window_overlap_sec": row["overlap_sec"],
        "label_source": "qwen_visual_only",
        "narration_used": False,
    }
    write_json(episode_dir / "metadata.json", episode_metadata)
    return result


def prepare(args: argparse.Namespace) -> None:
    pilot_dir = Path(args.output_dir).resolve()
    pilot_dir.mkdir(parents=True, exist_ok=True)
    if args.source_manifest:
        source_rows = load_source_manifest(
            Path(args.source_manifest).resolve(), args.min_duration_sec, args.max_duration_sec
        )
        if args.num_videos is not None:
            source_rows = source_rows[:args.num_videos]
    else:
        if not args.ego4d_root:
            raise ValueError(
                "set --source-manifest, --ego4d-root, or the EGO4D_ROOT environment variable"
            )
        source_rows = select_source_videos(
            Path(args.ego4d_root), args.num_videos or 10, args.min_duration_sec,
            args.max_duration_sec, args.seed,
        )
    windows: list[dict[str, Any]] = []
    for source in source_rows:
        for index, (start, end) in enumerate(analysis_windows(
            float(source["duration_sec"]), args.window_sec, args.overlap_sec
        )):
            dataset = safe_identifier(str(source.get("dataset") or "generic_mp4"))
            episode_id = f"{dataset}_long__{source['clip_uid']}__win{index:04d}"
            windows.append({
                "episode_id": episode_id,
                "clip_uid": source["clip_uid"],
                "video_uid": source["video_uid"],
                "dataset": source.get("dataset", "generic_mp4"),
                "task_hint": source.get("task_hint"),
                "source_clip_path": source["source_clip_path"],
                "source_duration_sec": source["duration_sec"],
                "window_index": index,
                "window_start_sec": start,
                "window_end_sec": end,
                "window_duration_sec": round(end - start, 3),
                "overlap_sec": args.overlap_sec,
            })
    exported: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(export_analysis_window, row, pilot_dir, args.overwrite) for row in windows]
        for future in as_completed(futures):
            exported.append(future.result())
    exported.sort(key=lambda row: (row["clip_uid"], row["window_index"]))
    write_jsonl(pilot_dir / "source_videos.jsonl", source_rows)
    write_jsonl(pilot_dir / "analysis_windows.jsonl", exported)
    (pilot_dir / "clip_uids.txt").write_text("\n".join(row["clip_uid"] for row in source_rows) + "\n")
    failed = [row for row in exported if not row.get("export_ok")]
    manifest = {
        "workflow": "long_video_subtask_v1",
        "datasets": sorted({str(row.get("dataset") or "generic_mp4") for row in source_rows}),
        "label_source": "qwen_visual_only",
        "narration_used": False,
        "num_videos": len(source_rows),
        "num_analysis_windows": len(exported),
        "exported_windows": len(exported) - len(failed),
        "failed_windows": len(failed),
        "window_sec": args.window_sec,
        "overlap_sec": args.overlap_sec,
        "min_duration_sec": args.min_duration_sec,
        "max_duration_sec": args.max_duration_sec,
        "seed": args.seed,
        "dataset_dir": str((pilot_dir / "dataset").resolve()),
        "qwen_output_dir": str((pilot_dir / "qwen").resolve()),
    }
    write_json(pilot_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if failed:
        raise RuntimeError(f"{len(failed)} analysis windows failed export")


def semantic_key(row: dict[str, Any]) -> tuple[str, str]:
    action = str(row.get("fine_action") or row.get("action") or "other").strip().lower()
    obj = " ".join(str(row.get("object") or "").strip().lower().split())
    return action, obj


def annotation_candidates(
    source_rows: Sequence[dict[str, Any]],
    window_rows: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    sources = {str(row["clip_uid"]): row for row in source_rows}
    windows = {str(row["episode_id"]): row for row in window_rows if row.get("export_ok")}
    candidates: list[dict[str, Any]] = []
    unknown: list[str] = []
    for annotation in annotations:
        episode_id = str(annotation.get("episode_id") or "")
        window = windows.get(episode_id)
        if window is None:
            unknown.append(episode_id)
            continue
        clip_uid = str(window["clip_uid"])
        source_duration = float(sources[clip_uid]["duration_sec"])
        window_start = float(window["window_start_sec"])
        window_duration = float(window["window_duration_sec"])
        overlap = float(window.get("overlap_sec") or 0.0)
        for subtask in annotation.get("subtasks") or []:
            local_start = max(0.0, min(window_duration, float(subtask.get("start_time_sec") or 0.0)))
            local_end = max(0.0, min(window_duration, float(subtask.get("end_time_sec") or 0.0)))
            if local_end <= local_start:
                continue
            global_start = max(0.0, min(source_duration, window_start + local_start))
            global_end = max(0.0, min(source_duration, window_start + local_end))
            left_edge = local_start <= overlap / 2 and window_start > 0
            right_edge = local_end >= window_duration - overlap / 2 and window["window_end_sec"] < source_duration
            subtask_flags = [str(value) for value in (subtask.get("quality_flags") or [])]
            episode_flags = [str(value) for value in (annotation.get("episode_quality_flags") or [])]
            flags = list(dict.fromkeys(
                subtask_flags + (["analysis_window_edge"] if left_edge or right_edge else [])
            ))
            candidates.append({
                "candidate_id": f"{episode_id}__qseg{int(subtask.get('subtask_id') or 0):03d}",
                "episode_id": episode_id,
                "clip_uid": clip_uid,
                "video_uid": sources[clip_uid].get("video_uid"),
                "source_clip_path": sources[clip_uid]["source_clip_path"],
                "window_index": window["window_index"],
                "window_start_sec": window_start,
                "window_end_sec": window["window_end_sec"],
                "local_start_sec": round(local_start, 3),
                "local_end_sec": round(local_end, 3),
                "start_sec": round(global_start, 3),
                "end_sec": round(global_end, 3),
                "action": str(subtask.get("action") or "other"),
                "fine_action": str(subtask.get("fine_action") or subtask.get("action") or "other"),
                "object": subtask.get("object"),
                "instruction": str(subtask.get("instruction") or "").strip(),
                "training_eligible": bool(subtask.get("training_eligible", True)) and not flags,
                "quality_flags": flags,
                "episode_quality_flags": episode_flags,
                "edge_margin_sec": round(min(local_start, window_duration - local_end), 3),
                "qwen_model": (annotation.get("annotation_metadata") or {}).get("model"),
                "prompt_version": (annotation.get("annotation_metadata") or {}).get("prompt_version"),
                "label_source": "qwen_visual_only",
                "narration_used": False,
            })
    return candidates, unknown


def _choose_candidate(active: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not active:
        return None
    return max(active, key=lambda row: (
        bool(row.get("training_eligible")),
        "analysis_window_edge" not in (row.get("quality_flags") or []),
        float(row.get("edge_margin_sec") or 0.0),
        float(row["end_sec"]) - float(row["start_sec"]),
        str(row["candidate_id"]),
    ))


def _coalesce_boundaries(values: Sequence[float], tolerance: float = 0.15) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group), 3) for group in groups]


def _segment_duration(row: dict[str, Any]) -> float:
    return round(float(row["end_sec"]) - float(row["start_sec"]), 3)


def _merge_instructions(left: str, right: str) -> str:
    left = left.strip().rstrip(".")
    right = right.strip()
    if not left:
        return right
    if not right or left.casefold() == right.rstrip(".").casefold():
        return left + "."
    return f"{left}. Then {right[0].lower() + right[1:] if right else right}"


def _merge_adjacent_segments(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Merge contiguous rows while preserving provenance and review safety."""
    same_semantics = semantic_key(left) == semantic_key(right)
    flags = list(dict.fromkeys(
        list(left.get("quality_flags") or []) + list(right.get("quality_flags") or [])
    ))
    uncovered_duration = sum(
        float(row.get("uncovered_duration_sec", _segment_duration(row) if "uncovered_by_qwen" in (row.get("quality_flags") or []) else 0.0))
        for row in (left, right)
    )
    if same_semantics:
        merged = {**left}
        merged["instruction"] = _merge_instructions(
            str(left.get("instruction") or ""), str(right.get("instruction") or "")
        )
        merged["training_eligible"] = (
            bool(left.get("training_eligible"))
            and bool(right.get("training_eligible"))
            and not flags
        )
    else:
        flags.append("merged_for_min_duration")
        merged = {
            **left,
            "action": "other",
            "fine_action": "compound",
            "object": None,
            "instruction": _merge_instructions(
                str(left.get("instruction") or ""), str(right.get("instruction") or "")
            ),
            "training_eligible": False,
            "label_source": "qwen_visual_only_min_duration_merge",
        }
    merged["start_sec"] = round(float(left["start_sec"]), 3)
    merged["end_sec"] = round(float(right["end_sec"]), 3)
    merged["candidate_ids"] = sorted(set(
        list(left.get("candidate_ids") or []) + list(right.get("candidate_ids") or [])
    ))
    merged["source_episode_ids"] = sorted(set(
        list(left.get("source_episode_ids") or []) + list(right.get("source_episode_ids") or [])
    ))
    merged["quality_flags"] = list(dict.fromkeys(flags))
    merged["uncovered_duration_sec"] = round(uncovered_duration, 3)
    return merged


def enforce_min_segment_duration(
    rows: Sequence[dict[str, Any]], min_segment_sec: float
) -> list[dict[str, Any]]:
    """Merge short rows until every exported interval meets the hard minimum."""
    if min_segment_sec <= 0:
        raise ValueError("min_segment_sec must be positive")
    merged = [dict(row) for row in rows]
    if merged and float(merged[-1]["end_sec"]) - float(merged[0]["start_sec"]) < min_segment_sec:
        raise ValueError("source duration is shorter than min_segment_sec")
    while len(merged) > 1:
        short = [index for index, row in enumerate(merged) if _segment_duration(row) < min_segment_sec]
        if not short:
            break
        index = min(short, key=lambda value: (_segment_duration(merged[value]), value))
        if index == 0:
            neighbor = 1
        elif index == len(merged) - 1:
            neighbor = index - 1
        else:
            options = [index - 1, index + 1]
            semantic_matches = [
                value for value in options if semantic_key(merged[value]) == semantic_key(merged[index])
            ]
            neighbor = min(
                semantic_matches or options,
                key=lambda value: (_segment_duration(merged[value]), value),
            )
        left_index = min(index, neighbor)
        merged[left_index] = _merge_adjacent_segments(merged[left_index], merged[left_index + 1])
        del merged[left_index + 1]
    if any(_segment_duration(row) < min_segment_sec for row in merged):
        raise RuntimeError("failed to enforce minimum final segment duration")
    return merged


def stitch_timeline(
    source: dict[str, Any], candidates: Sequence[dict[str, Any]], min_segment_sec: float = 1.0
) -> list[dict[str, Any]]:
    """Resolve overlapping window predictions into one complete, non-overlapping timeline."""
    duration = float(source["duration_sec"])
    clipped = [row for row in candidates if row["end_sec"] > row["start_sec"]]
    boundaries = _coalesce_boundaries(
        [0.0, duration] + [float(row[key]) for row in clipped for key in ("start_sec", "end_sec")]
    )
    boundaries[0], boundaries[-1] = 0.0, round(duration, 3)
    atomic: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 1e-4:
            continue
        midpoint = (start + end) / 2
        active = [row for row in clipped if float(row["start_sec"]) <= midpoint < float(row["end_sec"])]
        chosen = _choose_candidate(active)
        if chosen is None:
            row = {
                "action": "other", "fine_action": "other", "object": None,
                "instruction": "Review uncovered activity.",
                "training_eligible": False,
                "quality_flags": ["uncovered_by_qwen"],
                "candidate_ids": [], "source_episode_ids": [],
                "label_source": "coverage_filler",
            }
        else:
            same_semantics = [row for row in active if semantic_key(row) == semantic_key(chosen)]
            flags = list(dict.fromkeys(
                flag for row in same_semantics for flag in (row.get("quality_flags") or [])
                if flag != "analysis_window_edge" or all(
                    "analysis_window_edge" in (item.get("quality_flags") or []) for item in same_semantics
                )
            ))
            row = {
                "action": chosen["action"], "fine_action": chosen.get("fine_action") or chosen["action"],
                "object": chosen.get("object"),
                "instruction": chosen["instruction"],
                "training_eligible": bool(chosen.get("training_eligible")) and not flags,
                "quality_flags": flags,
                "candidate_ids": sorted({item["candidate_id"] for item in same_semantics}),
                "source_episode_ids": sorted({item["episode_id"] for item in same_semantics}),
                "label_source": "qwen_visual_only",
            }
        row.update({"start_sec": round(start, 3), "end_sec": round(end, 3)})
        if atomic and semantic_key(atomic[-1]) == semantic_key(row) and atomic[-1]["label_source"] == row["label_source"]:
            atomic[-1]["end_sec"] = row["end_sec"]
            atomic[-1]["candidate_ids"] = sorted(set(atomic[-1]["candidate_ids"] + row["candidate_ids"]))
            atomic[-1]["source_episode_ids"] = sorted(set(atomic[-1]["source_episode_ids"] + row["source_episode_ids"]))
            atomic[-1]["quality_flags"] = list(dict.fromkeys(atomic[-1]["quality_flags"] + row["quality_flags"]))
            atomic[-1]["training_eligible"] = atomic[-1]["training_eligible"] and row["training_eligible"]
        else:
            atomic.append(row)
    final: list[dict[str, Any]] = []
    atomic = enforce_min_segment_duration(atomic, min_segment_sec)
    for index, row in enumerate(atomic):
        segment_id = f"ego4d__{source['clip_uid']}__seg{index:04d}"
        duration_sec = round(float(row["end_sec"]) - float(row["start_sec"]), 3)
        flags = list(row["quality_flags"])
        training_eligible = bool(row["training_eligible"])
        if duration_sec < min_segment_sec:
            raise RuntimeError(f"{segment_id} is shorter than {min_segment_sec}s")
        fine_action = str(row.get("fine_action") or "").strip().lower()
        ambiguous_other = row["action"] == "other" and fine_action in {"", "other", "unknown", "n/a"}
        if row["action"] == "idle" or ambiguous_other:
            training_eligible = False
            flags.append("non_training_action")
        final.append({
            **row,
            "dataset": "ego4d_v2",
            "clip_uid": source["clip_uid"],
            "video_uid": source.get("video_uid"),
            "source_clip_path": source["source_clip_path"],
            "segment_id": segment_id,
            "duration_sec": duration_sec,
            "uncovered_duration_sec": round(float(row.get("uncovered_duration_sec", duration_sec if "uncovered_by_qwen" in flags else 0.0)), 3),
            "label": row["instruction"],
            "training_eligible": training_eligible and not flags,
            "quality_flags": list(dict.fromkeys(flags)),
            "narration_used": False,
        })
    return final


def export_final_segment(row: dict[str, Any], pilot_dir: Path, overwrite: bool) -> dict[str, Any]:
    target = pilot_dir / "final_segments" / str(row["clip_uid"]) / f"{row['segment_id']}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = {**row, "segment_clip_path": str(target.resolve()), "export_ok": False, "export_error": None}
    if overwrite or not target.is_file():
        completed = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{float(row['start_sec']):.3f}", "-i", str(row["source_clip_path"]),
            "-t", f"{float(row['duration_sec']):.3f}", "-map", "0:v:0", "-an",
            "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ])
        if completed.returncode:
            result["export_error"] = completed.stderr[-1000:] or "ffmpeg_failed"
            return result
    actual_duration = ffprobe_duration(target)
    result["exported_duration_sec"] = actual_duration
    result["export_ok"] = actual_duration is not None and abs(actual_duration - float(row["duration_sec"])) <= 0.4
    if not result["export_ok"]:
        result["export_error"] = "duration_mismatch"
    write_json(target.with_suffix(".json"), result)
    return result


def _deduplicate_captions(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        caption = " ".join(str(value or "").split())
        caption = re.sub(r"(?:then\s+)?review uncovered activity\.?", "", caption, flags=re.IGNORECASE)
        caption = re.sub(r"\s+", " " , caption).strip(" ;.")
        if caption:
            caption += "."
        if not caption:
            continue
        if result and result[-1].rstrip(".").casefold() == caption.rstrip(".").casefold():
            continue
        result.append(caption)
    return result


def _representative_captions(values: Sequence[str], limit: int = 8) -> list[str]:
    if len(values) <= limit:
        return list(values)
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in dict.fromkeys(indices)]


def compose_long_video_caption(values: Sequence[str]) -> str:
    captions = _representative_captions(_deduplicate_captions(values))
    if not captions:
        return "No reliable visible-action summary is available; review the uncovered timeline."
    cleaned = [caption.rstrip(".") for caption in captions]
    if len(cleaned) == 1:
        return cleaned[0] + "."
    if len(cleaned) == 2:
        return f"First, {cleaned[0]}; then, {cleaned[1]}."
    middle = "; then, ".join(cleaned[1:-1])
    return f"First, {cleaned[0]}; then, {middle}; finally, {cleaned[-1]}."


def build_video_summaries(
    sources: Sequence[dict[str, Any]],
    windows: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    segments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    window_lookup = {str(row["episode_id"]): row for row in windows}
    annotations_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        window = window_lookup.get(str(annotation.get("episode_id") or ""))
        if window is not None:
            annotations_by_clip[str(window["clip_uid"])].append({**annotation, "_window": window})
    segments_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        segments_by_clip[str(row["clip_uid"])].append(row)
    result: list[dict[str, Any]] = []
    for source in sources:
        clip_uid = str(source["clip_uid"])
        clip_segments = sorted(segments_by_clip[clip_uid], key=lambda row: float(row["start_sec"]))
        clip_annotations = sorted(
            annotations_by_clip[clip_uid], key=lambda row: float(row["_window"]["window_start_sec"])
        )
        segment_captions = _deduplicate_captions(str(row.get("label") or "") for row in clip_segments)
        window_captions = _deduplicate_captions(
            str(row.get("video_level_instruction") or "") for row in clip_annotations
        )
        caption_inputs = window_captions or segment_captions
        representatives = _representative_captions(caption_inputs)
        long_caption = compose_long_video_caption(caption_inputs)
        result.append({
            "clip_uid": clip_uid,
            "video_uid": source.get("video_uid"),
            "source_clip_path": source["source_clip_path"],
            "source_duration_sec": source["duration_sec"],
            "long_video_caption": long_caption,
            "summary_caption": long_caption,
            "summary_caption_source": "derived_from_qwen_window_captions" if window_captions else "derived_from_qwen_segment_labels",
            "summary_caption_generation": "deterministic_chronological_composition",
            "representative_captions": representatives,
            "segment_captions": segment_captions,
            "qwen_window_captions": window_captions,
            "segment_count": len(clip_segments),
            "training_segment_count": sum(bool(row.get("training_eligible")) for row in clip_segments),
            "review_segment_count": sum(not bool(row.get("training_eligible")) for row in clip_segments),
            "label_source": "qwen_visual_only",
            "narration_used": False,
        })
    return result


def build_review_html(
    rows: Sequence[dict[str, Any]], pilot_dir: Path, summary: dict[str, Any],
    video_summaries: Sequence[dict[str, Any]],
) -> None:
    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clip[str(row["clip_uid"])].append(row)
    summaries_by_clip = {str(row["clip_uid"]): row for row in video_summaries}
    sections: list[str] = []
    for clip_uid in sorted(by_clip):
        cards: list[str] = []
        for row in by_clip[clip_uid]:
            path = Path(str(row["segment_clip_path"]))
            try:
                media = path.resolve().relative_to(pilot_dir.resolve())
                video = f'<video controls preload="none" src="{html.escape(str(media))}"></video>'
            except ValueError:
                video = "<p class='bad'>missing media</p>"
            flags = ", ".join(row.get("quality_flags") or []) or "none"
            cards.append(f"""
<article class="{'accept' if row.get('training_eligible') else 'review'}">
  <h3>{html.escape(row['segment_id'])}</h3>{video}
  <p><b>{row['start_sec']:.3f}–{row['end_sec']:.3f}s</b> · {html.escape(str(row.get('fine_action') or row['action']))} (coarse={html.escape(row['action'])}) · object={html.escape(str(row.get('object') or 'none'))}</p>
  <p class="label">{html.escape(row['label'])}</p>
  <p>training_eligible={str(bool(row.get('training_eligible'))).lower()} · flags={html.escape(flags)}</p>
</article>""")
        long_caption = str(summaries_by_clip.get(clip_uid, {}).get("long_video_caption") or "")
        sections.append(
            f"<section><h2>{html.escape(clip_uid)} · {len(cards)} subtasks</h2>"
            f"<p class='label'><b>Long-video caption:</b> {html.escape(long_caption)}</p>{''.join(cards)}</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Long-video subtasks</title>
<style>body{{font-family:system-ui;margin:24px;background:#f5f6f8}}section{{margin:30px 0}}article{{background:white;padding:14px;margin:12px 0;border-left:6px solid #d99;border-radius:7px}}article.accept{{border-color:#49a66f}}video{{width:min(640px,100%);background:#111}}.label{{font-size:1.1rem}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>Long-video → short subtasks</h1>
<p>Labels and boundaries are Qwen visual predictions. Narration was not used. Videos load only after click.</p>
<details><summary>summary</summary><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre></details>
{''.join(sections)}</body></html>"""
    (pilot_dir / "review_index.html").write_text(page)


def finalize(args: argparse.Namespace) -> None:
    pilot_dir = Path(args.pilot_dir).resolve()
    sources = read_jsonl(pilot_dir / "source_videos.jsonl")
    windows = read_jsonl(pilot_dir / "analysis_windows.jsonl")
    annotations_path = Path(args.annotations).resolve() if args.annotations else pilot_dir / "qwen" / "annotations.jsonl"
    annotations = read_jsonl(annotations_path)
    expected = {str(row["episode_id"]) for row in windows if row.get("export_ok")}
    completed = {str(row.get("episode_id") or "") for row in annotations}
    pending = sorted(expected - completed)
    if pending and not args.allow_incomplete:
        raise RuntimeError(f"Qwen annotations incomplete: {len(pending)} pending windows")
    candidates, unknown = annotation_candidates(sources, windows, annotations)
    write_jsonl(pilot_dir / "candidate_subtasks.jsonl", candidates)
    candidates_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        candidates_by_clip[str(row["clip_uid"])].append(row)
    timeline: list[dict[str, Any]] = []
    for source in sources:
        timeline.extend(stitch_timeline(
            source, candidates_by_clip[str(source["clip_uid"])], args.min_final_segment_sec
        ))
    below_min = [
        row for row in timeline if float(row["duration_sec"]) < args.min_final_segment_sec
    ]
    if below_min:
        raise RuntimeError(f"{len(below_min)} timeline segments are below the hard minimum")
    exported: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(export_final_segment, row, pilot_dir, args.overwrite) for row in timeline]
        for future in as_completed(futures):
            exported.append(future.result())
    exported.sort(key=lambda row: (row["clip_uid"], row["start_sec"], row["end_sec"]))
    write_jsonl(pilot_dir / "final_timeline.jsonl", exported)
    training = [row for row in exported if row.get("training_eligible") and row.get("export_ok")]
    review = [row for row in exported if not row.get("training_eligible") or not row.get("export_ok")]
    write_jsonl(pilot_dir / "training_segments.jsonl", training)
    write_jsonl(pilot_dir / "review_queue.jsonl", review)
    video_summaries = build_video_summaries(sources, windows, annotations, exported)
    write_jsonl(pilot_dir / "video_summaries.jsonl", video_summaries)
    write_json(
        pilot_dir / "video_summaries.json",
        {"videos": video_summaries, "count": len(video_summaries)},
    )
    decisions = [{
        "segment_id": row["segment_id"], "review_status": "pending",
        "boundary_correct": None, "label_correct": None, "corrected_start_sec": None,
        "corrected_end_sec": None, "corrected_label": None, "reviewer_notes": "",
    } for row in exported]
    write_jsonl(pilot_dir / "review_decisions.jsonl", decisions)
    total_duration = sum(float(row["duration_sec"]) for row in exported)
    uncovered_duration = sum(float(row.get("uncovered_duration_sec") or 0.0) for row in exported)
    action_duration: Counter[str] = Counter()
    for row in exported:
        action_duration[str(row["action"])] += float(row["duration_sec"])
    summary = {
        "workflow": "long_video_subtask_v1",
        "datasets": sorted({str(row.get("dataset") or "generic_mp4") for row in sources}),
        "label_source": "qwen_visual_only",
        "narration_used": False,
        "source_videos": len(sources),
        "expected_analysis_windows": len(expected),
        "completed_analysis_windows": len(expected & completed),
        "pending_analysis_windows": len(pending),
        "unknown_annotation_ids": unknown,
        "candidate_subtasks": len(candidates),
        "final_segments": len(exported),
        "training_segments": len(training),
        "review_segments": len(review),
        "min_final_segment_sec": args.min_final_segment_sec,
        "segments_below_minimum": len(below_min),
        "video_summary_captions": len(video_summaries),
        "export_failures": sum(not row.get("export_ok") for row in exported),
        "total_source_duration_sec": round(sum(float(row["duration_sec"]) for row in sources), 3),
        "timeline_duration_sec": round(total_duration, 3),
        "uncovered_duration_sec": round(uncovered_duration, 3),
        "uncovered_ratio": round(uncovered_duration / total_duration, 4) if total_duration else 0.0,
        "idle_ratio": round(action_duration.get("idle", 0.0) / total_duration, 4) if total_duration else 0.0,
        "action_counts": dict(Counter(str(row["action"]) for row in exported)),
        "action_duration_sec": {key: round(value, 3) for key, value in action_duration.items()},
        "quality_flag_counts": dict(Counter(
            flag for row in exported for flag in (row.get("quality_flags") or [])
        )),
    }
    write_json(pilot_dir / "timeline_summary.json", summary)
    build_review_html(exported, pilot_dir, summary, video_summaries)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["export_failures"]:
        raise RuntimeError(f"{summary['export_failures']} final segment exports failed")


def status(args: argparse.Namespace) -> None:
    pilot_dir = Path(args.pilot_dir).resolve()
    windows = read_jsonl(pilot_dir / "analysis_windows.jsonl")
    annotations = read_jsonl(pilot_dir / "qwen" / "annotations.jsonl")
    errors = read_jsonl(pilot_dir / "qwen" / "errors.jsonl")
    expected = {str(row["episode_id"]) for row in windows if row.get("export_ok")}
    completed = {str(row.get("episode_id") or "") for row in annotations}
    result = {
        "expected": len(expected), "completed": len(expected & completed),
        "pending": len(expected - completed), "errors": len(errors),
        "pending_ids": sorted(expected - completed),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="select sources and export analysis windows")
    source_group = prepare_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-manifest",
        help="generic JSONL manifest with source_clip_path and optional dataset/clip_uid/task_hint",
    )
    source_group.add_argument(
        "--ego4d-root", default=DEFAULT_EGO4D_ROOT,
        help="Ego4D root containing ego4d.json and v2/clips (or set EGO4D_ROOT)",
    )
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--num-videos", type=int, default=None)
    prepare_parser.add_argument("--min-duration-sec", type=float, default=3.0)
    prepare_parser.add_argument("--max-duration-sec", type=float, default=3600.0)
    prepare_parser.add_argument("--window-sec", type=float, default=30.0)
    prepare_parser.add_argument("--overlap-sec", type=float, default=5.0)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--workers", type=int, default=8)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    finalize_parser = sub.add_parser("finalize", help="stitch Qwen predictions and export final clips")
    finalize_parser.add_argument("--work-dir", "--pilot-dir", dest="pilot_dir", required=True)
    finalize_parser.add_argument("--annotations", default=None)
    finalize_parser.add_argument("--min-final-segment-sec", type=float, default=3.0)
    finalize_parser.add_argument("--workers", type=int, default=8)
    finalize_parser.add_argument("--allow-incomplete", action="store_true")
    finalize_parser.add_argument("--overwrite", action="store_true")
    finalize_parser.set_defaults(func=finalize)

    status_parser = sub.add_parser("status", help="report Qwen window completion")
    status_parser.add_argument("--work-dir", "--pilot-dir", dest="pilot_dir", required=True)
    status_parser.set_defaults(func=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
