"""Adapters from existing VITRA/Ego4D manifests to the episode input contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _probe_video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _iter_droid_metadata(root: Path) -> Iterable[Path]:
    """Yield DROID trajectory metadata deterministically without indexing it all."""
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.startswith("metadata_") and filename.endswith(".json"):
                yield Path(directory) / filename


def prepare_droid_video_manifest(
    dataset_root: Path | str,
    output_manifest: Path | str,
    *,
    dataset: str = "droid-raw",
    camera: str = "wrist",
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict[str, Any]:
    """Build visual-only source rows from raw DROID trajectory directories.

    Each trajectory contains synchronized wrist/ext1/ext2 MP4 files and a
    metadata JSON. The dataset task is retained only as held-out reference
    text; it is never copied to ``task_hint``.
    """
    camera_serial_key = {
        "wrist": "wrist_cam_serial",
        "ext1": "ext1_cam_serial",
        "ext2": "ext2_cam_serial",
    }.get(camera)
    if camera_serial_key is None:
        raise ValueError("camera must be one of: wrist, ext1, ext2")
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if min_duration_sec < 0 or max_duration_sec < min_duration_sec:
        raise ValueError("invalid duration range")

    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DROID dataset root does not exist: {root}")

    rows: list[dict[str, Any]] = []
    scanned = eligible_seen = skipped_duration = missing_video = invalid_metadata = 0
    probe_failures = 0
    stop = offset + limit if limit else None
    for metadata_path in _iter_droid_metadata(root):
        scanned += 1
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            invalid_metadata += 1
            continue
        serial = str(metadata.get(camera_serial_key) or "").strip()
        video_path = metadata_path.parent / "recordings" / "MP4" / f"{serial}.mp4"
        if not serial or not video_path.is_file():
            missing_video += 1
            continue
        try:
            duration = round(_probe_video_duration(video_path), 3)
        except (OSError, subprocess.SubprocessError, ValueError):
            probe_failures += 1
            continue
        if not min_duration_sec <= duration <= max_duration_sec:
            skipped_duration += 1
            continue
        eligible_index = eligible_seen
        eligible_seen += 1
        if eligible_index < offset:
            continue
        if stop is not None and eligible_index >= stop:
            break

        video_uid = str(metadata.get("uuid") or metadata_path.parent.name)
        clip_uid = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in video_uid
        ).strip("-")
        reference_caption = (
            str(metadata.get("current_task") or "").strip() or None
            if include_reference_caption else None
        )
        rows.append({
            "dataset": dataset,
            "clip_uid": f"{dataset}-{clip_uid}",
            "video_uid": video_uid,
            "source_clip_path": str(video_path),
            "duration_sec_reference": duration,
            "camera_key": camera,
            "task_hint": None,
            "reference_caption": reference_caption,
            "reference_policy": "held_out_from_visual_prompt",
            "label_source": "qwen_visual_only",
            "narration_used": False,
        })
        if limit and len(rows) >= limit:
            break

    _write_jsonl(output, rows)
    summary = {
        "dataset": dataset,
        "dataset_root": str(root),
        "camera": camera,
        "manifest": str(output),
        "metadata_scanned": scanned,
        "eligible_seen": eligible_seen,
        "prepared": len(rows),
        "skipped_duration": skipped_duration,
        "missing_video": missing_video,
        "invalid_metadata": invalid_metadata,
        "probe_failures": probe_failures,
        "task_hint_non_null": sum(row["task_hint"] is not None for row in rows),
        "reference_caption_rows": sum(bool(row["reference_caption"]) for row in rows),
        "narration_used": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def prepare_lerobot_v3_video_manifest(
    dataset_root: Path | str,
    output_manifest: Path | str,
    *,
    dataset: str,
    camera_key: str,
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict[str, Any]:
    """Build generic source rows from a LeRobot v3 episode index.

    LeRobot v3 commonly stores many episodes inside one MP4 container.  The
    generated rows preserve the container path plus per-episode timestamp
    range; downstream Qwen prompts remain visual-only because ``task_hint`` is
    always null.  Dataset task text is optional held-out reference metadata.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "LeRobot manifest preparation requires pyarrow; install the "
            "'lerobot' optional dependency"
        ) from exc

    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_dir = root / "meta" / "episodes"
    if not info_path.is_file() or not episodes_dir.is_dir():
        raise FileNotFoundError(f"incomplete LeRobot v3 dataset under {root}")
    info = json.loads(info_path.read_text())
    feature = (info.get("features") or {}).get(camera_key)
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise ValueError(f"camera_key is not a video feature: {camera_key}")

    prefix = f"videos/{camera_key}"
    columns = [
        "episode_index", f"{prefix}/chunk_index", f"{prefix}/file_index",
        f"{prefix}/from_timestamp", f"{prefix}/to_timestamp", "tasks",
    ]
    available_files = sorted(episodes_dir.glob("chunk-*/*.parquet"))
    if not available_files:
        raise FileNotFoundError(f"no episode parquet files under {episodes_dir}")
    rows: list[dict[str, Any]] = []
    skipped_duration = missing_video = 0
    stop = offset + limit if limit else None
    seen_eligible = 0
    for parquet_path in available_files:
        table = pq.read_table(parquet_path, columns=columns)
        for episode in table.to_pylist():
            start = float(episode[f"{prefix}/from_timestamp"])
            end = float(episode[f"{prefix}/to_timestamp"])
            duration = round(end - start, 3)
            if not min_duration_sec <= duration <= max_duration_sec:
                skipped_duration += 1
                continue
            eligible_index = seen_eligible
            seen_eligible += 1
            if eligible_index < offset:
                continue
            if stop is not None and eligible_index >= stop:
                break
            chunk_index = int(episode[f"{prefix}/chunk_index"])
            file_index = int(episode[f"{prefix}/file_index"])
            video_path = (
                root / "videos" / camera_key / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not video_path.is_file():
                missing_video += 1
                continue
            episode_index = int(episode["episode_index"])
            tasks = [str(value) for value in (episode.get("tasks") or []) if value]
            reference_caption = tasks[0] if include_reference_caption and tasks else None
            rows.append({
                "dataset": dataset,
                "clip_uid": f"{dataset}-episode-{episode_index:06d}",
                "video_uid": f"episode-{episode_index:06d}",
                "source_clip_path": str(video_path),
                "source_start_sec": round(start, 6),
                "source_end_sec": round(end, 6),
                "duration_sec_reference": duration,
                "camera_key": camera_key,
                "task_hint": None,
                "reference_caption": reference_caption,
                "reference_policy": "held_out_from_visual_prompt",
                "label_source": "qwen_visual_only",
                "narration_used": False,
            })
        if stop is not None and seen_eligible >= stop:
            break
    _write_jsonl(output, rows)
    summary = {
        "dataset": dataset,
        "dataset_root": str(root),
        "camera_key": camera_key,
        "manifest": str(output),
        "prepared": len(rows),
        "eligible_seen": seen_eligible,
        "skipped_duration": skipped_duration,
        "missing_video": missing_video,
        "task_hint_non_null": sum(row["task_hint"] is not None for row in rows),
        "reference_caption_rows": sum(bool(row["reference_caption"]) for row in rows),
        "narration_used": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def prepare_lerobot_v2_video_manifest(
    dataset_root: Path | str,
    output_manifest: Path | str,
    *,
    dataset: str,
    camera_key: str,
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict[str, Any]:
    """Build source rows for LeRobot v2/v2.1 one-file-per-episode videos."""
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_manifest).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"incomplete LeRobot v2 dataset under {root}")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version") or "")
    if not version.startswith("v2"):
        raise ValueError(f"expected LeRobot v2 metadata, found {version!r}")
    video_path_template = str(info.get("video_path") or "")
    if not video_path_template:
        raise ValueError("meta/info.json is missing video_path")
    fps = float(info.get("fps") or 0.0)
    chunks_size = int(info.get("chunks_size") or 1000)
    if fps <= 0 or chunks_size <= 0:
        raise ValueError("meta/info.json must contain positive fps/chunks_size")
    episodes = [
        json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()
    ]
    eligible: list[dict[str, Any]] = []
    skipped_duration = missing_video = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode.get("length") or 0)
        duration = round(length / fps, 3) if length > 0 else 0.0
        if not min_duration_sec <= duration <= max_duration_sec:
            skipped_duration += 1
            continue
        episode_chunk = episode_index // chunks_size
        relative = video_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=camera_key,
        )
        video_path = root / relative
        if not video_path.is_file():
            missing_video += 1
            continue
        tasks = [str(value) for value in (episode.get("tasks") or []) if value]
        reference_caption = tasks[0] if include_reference_caption and tasks else None
        eligible.append({
            "dataset": dataset,
            "clip_uid": f"{dataset}-episode-{episode_index:06d}",
            "video_uid": f"episode-{episode_index:06d}",
            "source_clip_path": str(video_path),
            "duration_sec_reference": duration,
            "camera_key": camera_key,
            "task_hint": None,
            "reference_caption": reference_caption,
            "reference_policy": "held_out_from_visual_prompt",
            "label_source": "qwen_visual_only",
            "narration_used": False,
        })
    selected = eligible[offset: offset + limit if limit else None]
    _write_jsonl(output, selected)
    summary = {
        "dataset": dataset,
        "dataset_root": str(root),
        "codebase_version": version,
        "camera_key": camera_key,
        "manifest": str(output),
        "episodes": len(episodes),
        "eligible": len(eligible),
        "prepared": len(selected),
        "skipped_duration": skipped_duration,
        "missing_video": missing_video,
        "task_hint_non_null": sum(row["task_hint"] is not None for row in selected),
        "reference_caption_rows": sum(bool(row["reference_caption"]) for row in selected),
        "narration_used": False,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def prepare_ego4d_dataset(
    segments_path: Path | str,
    output_dir: Path | str,
    offset: int = 0,
    limit: int = 0,
    use_reference_as_task_hint: bool = False,
) -> dict[str, int | str]:
    segments_path = Path(segments_path)
    output_dir = Path(output_dir)
    rows = [
        json.loads(line) for line in segments_path.read_text().splitlines() if line.strip()
    ]
    eligible = [row for row in rows if row.get("export_ok") and row.get("processed_subtask")]
    selected = eligible[offset: offset + limit if limit else None]
    episode_ids: set[str] = set()
    prepared = missing_video = 0
    for row in selected:
        episode_id = str(row.get("subtask_id") or "").strip()
        if not episode_id:
            raise ValueError("each selected segment must have a non-empty subtask_id")
        if episode_id in episode_ids:
            raise ValueError(f"duplicate subtask_id in selected segments: {episode_id}")
        episode_ids.add(episode_id)
        video_path = Path(str(row["processed_subtask"])).resolve()
        if not video_path.is_file():
            missing_video += 1
            continue
        metadata = {
            "episode_id": episode_id,
            "source": str(row.get("dataset") or "ego4d_v2"),
            "task_hint": row.get("reference_caption") if use_reference_as_task_hint else None,
            "cameras": [{
                "name": "main",
                "role": "main",
                "path": str(video_path),
                "time_offset_sec": 0.0,
            }],
            "reference_caption": row.get("reference_caption"),
            "raw_narration_text": row.get("raw_narration_text"),
            "clip_uid": row.get("clip_uid"),
            "video_uid": row.get("video_uid"),
            "source_clip_path": row.get("source_clip_path"),
            "source_start_time_sec": row.get("start_sec"),
            "source_end_time_sec": row.get("end_sec"),
        }
        episode_path = output_dir / episode_id
        episode_path.mkdir(parents=True, exist_ok=True)
        (episode_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
        prepared += 1
    summary = {
        "segments": len(rows),
        "eligible": len(eligible),
        "selected": len(selected),
        "prepared": prepared,
        "missing_video": missing_video,
        "offset": offset,
        "limit": limit,
        "dataset_dir": str(output_dir.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary
