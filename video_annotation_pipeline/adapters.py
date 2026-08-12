"""Adapters from existing VITRA/Ego4D manifests to the episode input contract."""

from __future__ import annotations

import json
from pathlib import Path


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
