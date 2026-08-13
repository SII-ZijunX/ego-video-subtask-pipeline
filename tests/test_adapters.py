from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_annotation_pipeline.adapters import (
    prepare_droid_video_manifest,
    prepare_lerobot_v2_video_manifest,
    prepare_lerobot_v3_video_manifest,
)


def test_prepare_lerobot_v3_manifest_keeps_tasks_out_of_prompt(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "dataset"
    episodes = root / "meta" / "episodes" / "chunk-000"
    videos = root / "videos" / "observation.image.head" / "chunk-000"
    episodes.mkdir(parents=True)
    videos.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "features": {"observation.image.head": {"dtype": "video"}}
    }))
    table = pa.table({
        "episode_index": [7],
        "videos/observation.image.head/chunk_index": [0],
        "videos/observation.image.head/file_index": [2],
        "videos/observation.image.head/from_timestamp": [101.25],
        "videos/observation.image.head/to_timestamp": [114.75],
        "tasks": [["Open the drawer."]],
    })
    pq.write_table(table, episodes / "file-000.parquet")
    (videos / "file-002.mp4").write_bytes(b"container")
    output = tmp_path / "sources.jsonl"

    summary = prepare_lerobot_v3_video_manifest(
        root, output, dataset="airoa-moma", camera_key="observation.image.head"
    )

    row = json.loads(output.read_text())
    assert summary["prepared"] == 1
    assert summary["task_hint_non_null"] == 0
    assert row["source_start_sec"] == 101.25
    assert row["source_end_sec"] == 114.75
    assert row["task_hint"] is None
    assert row["reference_caption"] == "Open the drawer."
    assert row["reference_policy"] == "held_out_from_visual_prompt"
    assert row["narration_used"] is False


def test_prepare_lerobot_v2_manifest_keeps_tasks_out_of_prompt(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    meta = root / "meta"
    video = root / "videos" / "chunk-001" / "observation.images.cam_high"
    meta.mkdir(parents=True)
    video.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 50, "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }))
    (meta / "episodes.jsonl").write_text(json.dumps({
        "episode_index": 1083, "length": 312, "tasks": ["Reposition the bottle."]
    }) + "\n")
    (video / "episode_001083.mp4").write_bytes(b"episode")
    output = tmp_path / "sources.jsonl"

    summary = prepare_lerobot_v2_video_manifest(
        root, output, dataset="robopro", camera_key="observation.images.cam_high"
    )

    row = json.loads(output.read_text())
    assert summary["prepared"] == 1
    assert summary["task_hint_non_null"] == 0
    assert row["duration_sec_reference"] == 6.24
    assert row["task_hint"] is None
    assert row["reference_caption"] == "Reposition the bottle."
    assert row["reference_policy"] == "held_out_from_visual_prompt"
    assert row["narration_used"] is False


def test_prepare_droid_manifest_keeps_task_out_of_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "droid"
    episode = root / "AUTOLab" / "success" / "2023-10-21" / "episode"
    videos = episode / "recordings" / "MP4"
    videos.mkdir(parents=True)
    (episode / "metadata_example.json").write_text(json.dumps({
        "uuid": "AUTOLab+user+2023-10-21-19h-37m-47s",
        "current_task": "Put the brick in the drawer.",
        "success": True,
        "wrist_cam_serial": "18026681",
        "ext1_cam_serial": "22008760",
        "ext2_cam_serial": "24400334",
    }))
    wrist = videos / "18026681.mp4"
    wrist.write_bytes(b"video")
    monkeypatch.setattr(
        "video_annotation_pipeline.adapters._probe_video_duration",
        lambda path: 4.184,
    )
    output = tmp_path / "sources.jsonl"

    summary = prepare_droid_video_manifest(root, output, limit=1)

    row = json.loads(output.read_text())
    assert summary["prepared"] == 1
    assert summary["task_hint_non_null"] == 0
    assert row["source_clip_path"] == str(wrist)
    assert row["duration_sec_reference"] == 4.184
    assert row["camera_key"] == "wrist"
    assert row["task_hint"] is None
    assert row["reference_caption"] == "Put the brick in the drawer."
    assert row["reference_policy"] == "held_out_from_visual_prompt"
    assert row["narration_used"] is False


def test_prepare_droid_manifest_enforces_duration_and_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "droid"
    episode = root / "lab" / "success" / "day" / "episode"
    videos = episode / "recordings" / "MP4"
    videos.mkdir(parents=True)
    (episode / "metadata_example.json").write_text(json.dumps({
        "uuid": "example", "success": True, "ext1_cam_serial": "external",
    }))
    external = videos / "external.mp4"
    external.write_bytes(b"video")
    monkeypatch.setattr(
        "video_annotation_pipeline.adapters._probe_video_duration",
        lambda path: 2.999,
    )
    output = tmp_path / "sources.jsonl"

    summary = prepare_droid_video_manifest(root, output, camera="ext1")

    assert summary["prepared"] == 0
    assert summary["skipped_duration"] == 1
    assert output.read_text() == ""


def test_prepare_droid_manifest_defaults_to_success_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "droid"
    for name, success in (("failure", False), ("success", True)):
        episode = root / "lab" / name / "day" / "episode"
        videos = episode / "recordings" / "MP4"
        videos.mkdir(parents=True)
        (episode / f"metadata_{name}.json").write_text(json.dumps({
            "uuid": name, "success": success, "wrist_cam_serial": "wrist",
        }))
        (videos / "wrist.mp4").write_bytes(b"video")
    monkeypatch.setattr(
        "video_annotation_pipeline.adapters._probe_video_duration",
        lambda path: 4.0,
    )
    output = tmp_path / "sources.jsonl"

    summary = prepare_droid_video_manifest(root, output)

    row = json.loads(output.read_text())
    assert summary["outcome"] == "success"
    assert summary["metadata_scanned"] == 1
    assert summary["skipped_outcome"] == 0
    assert summary["prepared"] == 1
    assert row["video_uid"] == "success"
    assert row["trajectory_success"] is True
