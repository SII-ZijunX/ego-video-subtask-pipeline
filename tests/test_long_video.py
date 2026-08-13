"""CPU-only tests for the long-video subtask workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_annotation_pipeline import long_video as pilot


def test_analysis_windows_cover_full_video() -> None:
    windows = pilot.analysis_windows(480.0, 30.0, 5.0)
    assert windows[0] == (0.0, 30.0)
    assert windows[-1] == (450.0, 480.0)
    assert all(end > start for start, end in windows)
    assert all(current[0] <= previous[1] for previous, current in zip(windows, windows[1:]))
    assert max(end - start for start, end in windows) <= 30.0


def test_short_video_is_one_analysis_window() -> None:
    assert pilot.analysis_windows(12.5, 30.0, 5.0) == [(0.0, 12.5)]


def test_generic_source_manifest_uses_probed_duration(tmp_path, monkeypatch) -> None:
    video = tmp_path / "source video.mp4"
    video.write_bytes(b"fake")
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({
        "dataset": "robot/actionnet",
        "clip_uid": "episode 1",
        "source_clip_path": str(video),
        "task_hint": "pick up a cup",
    }) + "\n")
    monkeypatch.setattr(pilot, "ffprobe_duration", lambda _: 12.5)
    rows = pilot.load_source_manifest(manifest, 3.0, 30.0)
    assert rows[0]["dataset"] == "robot-actionnet"
    assert rows[0]["clip_uid"] == "episode-1"
    assert rows[0]["duration_sec"] == 12.5
    assert rows[0]["narration_used"] is False


def test_generic_source_manifest_rejects_below_minimum(tmp_path, monkeypatch) -> None:
    video = tmp_path / "short.mp4"
    video.write_bytes(b"fake")
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({"source_clip_path": str(video)}) + "\n")
    monkeypatch.setattr(pilot, "ffprobe_duration", lambda _: 2.9)
    try:
        pilot.load_source_manifest(manifest, 3.0, 30.0)
    except ValueError as error:
        assert "outside [3.0, 30.0]" in str(error)
    else:
        raise AssertionError("expected duration validation failure")


def test_generic_source_manifest_supports_episode_range_in_shared_container(
    tmp_path, monkeypatch
) -> None:
    video = tmp_path / "shared-container.mp4"
    video.write_bytes(b"fake")
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text(json.dumps({
        "dataset": "airoa-moma",
        "clip_uid": "episode-7",
        "source_clip_path": str(video),
        "source_start_sec": 101.25,
        "source_end_sec": 114.75,
        "reference_caption": "dataset text must remain out of the prompt",
        "task_hint": None,
    }) + "\n")
    monkeypatch.setattr(pilot, "ffprobe_duration", lambda _: 500.0)

    rows = pilot.load_source_manifest(manifest, 3.0, 30.0)

    assert rows[0]["duration_sec"] == 13.5
    assert rows[0]["source_seek_offset_sec"] == 101.25
    assert rows[0]["clip_video_start_sec"] == 101.25
    assert rows[0]["clip_video_end_sec"] == 114.75
    assert rows[0]["task_hint"] is None
    assert rows[0]["narration_used"] is False


def test_export_analysis_window_adds_episode_offset(tmp_path, monkeypatch) -> None:
    source = tmp_path / "shared.mp4"
    source.write_bytes(b"fake")
    commands = []

    def fake_run(command):
        commands.append(command)
        target = Path(command[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"window")
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(pilot, "run", fake_run)
    monkeypatch.setattr(pilot, "ffprobe_duration", lambda _: 8.0)
    row = {
        "dataset": "airoa-moma", "episode_id": "win", "clip_uid": "episode-7",
        "video_uid": "episode-7", "source_clip_path": str(source),
        "source_duration_sec": 13.5, "source_seek_offset_sec": 101.25,
        "window_index": 1, "window_start_sec": 5.5, "window_end_sec": 13.5,
        "window_duration_sec": 8.0, "overlap_sec": 2.0, "task_hint": None,
    }

    result = pilot.export_analysis_window(row, tmp_path / "run", overwrite=True)

    assert result["export_ok"]
    assert commands[0][commands[0].index("-ss") + 1] == "106.750"
    metadata = json.loads((tmp_path / "run" / "dataset" / "win" / "metadata.json").read_text())
    assert metadata["container_window_start_sec"] == 106.75
    assert metadata["container_window_end_sec"] == 114.75
    assert metadata["task_hint"] is None


def test_prepare_cleanup_removes_only_stale_generated_episodes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stale_media = run / "analysis_clips" / "old-clip" / "old-window.mp4"
    stale_media.parent.mkdir(parents=True)
    stale_media.write_bytes(b"old")
    stale_episode = run / "dataset" / "old-window"
    stale_episode.mkdir(parents=True)
    (stale_episode / "metadata.json").write_text(json.dumps({
        "episode_id": "old-window",
        "source": "droid-raw_long_window",
        "label_source": "qwen_visual_only",
        "cameras": [{"path": str(stale_media)}],
    }))
    active_episode = run / "dataset" / "active-window"
    active_episode.mkdir(parents=True)
    (active_episode / "metadata.json").write_text(json.dumps({
        "episode_id": "active-window",
        "source": "droid-raw_long_window",
        "label_source": "qwen_visual_only",
        "cameras": [],
    }))
    unmanaged = run / "dataset" / "manual"
    unmanaged.mkdir(parents=True)
    (unmanaged / "metadata.json").write_text(json.dumps({
        "episode_id": "manual", "source": "manual",
    }))
    qwen = run / "qwen"
    (qwen / "episodes" / "old-window").mkdir(parents=True)
    (qwen / "episodes" / "active-window").mkdir(parents=True)
    qwen.mkdir(exist_ok=True)
    (qwen / "annotations.jsonl").write_text(
        json.dumps({"episode_id": "old-window"}) + "\n"
        + json.dumps({"episode_id": "active-window"}) + "\n"
    )

    result = pilot.prune_stale_prepared_episodes(run, {"active-window"})

    assert result == {
        "removed_stale_episodes": 1,
        "removed_stale_analysis_clips": 1,
        "removed_stale_qwen_episode_dirs": 1,
    }
    assert not stale_episode.exists()
    assert not stale_media.exists()
    assert active_episode.is_dir()
    assert unmanaged.is_dir()
    assert (qwen / "episodes" / "active-window").is_dir()
    assert [row["episode_id"] for row in pilot.read_jsonl(qwen / "annotations.jsonl")] == [
        "active-window"
    ]


def test_annotation_candidates_map_local_to_source_time() -> None:
    sources = [{
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 100.0,
    }]
    windows = [{
        "episode_id": "win", "clip_uid": "clip", "window_index": 2,
        "window_start_sec": 50.0, "window_end_sec": 80.0,
        "window_duration_sec": 30.0, "overlap_sec": 5.0, "export_ok": True,
    }]
    annotations = [{
        "episode_id": "win", "episode_quality_flags": ["out_of_vocabulary_action"],
        "annotation_metadata": {"model": "qwen", "prompt_version": "v1"},
        "subtasks": [{
            "subtask_id": 0, "start_time_sec": 3.0, "end_time_sec": 10.0,
            "action": "move", "object": "cup", "instruction": "Move the cup.",
            "training_eligible": True, "quality_flags": [],
        }],
    }]
    rows, unknown = pilot.annotation_candidates(sources, windows, annotations)
    assert not unknown
    assert rows[0]["start_sec"] == 53.0
    assert rows[0]["end_sec"] == 60.0
    assert rows[0]["training_eligible"]
    assert rows[0]["narration_used"] is False

    assert rows[0]["episode_quality_flags"] == ["out_of_vocabulary_action"]

def test_stitch_merges_same_semantics_across_overlap() -> None:
    source = {
        "dataset": "actionnet", "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 20.0,
    }
    base = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "training_eligible": True, "quality_flags": [], "edge_margin_sec": 4.0,
        "label_source": "qwen_visual_only",
    }
    candidates = [
        {**base, "candidate_id": "a", "episode_id": "w0", "start_sec": 0.0,
         "end_sec": 8.0, "action": "move", "object": "cup", "instruction": "Move the cup."},
        {**base, "candidate_id": "b", "episode_id": "w1", "start_sec": 5.0,
         "end_sec": 12.0, "action": "move", "object": "cup", "instruction": "Move the cup."},
        {**base, "candidate_id": "c", "episode_id": "w1", "start_sec": 12.0,
         "end_sec": 20.0, "action": "open", "object": "drawer", "instruction": "Open the drawer."},
    ]
    timeline = pilot.stitch_timeline(source, candidates)
    assert [(row["start_sec"], row["end_sec"], row["action"]) for row in timeline] == [
        (0.0, 12.0, "move"), (12.0, 20.0, "open"),
    ]
    assert timeline[0]["candidate_ids"] == ["a", "b"]
    assert timeline[0]["dataset"] == "actionnet"
    assert timeline[0]["segment_id"] == "actionnet__clip__seg0000"


def test_stitch_is_complete_and_non_overlapping_with_gap() -> None:
    source = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 20.0,
    }
    candidate = {
        "candidate_id": "a", "episode_id": "w0", "clip_uid": "clip",
        "start_sec": 3.0, "end_sec": 8.0, "action": "wipe", "object": "table",
        "instruction": "Wipe the table.", "training_eligible": True,
        "quality_flags": [], "edge_margin_sec": 3.0, "label_source": "qwen_visual_only",
    }
    timeline = pilot.stitch_timeline(source, [candidate])
    assert timeline[0]["start_sec"] == 0.0
    assert timeline[-1]["end_sec"] == 20.0
    assert all(a["end_sec"] == b["start_sec"] for a, b in zip(timeline, timeline[1:]))
    assert sum(row["duration_sec"] for row in timeline) == 20.0
    uncovered = [row for row in timeline if "uncovered_by_qwen" in row["quality_flags"]]
    assert len(uncovered) == 2
    assert all(not row["training_eligible"] for row in uncovered)
    assert sum(row["uncovered_duration_sec"] for row in timeline) == 15.0

def test_open_fine_action_is_trainable_and_prevents_wrong_merge() -> None:
    source = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 10.0,
    }
    base = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "action": "other", "object": "vegetable", "training_eligible": True,
        "quality_flags": [], "edge_margin_sec": 4.0, "label_source": "qwen_visual_only",
    }
    candidates = [
        {**base, "candidate_id": "a", "episode_id": "w0", "start_sec": 0.0,
         "end_sec": 6.0, "fine_action": "peel", "instruction": "Peel the vegetable."},
        {**base, "candidate_id": "b", "episode_id": "w0", "start_sec": 6.0,
         "end_sec": 10.0, "fine_action": "wash", "instruction": "Wash the vegetable."},
    ]
    timeline = pilot.stitch_timeline(source, candidates)
    assert [(row["fine_action"], row["training_eligible"]) for row in timeline] == [
        ("peel", True), ("wash", True),
    ]


def test_stitch_enforces_three_second_minimum() -> None:
    source = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 10.0,
    }
    base = {
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "training_eligible": True, "quality_flags": [], "edge_margin_sec": 2.0,
        "label_source": "qwen_visual_only",
    }
    candidates = [
        {**base, "candidate_id": "a", "episode_id": "w0", "start_sec": 0.0,
         "end_sec": 4.0, "action": "move", "object": "cup", "instruction": "Move the cup."},
        {**base, "candidate_id": "b", "episode_id": "w0", "start_sec": 4.0,
         "end_sec": 5.5, "action": "open", "object": "drawer", "instruction": "Open the drawer."},
        {**base, "candidate_id": "c", "episode_id": "w0", "start_sec": 5.5,
         "end_sec": 10.0, "action": "close", "object": "drawer", "instruction": "Close the drawer."},
    ]
    timeline = pilot.stitch_timeline(source, candidates, min_segment_sec=3.0)
    assert all(row["duration_sec"] >= 3.0 for row in timeline)
    assert all(row["narration_used"] is False for row in timeline)
    assert timeline[0]["start_sec"] == 0.0
    assert timeline[-1]["end_sec"] == 10.0
    assert all(left["end_sec"] == right["start_sec"] for left, right in zip(timeline, timeline[1:]))
    merged = [row for row in timeline if "merged_for_min_duration" in row["quality_flags"]]
    assert len(merged) == 1
    assert not merged[0]["training_eligible"]


def test_long_video_summary_caption_is_chronological() -> None:
    sources = [{
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
        "duration_sec": 10.0,
    }]
    windows = [{
        "episode_id": "win", "clip_uid": "clip", "window_start_sec": 0.0,
    }]
    annotations = [{
        "episode_id": "win", "video_level_instruction": "Handle a cup and drawer.",
    }]
    segments = [
        {"clip_uid": "clip", "start_sec": 0.0, "label": "Pick up the cup.",
         "training_eligible": True},
        {"clip_uid": "clip", "start_sec": 4.0, "label": "Open the drawer.",
         "training_eligible": True},
        {"clip_uid": "clip", "start_sec": 7.0, "label": "Close the drawer.",
         "training_eligible": False},
    ]
    summaries = pilot.build_video_summaries(sources, windows, annotations, segments)
    assert len(summaries) == 1
    assert summaries[0]["long_video_caption"] == "Handle a cup and drawer."
    assert summaries[0]["segment_count"] == 3
    assert summaries[0]["summary_caption_source"] == "derived_from_qwen_window_captions"


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"[ok] {name}")
