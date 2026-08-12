"""CPU-only tests for the long-video subtask workflow."""

from __future__ import annotations

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
        "clip_uid": "clip", "video_uid": "video", "source_clip_path": "/raw.mp4",
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
