"""Aggregate action, duration, object, and quality statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import html
import json

from .batch import read_jsonl
from .schemas import EpisodeAnnotation


def compute_stats(annotations: list[EpisodeAnnotation]) -> dict:
    action_counts: Counter[str] = Counter()
    action_durations: defaultdict[str, float] = defaultdict(float)
    object_counts: Counter[str] = Counter()
    episode_flags: Counter[str] = Counter()
    subtask_counts: list[int] = []
    durations: list[float] = []
    flagged_episode_count = 0
    for annotation in annotations:
        subtask_counts.append(len(annotation.subtasks))
        episode_flags.update(annotation.episode_quality_flags)
        episode_has_flags = bool(annotation.episode_quality_flags)
        for subtask in annotation.subtasks:
            duration = subtask.end_time_sec - subtask.start_time_sec
            action_counts[subtask.action] += 1
            action_durations[subtask.action] += duration
            durations.append(duration)
            if subtask.object:
                object_counts[subtask.object] += 1
            if subtask.quality_flags:
                episode_has_flags = True
                episode_flags.update(subtask.quality_flags)
        flagged_episode_count += int(episode_has_flags)
    action_average = {
        action: round(action_durations[action] / count, 3)
        for action, count in action_counts.items() if count
    }
    total_subtasks = sum(action_counts.values())
    duration_bins = {"<0.5": 0, "0.5-1": 0, "1-2": 0, "2-5": 0, "5-10": 0, ">=10": 0}
    for duration in durations:
        key = "<0.5" if duration < 0.5 else "0.5-1" if duration < 1 else "1-2" if duration < 2 else "2-5" if duration < 5 else "5-10" if duration < 10 else ">=10"
        duration_bins[key] += 1
    subtasks_per_episode_distribution = dict(Counter(subtask_counts))
    return {
        "episodes": len(annotations),
        "subtasks": total_subtasks,
        "action_counts": dict(action_counts.most_common()),
        "action_total_duration_sec": {k: round(v, 3) for k, v in action_durations.items()},
        "action_average_duration_sec": action_average,
        "subtask_duration_sec": durations,
        "subtask_duration_histogram": duration_bins,
        "subtasks_per_episode": subtask_counts,
        "subtasks_per_episode_distribution": subtasks_per_episode_distribution,
        "other_ratio": round(action_counts.get("other", 0) / total_subtasks, 4) if total_subtasks else 0,
        "idle_ratio": round(action_counts.get("idle", 0) / total_subtasks, 4) if total_subtasks else 0,
        "object_counts": dict(object_counts.most_common()),
        "quality_flag_counts": dict(episode_flags.most_common()),
        "episodes_with_quality_flags": flagged_episode_count,
    }


def generate_stats_report(annotations_path: Path | str, output_path: Path | str) -> dict:
    annotations = [EpisodeAnnotation.model_validate(row) for row in read_jsonl(Path(annotations_path))]
    stats = compute_stats(annotations)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    rows = "".join(
        f"<tr><td>{html.escape(action)}</td><td>{count}</td><td>{stats['action_total_duration_sec'].get(action,0)}</td><td>{stats['action_average_duration_sec'].get(action,0)}</td></tr>"
        for action, count in stats["action_counts"].items()
    )
    objects = "".join(f"<li>{html.escape(obj)}: {count}</li>" for obj, count in stats["object_counts"].items())
    flags = "".join(f"<li>{html.escape(flag)}: {count}</li>" for flag, count in stats["quality_flag_counts"].items()) or "<li>None</li>"
    duration_rows = "".join(f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in stats["subtask_duration_histogram"].items())
    episode_rows = "".join(f"<tr><td>{count}</td><td>{episodes}</td></tr>" for count, episodes in sorted(stats["subtasks_per_episode_distribution"].items()))
    output_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Annotation statistics</title><style>body{{font:15px system-ui;margin:30px;max-width:1000px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:7px;text-align:left}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{padding:15px;background:#f2f4f7;border-radius:8px}}.grids{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}</style></head><body><h1>Annotation statistics</h1><div class="cards"><div class="card">Episodes: {stats['episodes']}</div><div class="card">Subtasks: {stats['subtasks']}</div><div class="card">Other: {stats['other_ratio']:.1%}</div><div class="card">Idle: {stats['idle_ratio']:.1%}</div><div class="card">Flagged episodes: {stats['episodes_with_quality_flags']}</div></div><h2>Actions</h2><table><tr><th>Action</th><th>Count</th><th>Total seconds</th><th>Average seconds</th></tr>{rows}</table><div class="grids"><div><h2>Subtask duration distribution</h2><table><tr><th>Seconds</th><th>Count</th></tr>{duration_rows}</table></div><div><h2>Subtasks per episode</h2><table><tr><th>Subtasks</th><th>Episodes</th></tr>{episode_rows}</table></div></div><h2>Objects</h2><ul>{objects}</ul><h2>Quality flags</h2><ul>{flags}</ul></body></html>""")
    return stats
