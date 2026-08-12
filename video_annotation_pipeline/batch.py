"""Deterministic batch execution with resume/skip/retry-failed semantics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .backends import create_backend
from .config import PipelineConfig
from .runner import AnnotationRunError, annotate_episode


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(path)


def discover_episodes(dataset_dir: Path | str) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    return sorted(path.parent for path in dataset_dir.glob("*/metadata.json"))


def _episode_id(episode_dir: Path) -> str:
    try:
        return str(json.loads((episode_dir / "metadata.json").read_text())["episode_id"])
    except Exception:
        return episode_dir.name


def annotate_batch(
    dataset_dir: Path | str,
    output_dir: Path | str,
    config: PipelineConfig,
    resume: bool = False,
    skip_existing: bool = False,
    retry_failed: bool = False,
) -> dict[str, int]:
    output_dir = Path(output_dir)
    annotations_path = output_dir / "annotations.jsonl"
    errors_path = output_dir / "errors.jsonl"
    annotations = {row["episode_id"]: row for row in read_jsonl(annotations_path)}
    errors = {row["episode_id"]: row for row in read_jsonl(errors_path)}
    episodes = discover_episodes(dataset_dir)
    episode_ids = [_episode_id(episode_dir) for episode_dir in episodes]
    episode_id_counts = Counter(episode_ids)
    duplicates = sorted(episode_id for episode_id, count in episode_id_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate episode_id values: {duplicates}")
    backend = create_backend(config.backend)
    completed = failed = skipped = 0

    for episode_dir in episodes:
        episode_id = _episode_id(episode_dir)
        if retry_failed and episode_id not in errors:
            skipped += 1
            continue
        if (resume or skip_existing) and episode_id in annotations:
            skipped += 1
            continue
        try:
            annotation = annotate_episode(
                episode_dir,
                output_dir / "episodes" / episode_id,
                config,
                backend=backend,
            )
            annotations[episode_id] = annotation.model_dump(mode="json")
            errors.pop(episode_id, None)
            completed += 1
        except AnnotationRunError as exc:
            errors[episode_id] = {
                "episode_id": episode_id,
                "episode_dir": str(episode_dir.resolve()),
                "error": str(exc),
                "quality_flags": exc.flags,
            }
            failed += 1
        # Persist after every episode so preemption loses at most current work.
        write_jsonl(annotations_path, [annotations[key] for key in sorted(annotations)])
        write_jsonl(errors_path, [errors[key] for key in sorted(errors)])

    return {"completed": completed, "failed": failed, "skipped": skipped, "total": len(annotations)}
