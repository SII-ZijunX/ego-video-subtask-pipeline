"""CLI command implementations shared by Typer and argparse fallback."""

from __future__ import annotations

import json
from pathlib import Path

from .adapters import (
    prepare_droid_video_manifest,
    prepare_ego4d_dataset,
    prepare_lerobot_v2_video_manifest,
    prepare_lerobot_v3_video_manifest,
)
from .batch import annotate_batch, discover_episodes, read_jsonl
from .config import load_config
from .finalize import finalize_annotations
from .reference_eval import evaluate_references
from .runner import annotate_episode
from .schemas import EpisodeAnnotation, EpisodeMetadata
from .stats import generate_stats_report
from .validator import annotation_is_valid, validate_annotation
from .visualization import generate_episode_report


def annotate_command(input_dir: Path, output_dir: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    annotation = annotate_episode(input_dir, output_dir, config)
    return annotation.model_dump(mode="json")


def annotate_batch_command(
    dataset: Path,
    output: Path,
    config_path: Path,
    resume: bool = False,
    skip_existing: bool = False,
    retry_failed: bool = False,
) -> dict:
    return annotate_batch(
        dataset, output, load_config(config_path),
        resume=resume, skip_existing=skip_existing, retry_failed=retry_failed,
    )


def validate_command(annotations_path: Path, output_path: Path, config_path: Path) -> dict:
    config = load_config(config_path)
    rows = read_jsonl(annotations_path) if annotations_path.suffix == ".jsonl" else [json.loads(annotations_path.read_text())]
    reports = []
    invalid = 0
    for row in rows:
        try:
            annotation = EpisodeAnnotation.model_validate(row)
            normalized = {
                "video_level_instruction": annotation.video_level_instruction,
                "subtasks": [subtask.model_dump(mode="json") for subtask in annotation.subtasks],
            }
            flags, retry_reasons, coverage = validate_annotation(
                normalized, annotation.duration_sec, config
            )
            valid = annotation_is_valid(flags, retry_reasons)
            if not valid:
                invalid += 1
            reports.append({
                "episode_id": annotation.episode_id,
                "valid": valid,
                "quality_flags": flags,
                "retry_reasons": retry_reasons,
                "coverage_ratio": coverage,
            })
        except Exception as exc:
            invalid += 1
            reports.append({"episode_id": row.get("episode_id"), "valid": False, "error": str(exc)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    return {"annotations": len(rows), "invalid": invalid, "output": str(output_path)}


def visualize_command(dataset: Path, annotations_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dirs: dict[str, Path] = {}
    metadata_by_id: dict[str, EpisodeMetadata] = {}
    for episode_dir in discover_episodes(dataset):
        metadata = EpisodeMetadata.model_validate_json((episode_dir / "metadata.json").read_text())
        episode_dirs[metadata.episode_id] = episode_dir
        metadata_by_id[metadata.episode_id] = metadata
    links = []
    missing = []
    for row in read_jsonl(annotations_path):
        annotation = EpisodeAnnotation.model_validate(row)
        if annotation.episode_id not in episode_dirs:
            missing.append(annotation.episode_id)
            continue
        report_path = output_dir / annotation.episode_id / "review.html"
        raw_candidates = sorted(
            (annotations_path.parent / "episodes" / annotation.episode_id).glob(
                "raw_response_attempt_*.txt"
            )
        )
        raw_response = raw_candidates[-1].read_text() if raw_candidates else ""
        generate_episode_report(
            annotation,
            metadata_by_id[annotation.episode_id],
            episode_dirs[annotation.episode_id],
            report_path,
            raw_response,
        )
        links.append((annotation.episode_id, report_path.relative_to(output_dir)))
    index = "".join(f'<li><a href="{path}">{episode_id}</a></li>' for episode_id, path in links)
    (output_dir / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><h1>Annotation review</h1><ul>{index}</ul>"
    )
    return {"reports": len(links), "missing_episode_inputs": missing, "index": str(output_dir / "index.html")}


def stats_command(annotations_path: Path, output_path: Path) -> dict:
    return generate_stats_report(annotations_path, output_path)


def prepare_ego4d_command(
    segments: Path,
    output: Path,
    offset: int = 0,
    limit: int = 0,
    use_reference_as_task_hint: bool = False,
) -> dict:
    return prepare_ego4d_dataset(
        segments, output, offset, limit, use_reference_as_task_hint
    )


def prepare_lerobot_v3_command(
    dataset_root: Path,
    output: Path,
    dataset: str,
    camera_key: str,
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict:
    return prepare_lerobot_v3_video_manifest(
        dataset_root, output, dataset=dataset, camera_key=camera_key,
        offset=offset, limit=limit, min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        include_reference_caption=include_reference_caption,
    )


def prepare_lerobot_v2_command(
    dataset_root: Path,
    output: Path,
    dataset: str,
    camera_key: str,
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict:
    return prepare_lerobot_v2_video_manifest(
        dataset_root, output, dataset=dataset, camera_key=camera_key,
        offset=offset, limit=limit, min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        include_reference_caption=include_reference_caption,
    )


def prepare_droid_command(
    dataset_root: Path,
    output: Path,
    dataset: str = "droid-raw",
    camera: str = "wrist",
    offset: int = 0,
    limit: int = 0,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 3600.0,
    include_reference_caption: bool = True,
) -> dict:
    return prepare_droid_video_manifest(
        dataset_root, output, dataset=dataset, camera=camera,
        offset=offset, limit=limit, min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        include_reference_caption=include_reference_caption,
    )


def finalize_command(
    annotations: Path,
    validation: Path,
    output: Path,
    allow_flags: list[str] | None = None,
    reference_eval: Path | None = None,
) -> dict:
    return finalize_annotations(
        annotations, validation, output,
        allowed_flags=set(allow_flags) if allow_flags is not None else None,
        reference_eval_path=reference_eval,
    )


def evaluate_references_command(
    annotations: Path,
    dataset: Path,
    output: Path,
    low_overlap_threshold: float = 0.1,
) -> dict:
    return evaluate_references(
        annotations, dataset, output, low_overlap_threshold
    )
