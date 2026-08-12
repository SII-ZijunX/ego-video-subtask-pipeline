"""End-to-end episode annotation orchestration and reproducibility artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .backends import VideoAnnotatorBackend, create_backend
from .config import PipelineConfig, write_config_snapshot
from .parser import AnnotationParseError, parse_model_json
from .prompts import SYSTEM_PROMPT, build_user_prompt, response_json_schema
from .repair import normalize_and_repair
from .sampling import FrameSamplingError, sample_episode
from .schemas import AnnotationMetadata, EpisodeAnnotation, ValidationReport
from .validator import annotation_is_valid, validate_annotation
from .video_io import EpisodeInputError, load_and_validate_episode


class AnnotationRunError(RuntimeError):
    def __init__(self, message: str, flags: list[str] | None = None):
        super().__init__(message)
        self.flags = flags or []


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _git_commit(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _archive_previous_artifacts(output_dir: Path) -> None:
    existing = [path for path in output_dir.iterdir() if path.name != "history"]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = output_dir / "history" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))


def annotate_episode(
    episode_dir: Path | str,
    output_dir: Path | str,
    config: PipelineConfig,
    backend: VideoAnnotatorBackend | None = None,
    repo_root: Path | None = None,
) -> EpisodeAnnotation:
    """Annotate one episode and persist every prompt/response/validation attempt."""
    started = time.time()
    episode_dir = Path(episode_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _archive_previous_artifacts(output_dir)
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    write_config_snapshot(config, output_dir / "config.snapshot.yaml")
    _write_json(output_dir / "response_schema.json", response_json_schema())

    try:
        metadata, video_infos, duration_sec = load_and_validate_episode(episode_dir, config)
        samples = sample_episode(video_infos, duration_sec, output_dir, config)
    except (EpisodeInputError, FrameSamplingError) as exc:
        flags = getattr(exc, "flags", ["frame_sampling_failure"])
        _write_json(output_dir / "error.json", {"error": str(exc), "quality_flags": flags})
        raise AnnotationRunError(str(exc), flags) from exc

    _write_json(output_dir / "input_metadata.json", metadata.model_dump(mode="json"))
    _write_json(output_dir / "video_info.json", [info.model_dump(mode="json") for info in video_infos])
    _write_json(output_dir / "sample_manifest.json", [row.model_dump(mode="json") for row in samples])
    backend = backend or create_backend(config.backend)
    timestamps = [sample.original_time_sec for sample in samples]
    camera_roles = [camera.role for camera in metadata.cameras]

    previous_response: str | None = None
    retry_errors: list[str] = []
    final_normalized: dict[str, Any] | None = None
    final_flags: list[str] = []
    final_retry_reasons: list[str] = []
    final_coverage = 0.0
    final_backend_response = None
    repair_flags: list[str] = []
    final_attempt = 0

    for attempt in range(config.backend.max_retries + 1):
        user_prompt = build_user_prompt(
            metadata, duration_sec, samples, retry_errors or None, previous_response,
            simplify=attempt >= 2,
        )
        (output_dir / f"prompt_attempt_{attempt}.txt").write_text(
            "SYSTEM\n" + SYSTEM_PROMPT + "\n\nUSER\n" + user_prompt
        )
        attempt_started = time.time()
        try:
            response = backend.annotate(
                frames=samples,
                timestamps=timestamps,
                camera_roles=camera_roles,
                metadata=metadata,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            _write_json(output_dir / f"backend_error_attempt_{attempt}.json", {
                "error": repr(exc), "runtime_sec": round(time.time() - attempt_started, 3)
            })
            if attempt < config.backend.max_retries:
                retry_errors = ["backend_failure"]
                previous_response = None
                continue
            raise AnnotationRunError(f"backend failed after retries: {exc}", ["backend_failure"]) from exc

        final_backend_response = response
        final_attempt = attempt
        previous_response = response.text
        (output_dir / f"raw_response_attempt_{attempt}.txt").write_text(response.text)
        _write_json(output_dir / f"backend_metadata_attempt_{attempt}.json", {
            "model": response.model,
            "model_path": response.model_path,
            "usage": response.usage,
            "runtime_sec": round(time.time() - attempt_started, 3),
        })
        try:
            parsed = parse_model_json(response.text)
        except AnnotationParseError as exc:
            _write_json(output_dir / f"parse_error_attempt_{attempt}.json", {"error": str(exc)})
            retry_errors = ["vlm_parse_failure"]
            if attempt < config.backend.max_retries:
                continue
            raise AnnotationRunError(str(exc), retry_errors) from exc

        _write_json(output_dir / f"parsed_response_attempt_{attempt}.json", parsed)
        normalized, repair_flags = normalize_and_repair(
            parsed, duration_sec, video_infos[0].fps, config
        )
        flags, retry_reasons, coverage = validate_annotation(normalized, duration_sec, config)
        _write_json(output_dir / f"validation_attempt_{attempt}.json", {
            "quality_flags": flags,
            "retry_reasons": retry_reasons,
            "coverage_ratio": coverage,
            "repair_flags": repair_flags,
            "normalized_annotation": normalized,
        })
        final_normalized = normalized
        final_flags = list(dict.fromkeys(flags + repair_flags))
        final_retry_reasons = retry_reasons
        final_coverage = coverage
        if retry_reasons and attempt < config.backend.max_retries:
            retry_errors = retry_reasons
            continue
        break

    if final_normalized is None or final_backend_response is None:
        raise AnnotationRunError("no annotation was produced", ["empty_annotation"])

    runtime_sec = round(time.time() - started, 3)
    try:
        annotation = EpisodeAnnotation(
            episode_id=metadata.episode_id,
            source=metadata.source,
            duration_sec=duration_sec,
            video_level_instruction=final_normalized["video_level_instruction"],
            subtasks=final_normalized["subtasks"],
            annotation_metadata=AnnotationMetadata(
                model=final_backend_response.model,
                model_path=final_backend_response.model_path,
                prompt_version=config.prompt_version,
                sampling_fps=config.sampling.fps,
                retry_count=final_attempt,
                git_commit=_git_commit(repo_root),
                runtime_sec=runtime_sec,
            ),
            episode_quality_flags=final_flags,
        )
    except ValidationError as exc:
        _write_json(output_dir / "schema_error.json", {"error": str(exc), "normalized": final_normalized})
        raise AnnotationRunError(f"normalized annotation failed schema: {exc}", ["schema_failure"]) from exc

    _write_json(output_dir / "annotation.json", annotation.model_dump(mode="json"))
    report = ValidationReport(
        episode_id=metadata.episode_id,
        valid=annotation_is_valid(final_flags, final_retry_reasons),
        quality_flags=final_flags,
        retry_reasons=final_retry_reasons,
        coverage_ratio=final_coverage,
        normalized_annotation=annotation.model_dump(mode="json"),
    )
    _write_json(output_dir / "validation.json", report.model_dump(mode="json"))
    _write_json(output_dir / "run_summary.json", {
        "episode_id": metadata.episode_id,
        "backend": config.backend.type,
        "model": final_backend_response.model,
        "duration_sec": duration_sec,
        "sampled_frames": len(samples),
        "subtasks": len(annotation.subtasks),
        "runtime_sec": runtime_sec,
        "quality_flags": final_flags,
    })
    if config.output.generate_html:
        from .visualization import generate_episode_report
        generate_episode_report(
            annotation, metadata, episode_dir, output_dir / "review.html", previous_response or ""
        )
    return annotation
