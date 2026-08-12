"""Typer command line interface required by annotation.md."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .commands import (
    annotate_batch_command,
    annotate_command,
    evaluate_references_command,
    finalize_command,
    prepare_ego4d_command,
    prepare_lerobot_v2_command,
    prepare_lerobot_v3_command,
    stats_command,
    validate_command,
    visualize_command,
)


app = typer.Typer(no_args_is_help=True, help="Egocentric-video subtask annotation pipeline")


@app.command("annotate")
def annotate(
    input_dir: Path = typer.Option(..., "--input", exists=True, file_okay=False),
    output_dir: Path = typer.Option(..., "--output"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    typer.echo(json.dumps(annotate_command(input_dir, output_dir, config), ensure_ascii=False, indent=2))


@app.command("annotate-batch")
def annotate_batch_cli(
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    resume: bool = typer.Option(False, "--resume"),
    skip_existing: bool = typer.Option(False, "--skip-existing"),
    retry_failed: bool = typer.Option(False, "--retry-failed"),
) -> None:
    typer.echo(json.dumps(annotate_batch_command(
        dataset, output, config, resume, skip_existing, retry_failed
    ), ensure_ascii=False, indent=2))


@app.command("validate")
def validate_cli(
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
) -> None:
    result = validate_command(annotations, output, config)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    if result["invalid"]:
        raise typer.Exit(1)


@app.command("visualize")
def visualize_cli(
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
) -> None:
    typer.echo(json.dumps(visualize_command(dataset, annotations, output), ensure_ascii=False, indent=2))


@app.command("stats")
def stats_cli(
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
) -> None:
    typer.echo(json.dumps(stats_command(annotations, output), ensure_ascii=False, indent=2))


@app.command("prepare-ego4d")
def prepare_ego4d_cli(
    segments: Path = typer.Option(..., "--segments", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    offset: int = typer.Option(0, "--offset", min=0),
    limit: int = typer.Option(0, "--limit", min=0),
    use_reference_as_task_hint: bool = typer.Option(False, "--use-reference-as-task-hint"),
) -> None:
    typer.echo(json.dumps(prepare_ego4d_command(
        segments, output, offset, limit, use_reference_as_task_hint
    ), ensure_ascii=False, indent=2))


@app.command("prepare-lerobot-v3")
def prepare_lerobot_v3_cli(
    dataset_root: Path = typer.Option(..., "--dataset-root", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    dataset: str = typer.Option(..., "--dataset"),
    camera_key: str = typer.Option(..., "--camera-key"),
    offset: int = typer.Option(0, "--offset", min=0),
    limit: int = typer.Option(0, "--limit", min=0),
    min_duration_sec: float = typer.Option(3.0, "--min-duration-sec", min=0),
    max_duration_sec: float = typer.Option(3600.0, "--max-duration-sec", min=0),
    include_reference_caption: bool = typer.Option(
        True, "--include-reference-caption/--no-reference-caption"
    ),
) -> None:
    typer.echo(json.dumps(prepare_lerobot_v3_command(
        dataset_root, output, dataset, camera_key, offset, limit,
        min_duration_sec, max_duration_sec, include_reference_caption,
    ), ensure_ascii=False, indent=2))


@app.command("prepare-lerobot-v2")
def prepare_lerobot_v2_cli(
    dataset_root: Path = typer.Option(..., "--dataset-root", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    dataset: str = typer.Option(..., "--dataset"),
    camera_key: str = typer.Option(..., "--camera-key"),
    offset: int = typer.Option(0, "--offset", min=0),
    limit: int = typer.Option(0, "--limit", min=0),
    min_duration_sec: float = typer.Option(3.0, "--min-duration-sec", min=0),
    max_duration_sec: float = typer.Option(3600.0, "--max-duration-sec", min=0),
    include_reference_caption: bool = typer.Option(
        True, "--include-reference-caption/--no-reference-caption"
    ),
) -> None:
    typer.echo(json.dumps(prepare_lerobot_v2_command(
        dataset_root, output, dataset, camera_key, offset, limit,
        min_duration_sec, max_duration_sec, include_reference_caption,
    ), ensure_ascii=False, indent=2))


@app.command("finalize")
def finalize_cli(
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    validation: Path = typer.Option(..., "--validation", exists=True, dir_okay=False),
    output: Path = typer.Option(..., "--output"),
    allow_flag: list[str] | None = typer.Option(None, "--allow-flag"),
    reference_eval: Path | None = typer.Option(None, "--reference-eval", exists=True, dir_okay=False),
) -> None:
    typer.echo(json.dumps(finalize_command(
        annotations, validation, output, allow_flag, reference_eval
    ), ensure_ascii=False, indent=2))


@app.command("evaluate-references")
def evaluate_references_cli(
    annotations: Path = typer.Option(..., "--annotations", exists=True, dir_okay=False),
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    low_overlap_threshold: float = typer.Option(0.1, "--low-overlap-threshold", min=0, max=1),
) -> None:
    typer.echo(json.dumps(evaluate_references_command(
        annotations, dataset, output, low_overlap_threshold
    ), ensure_ascii=False, indent=2))


def run() -> None:
    app()
