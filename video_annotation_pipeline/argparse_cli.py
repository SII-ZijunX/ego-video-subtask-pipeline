"""Small fallback for the existing Qwen env, which currently lacks Typer.

The supported project CLI is Typer.  This fallback keeps the already-provisioned
offline GPU environment runnable without mutating it before the real-model
canary; installing the ``annotation`` optional dependencies enables Typer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import (
    annotate_batch_command,
    annotate_command,
    evaluate_references_command,
    finalize_command,
    prepare_droid_command,
    prepare_ego4d_command,
    prepare_lerobot_v2_command,
    prepare_lerobot_v3_command,
    stats_command,
    validate_command,
    visualize_command,
)


def run() -> None:
    parser = argparse.ArgumentParser(description="Egocentric-video subtask annotation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    annotate_parser = subparsers.add_parser("annotate")
    annotate_parser.add_argument("--input", required=True, type=Path)
    annotate_parser.add_argument("--output", required=True, type=Path)
    annotate_parser.add_argument("--config", required=True, type=Path)
    batch_parser = subparsers.add_parser("annotate-batch")
    batch_parser.add_argument("--dataset", required=True, type=Path)
    batch_parser.add_argument("--output", required=True, type=Path)
    batch_parser.add_argument("--config", required=True, type=Path)
    batch_parser.add_argument("--resume", action="store_true")
    batch_parser.add_argument("--skip-existing", action="store_true")
    batch_parser.add_argument("--retry-failed", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--annotations", required=True, type=Path)
    validate_parser.add_argument("--output", required=True, type=Path)
    validate_parser.add_argument("--config", required=True, type=Path)
    visualize_parser = subparsers.add_parser("visualize")
    visualize_parser.add_argument("--dataset", required=True, type=Path)
    visualize_parser.add_argument("--annotations", required=True, type=Path)
    visualize_parser.add_argument("--output", required=True, type=Path)
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("--annotations", required=True, type=Path)
    stats_parser.add_argument("--output", required=True, type=Path)
    prepare_parser = subparsers.add_parser("prepare-ego4d")
    prepare_parser.add_argument("--segments", required=True, type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    prepare_parser.add_argument("--offset", type=int, default=0)
    prepare_parser.add_argument("--limit", type=int, default=0)
    prepare_parser.add_argument("--use-reference-as-task-hint", action="store_true")
    lerobot_v2_parser = subparsers.add_parser("prepare-lerobot-v2")
    lerobot_v2_parser.add_argument("--dataset-root", required=True, type=Path)
    lerobot_v2_parser.add_argument("--output", required=True, type=Path)
    lerobot_v2_parser.add_argument("--dataset", required=True)
    lerobot_v2_parser.add_argument("--camera-key", required=True)
    lerobot_v2_parser.add_argument("--offset", type=int, default=0)
    lerobot_v2_parser.add_argument("--limit", type=int, default=0)
    lerobot_v2_parser.add_argument("--min-duration-sec", type=float, default=3.0)
    lerobot_v2_parser.add_argument("--max-duration-sec", type=float, default=3600.0)
    lerobot_v2_parser.add_argument("--no-reference-caption", action="store_true")
    lerobot_parser = subparsers.add_parser("prepare-lerobot-v3")
    lerobot_parser.add_argument("--dataset-root", required=True, type=Path)
    lerobot_parser.add_argument("--output", required=True, type=Path)
    lerobot_parser.add_argument("--dataset", required=True)
    lerobot_parser.add_argument("--camera-key", required=True)
    lerobot_parser.add_argument("--offset", type=int, default=0)
    lerobot_parser.add_argument("--limit", type=int, default=0)
    lerobot_parser.add_argument("--min-duration-sec", type=float, default=3.0)
    lerobot_parser.add_argument("--max-duration-sec", type=float, default=3600.0)
    lerobot_parser.add_argument("--no-reference-caption", action="store_true")
    droid_parser = subparsers.add_parser("prepare-droid")
    droid_parser.add_argument("--dataset-root", required=True, type=Path)
    droid_parser.add_argument("--output", required=True, type=Path)
    droid_parser.add_argument("--dataset", default="droid-raw")
    droid_parser.add_argument("--camera", choices=["wrist", "ext1", "ext2"], default="wrist")
    droid_parser.add_argument("--offset", type=int, default=0)
    droid_parser.add_argument("--limit", type=int, default=0)
    droid_parser.add_argument("--min-duration-sec", type=float, default=3.0)
    droid_parser.add_argument("--max-duration-sec", type=float, default=3600.0)
    droid_parser.add_argument("--no-reference-caption", action="store_true")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--annotations", required=True, type=Path)
    finalize_parser.add_argument("--validation", required=True, type=Path)
    finalize_parser.add_argument("--output", required=True, type=Path)
    finalize_parser.add_argument("--allow-flag", action="append", default=None)
    finalize_parser.add_argument("--reference-eval", type=Path, default=None)
    references_parser = subparsers.add_parser("evaluate-references")
    references_parser.add_argument("--annotations", required=True, type=Path)
    references_parser.add_argument("--dataset", required=True, type=Path)
    references_parser.add_argument("--output", required=True, type=Path)
    references_parser.add_argument("--low-overlap-threshold", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "annotate":
        result = annotate_command(args.input, args.output, args.config)
    elif args.command == "annotate-batch":
        result = annotate_batch_command(
            args.dataset, args.output, args.config, args.resume, args.skip_existing, args.retry_failed
        )
    elif args.command == "validate":
        result = validate_command(args.annotations, args.output, args.config)
    elif args.command == "visualize":
        result = visualize_command(args.dataset, args.annotations, args.output)
    elif args.command == "stats":
        result = stats_command(args.annotations, args.output)
    elif args.command == "prepare-ego4d":
        result = prepare_ego4d_command(
            args.segments, args.output, args.offset, args.limit, args.use_reference_as_task_hint
        )
    elif args.command == "prepare-lerobot-v2":
        result = prepare_lerobot_v2_command(
            args.dataset_root, args.output, args.dataset, args.camera_key,
            args.offset, args.limit, args.min_duration_sec, args.max_duration_sec,
            not args.no_reference_caption,
        )
    elif args.command == "prepare-lerobot-v3":
        result = prepare_lerobot_v3_command(
            args.dataset_root, args.output, args.dataset, args.camera_key,
            args.offset, args.limit, args.min_duration_sec, args.max_duration_sec,
            not args.no_reference_caption,
        )
    elif args.command == "prepare-droid":
        result = prepare_droid_command(
            args.dataset_root, args.output, args.dataset, args.camera,
            args.offset, args.limit, args.min_duration_sec, args.max_duration_sec,
            not args.no_reference_caption,
        )
    elif args.command == "finalize":
        result = finalize_command(
            args.annotations, args.validation, args.output, args.allow_flag, args.reference_eval
        )
    else:
        result = evaluate_references_command(
            args.annotations, args.dataset, args.output, args.low_overlap_threshold
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "validate" and result["invalid"]:
        raise SystemExit(1)
