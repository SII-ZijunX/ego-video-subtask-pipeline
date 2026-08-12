"""Publish accepted annotations separately from the human review queue."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .batch import read_jsonl, write_jsonl


DEFAULT_ALLOWED_FLAGS = frozenset({"same_interaction_merged", "action_synonym_mapped"})


def finalize_annotations(
    annotations_path: Path | str,
    validation_path: Path | str,
    output_dir: Path | str,
    allowed_flags: set[str] | None = None,
    reference_eval_path: Path | str | None = None,
) -> dict[str, Any]:
    annotations_path = Path(annotations_path)
    validation_path = Path(validation_path)
    output_dir = Path(output_dir)
    allowed_flags = set(DEFAULT_ALLOWED_FLAGS if allowed_flags is None else allowed_flags)
    validations = {row["episode_id"]: row for row in read_jsonl(validation_path)}
    reference_evaluations = (
        {row["episode_id"]: row for row in read_jsonl(Path(reference_eval_path))}
        if reference_eval_path else {}
    )
    accepted: list[dict] = []
    review: list[dict] = []
    reason_counts: Counter[str] = Counter()
    for annotation in read_jsonl(annotations_path):
        episode_id = annotation["episode_id"]
        validation = validations.get(episode_id)
        reasons: list[str] = []
        if validation is None:
            reasons.append("missing_validation")
        else:
            if not validation.get("valid"):
                reasons.extend(f"validation:{reason}" for reason in validation.get("retry_reasons", []))
                if not validation.get("retry_reasons"):
                    reasons.append("validation:invalid")
            combined_flags = set(validation.get("quality_flags", [])) | set(
                annotation.get("episode_quality_flags", [])
            )
            reasons.extend(f"flag:{flag}" for flag in sorted(combined_flags - allowed_flags))
        ineligible = [
            subtask.get("subtask_id") for subtask in annotation.get("subtasks", [])
            if not subtask.get("training_eligible", False)
        ]
        if ineligible:
            reasons.append("ineligible_subtasks:" + ",".join(map(str, ineligible)))
        reference_evaluation = reference_evaluations.get(episode_id)
        if reference_eval_path and reference_evaluation is None:
            reasons.append("missing_reference_evaluation")
        elif reference_evaluation:
            reasons.extend(
                f"reference:{flag}" for flag in reference_evaluation.get("flags", [])
                if flag not in allowed_flags
            )
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            reason_counts.update(reasons)
            review.append({
                "episode_id": episode_id,
                "review_reasons": reasons,
                "annotation": annotation,
                "reference_evaluation": reference_evaluation,
                "review_page": f"reports/{episode_id}/review.html",
            })
        else:
            accepted.append(annotation)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "accepted_annotations.jsonl", accepted)
    write_jsonl(output_dir / "review_queue.jsonl", review)
    markdown_rows = []
    for index, row in enumerate(review, 1):
        reference = str((row.get("reference_evaluation") or {}).get("reference_caption") or "")
        prediction = str(row["annotation"].get("video_level_instruction") or "")
        escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
        markdown_rows.append(
            f"| {index} | [{escape(row['episode_id'])}]({row['review_page']}) | "
            f"{escape(reference)} | {escape(prediction)} | {escape(', '.join(row['review_reasons']))} |"
        )
    (output_dir / "review_queue.md").write_text(
        "# Human Review Queue\n\n"
        f"Accepted automatically: `{len(accepted)}`; requires review: `{len(review)}`.\n\n"
        "| # | Episode | Official reference | Predicted task | Review reasons |\n"
        "|---:|---|---|---|---|\n"
        + "\n".join(markdown_rows)
        + "\n"
    )
    summary = {
        "annotations": len(accepted) + len(review),
        "accepted": len(accepted),
        "review": len(review),
        "allowed_flags": sorted(allowed_flags),
        "reference_eval": str(Path(reference_eval_path)) if reference_eval_path else None,
        "review_reason_counts": dict(reason_counts.most_common()),
    }
    (output_dir / "finalize_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary
