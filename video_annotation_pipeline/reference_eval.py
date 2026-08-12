"""Ego4D narration comparison signals for review, not automatic ground truth."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .batch import discover_episodes, read_jsonl, write_jsonl
from .schemas import EpisodeMetadata


STOP_WORDS = {
    "a", "an", "and", "the", "to", "from", "in", "on", "at", "with", "of", "it",
    "his", "her", "their", "both", "camera", "wearer", "person", "then", "into",
}
REFERENCE_ACTION_PATTERNS = {
    "move": r"\b(pick|picks|picked|take|takes|took|put|puts|place|places|move|moves|carry|carries|grab|grabs|hold|holds)\b",
    "fold": r"\b(fold|folds|folded)\b",
    "pour": r"\b(pour|pours|poured)\b",
    "unfold": r"\b(unfold|unfolds|unfolded)\b",
    "push": r"\b(push|pushes|pushed)\b",
    "wipe": r"\b(wipe|wipes|wiped|clean|cleans|cleaned)\b",
    "pull": r"\b(pull|pulls|pulled)\b",
    "stir": r"\b(stir|stirs|stirred|mix|mixes|mixed)\b",
    "rotate": r"\b(rotate|rotates|rotated|turn|turns|turned|twist|twists|twisted)\b",
    "cut": r"\b(cut|cuts|cutting)\b",
    "open": r"\b(open|opens|opened)\b",
    "press": r"\b(press|presses|pressed|tap|taps|tapped)\b",
    "close": r"\b(close|closes|closed)\b",
    "attach": r"\b(attach|attaches|attached|insert|inserts|inserted|connect|connects|connected)\b",
    "detach": r"\b(detach|detaches|detached|remove|removes|removed|disconnect|disconnects|disconnected)\b",
    "transit": r"\b(walk|walks|walked|reach|reaches|reached)\b",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def _reference_actions(text: str) -> set[str]:
    lower = text.lower()
    return {action for action, pattern in REFERENCE_ACTION_PATTERNS.items() if re.search(pattern, lower)}


def evaluate_references(
    annotations_path: Path | str,
    dataset_dir: Path | str,
    output_path: Path | str,
    low_overlap_threshold: float = 0.1,
) -> dict:
    metadata_by_id: dict[str, EpisodeMetadata] = {}
    for episode_dir in discover_episodes(dataset_dir):
        metadata = EpisodeMetadata.model_validate_json((episode_dir / "metadata.json").read_text())
        metadata_by_id[metadata.episode_id] = metadata
    output_rows = []
    flag_counts: Counter[str] = Counter()
    for annotation in read_jsonl(Path(annotations_path)):
        episode_id = annotation["episode_id"]
        metadata = metadata_by_id.get(episode_id)
        reference = str(getattr(metadata, "reference_caption", "") or "") if metadata else ""
        prediction = " ".join(
            [str(annotation.get("video_level_instruction") or "")]
            + [str(row.get("instruction") or "") for row in annotation.get("subtasks", [])]
        )
        reference_tokens = _tokens(reference)
        prediction_tokens = _tokens(prediction)
        union = reference_tokens | prediction_tokens
        overlap = len(reference_tokens & prediction_tokens) / len(union) if union else 0.0
        reference_actions = _reference_actions(reference)
        predicted_actions = {str(row.get("action")) for row in annotation.get("subtasks", [])}
        flags = []
        if reference and overlap < low_overlap_threshold:
            flags.append("low_reference_overlap")
        if reference_actions and reference_actions.isdisjoint(predicted_actions):
            flags.append("reference_action_mismatch")
        flag_counts.update(flags)
        output_rows.append({
            "episode_id": episode_id,
            "reference_caption": reference,
            "video_level_instruction": annotation.get("video_level_instruction"),
            "predicted_actions": sorted(predicted_actions),
            "reference_action_candidates": sorted(reference_actions),
            "lexical_jaccard": round(overlap, 4),
            "flags": flags,
        })
    output_path = Path(output_path)
    write_jsonl(output_path, output_rows)
    summary = {
        "episodes": len(output_rows),
        "with_reference": sum(bool(row["reference_caption"]) for row in output_rows),
        "flag_counts": dict(flag_counts.most_common()),
        "mean_lexical_jaccard": round(
            sum(row["lexical_jaccard"] for row in output_rows) / len(output_rows), 4
        ) if output_rows else 0.0,
        "note": "Reference overlap/action candidates are review signals, not automatic correctness labels.",
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary
