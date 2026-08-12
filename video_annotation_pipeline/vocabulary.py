"""Closed atomic-action vocabulary and deterministic synonym mapping."""

from __future__ import annotations

from typing import Optional


ACTION_VOCABULARY = [
    "move",
    "fold",
    "pour",
    "unfold",
    "push",
    "wipe",
    "pull",
    "stir",
    "rotate",
    "cut",
    "open",
    "press",
    "close",
    "attach",
    "detach",
    "transit",
    "idle",
    "other",
]
ACTION_SET = frozenset(ACTION_VOCABULARY)

ACTION_ALIASES = {
    "pick": "move",
    "pick up": "move",
    "pick_up": "move",
    "pickup": "move",
    "place": "move",
    "carry": "move",
    "relocate": "move",
    "grab": "move",
    "grasp": "move",
    "release": "move",
    "insert": "attach",
    "install": "attach",
    "connect": "attach",
    "remove": "detach",
    "disconnect": "detach",
    "take off": "detach",
    "take_off": "detach",
    "turn": "rotate",
    "twist": "rotate",
    "walk": "transit",
    "reach": "transit",
    "reposition gripper": "transit",
    "reposition_gripper": "transit",
}

OBJECT_OPTIONAL_ACTIONS = frozenset({"transit", "idle", "other"})


def canonicalize_token(value: object) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def normalize_action(value: object) -> tuple[str, list[str]]:
    """Return an allowed action and flags describing any lossy normalization."""
    token = canonicalize_token(value)
    if token in ACTION_SET:
        return token, []
    if token in ACTION_ALIASES:
        return ACTION_ALIASES[token], ["action_synonym_mapped"]
    underscored = token.replace(" ", "_")
    if underscored in ACTION_ALIASES:
        return ACTION_ALIASES[underscored], ["action_synonym_mapped"]
    return "other", ["out_of_vocabulary_action"]


def object_is_required(action: str) -> bool:
    return action not in OBJECT_OPTIONAL_ACTIONS


def normalize_object(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    return text
