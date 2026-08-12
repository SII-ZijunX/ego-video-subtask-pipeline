"""Strict-enough JSON extraction with preserved raw responses."""

from __future__ import annotations

import json
import re
from typing import Any


class AnnotationParseError(ValueError):
    pass


def parse_model_json(response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise AnnotationParseError("response does not contain a JSON object")
        try:
            parsed = json.loads(text[start: end + 1])
        except json.JSONDecodeError as exc:
            raise AnnotationParseError(f"invalid JSON response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AnnotationParseError("top-level response must be a JSON object")
    return parsed
