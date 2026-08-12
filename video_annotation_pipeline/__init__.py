"""Auditable egocentric-video segmentation and language annotation tools.

The base package is deliberately importable without torch so validation,
reports, and the mock backend can run on CPU machines and in CI.
"""

from .schemas import EpisodeAnnotation, EpisodeMetadata, SubtaskAnnotation
from .vocabulary import ACTION_VOCABULARY

__all__ = [
    "ACTION_VOCABULARY",
    "EpisodeAnnotation",
    "EpisodeMetadata",
    "SubtaskAnnotation",
]

__version__ = "0.1.0"
