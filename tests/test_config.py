import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_annotation_pipeline.config import load_config


def test_public_config_has_no_private_model_default() -> None:
    config = load_config(ROOT / "configs" / "mock.yaml")
    assert config.backend.type == "mock"
    assert config.backend.model_path is None


def test_default_minimum_is_three_seconds() -> None:
    config = load_config(ROOT / "configs" / "mock.yaml")
    assert config.segmentation.min_subtask_duration_sec == 3.0
    assert config.validation.minimum_video_duration_sec == 3.0


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"[ok] {name}")
