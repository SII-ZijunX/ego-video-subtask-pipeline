"""Timestamp-aware single/multi-camera frame sampling."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .schemas import SampleManifestEntry, VideoInfo


class FrameSamplingError(RuntimeError):
    pass


def sampling_timestamps(duration_sec: float, fps: float, max_frames: int) -> list[float]:
    desired = max(1, int(math.ceil(duration_sec * fps)))
    count = min(desired, max_frames)
    if count == 1:
        return [round(duration_sec / 2.0, 3)]
    # Sample frame centers and avoid requesting exactly at EOF.
    step = duration_sec / count
    return [round(min(duration_sec - 1e-3, (index + 0.5) * step), 3) for index in range(count)]


def _read_frame(capture: cv2.VideoCapture, time_sec: float, path: str) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise FrameSamplingError(f"cannot decode {path} at {time_sec:.3f}s")
    return frame


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = long_side / max(height, width)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))


def _labeled_tile(frame: np.ndarray, label: str, tile_size: int) -> np.ndarray:
    frame = _resize_long_side(frame, tile_size)
    height, width = frame.shape[:2]
    scale = min(tile_size / max(1, width), tile_size / max(1, height))
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    y = (tile_size - resized.shape[0]) // 2
    x = (tile_size - resized.shape[1]) // 2
    canvas[y: y + resized.shape[0], x: x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (tile_size, 34), (0, 0, 0), -1)
    cv2.putText(canvas, label.upper(), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return canvas


def _grid(frames: list[np.ndarray], labels: list[str], long_side: int) -> np.ndarray:
    count = len(frames)
    columns = 1 if count == 1 else 2
    rows = int(math.ceil(count / columns))
    tile_size = max(128, long_side // columns if count > 1 else long_side)
    tiles = [_labeled_tile(frame, label, tile_size) for frame, label in zip(frames, labels)]
    while len(tiles) < rows * columns:
        tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
    return np.vstack([
        np.hstack(tiles[row * columns: (row + 1) * columns]) for row in range(rows)
    ])


def sample_episode(
    infos: list[VideoInfo], duration_sec: float, output_dir: Path, config: PipelineConfig
) -> list[SampleManifestEntry]:
    frame_dir = output_dir / "sampled_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    timestamps = sampling_timestamps(
        duration_sec, config.sampling.fps, config.sampling.max_frames
    )
    captures = {info.camera_name: cv2.VideoCapture(info.path) for info in infos}
    if not all(capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise FrameSamplingError("one or more camera videos cannot be opened by OpenCV")

    manifest: list[SampleManifestEntry] = []
    try:
        for sample_index, timestamp in enumerate(timestamps):
            frames: list[np.ndarray] = []
            labels: list[str] = []
            source_times: dict[str, float] = {}
            for info in infos:
                source_time = max(0.0, timestamp + info.time_offset_sec)
                frames.append(_read_frame(captures[info.camera_name], source_time, info.path))
                labels.append(f"{info.camera_role}: {info.camera_name}")
                source_times[info.camera_name] = round(source_time, 3)
            combined = _grid(frames, labels, config.sampling.resize_long_side)
            cv2.putText(
                combined,
                f"t={timestamp:.3f}s",
                (10, combined.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            image_path = frame_dir / f"frame_{sample_index:04d}_t{timestamp:09.3f}.jpg"
            if not cv2.imwrite(str(image_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise FrameSamplingError(f"failed to write {image_path}")
            manifest.append(SampleManifestEntry(
                sample_index=sample_index,
                original_time_sec=timestamp,
                image_path=str(image_path.resolve()),
                camera_roles=[info.camera_role for info in infos],
                source_times_sec=source_times,
            ))
    finally:
        for capture in captures.values():
            capture.release()
    return manifest
