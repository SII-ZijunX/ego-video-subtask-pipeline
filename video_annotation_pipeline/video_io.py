"""Episode metadata loading and ffprobe-based input validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .schemas import EpisodeMetadata, VideoInfo


class EpisodeInputError(ValueError):
    def __init__(self, message: str, flags: list[str] | None = None):
        super().__init__(message)
        self.flags = flags or []


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def probe_video(path: Path, camera_name: str, camera_role: str, offset: float) -> VideoInfo:
    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        str(path),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise EpisodeInputError(
            f"ffprobe failed for {path}: {completed.stderr.strip()}", ["video_decode_failure"]
        )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise EpisodeInputError(f"no readable video stream in {path}", ["video_decode_failure"]) from exc

    fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    frame_count_raw = stream.get("nb_frames")
    frame_count = int(frame_count_raw) if str(frame_count_raw or "").isdigit() else int(round(duration * fps))
    try:
        return VideoInfo(
            camera_name=camera_name,
            camera_role=camera_role,
            path=str(path.resolve()),
            codec_name=stream.get("codec_name"),
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            time_offset_sec=offset,
        )
    except Exception as exc:
        raise EpisodeInputError(f"invalid video metadata for {path}: {exc}", ["video_decode_failure"]) from exc


def load_and_validate_episode(
    episode_dir: Path | str, config: PipelineConfig
) -> tuple[EpisodeMetadata, list[VideoInfo], float]:
    episode_dir = Path(episode_dir)
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        raise EpisodeInputError(f"missing {metadata_path}", ["missing_metadata"])
    try:
        metadata = EpisodeMetadata.model_validate_json(metadata_path.read_text())
    except Exception as exc:
        raise EpisodeInputError(f"invalid episode metadata: {exc}", ["invalid_metadata"]) from exc

    infos: list[VideoInfo] = []
    missing: list[str] = []
    for camera in metadata.cameras:
        path = metadata.resolved_camera_path(episode_dir, camera)
        if not path.is_file():
            missing.append(str(path))
            continue
        infos.append(probe_video(path, camera.name, camera.role, camera.time_offset_sec))
    if missing:
        raise EpisodeInputError(f"missing video files: {missing}", ["missing_video"])
    if not infos:
        raise EpisodeInputError("episode has no readable videos", ["missing_video"])

    usable_durations = [info.duration_sec - max(0.0, info.time_offset_sec) for info in infos]
    duration = min(usable_durations)
    if duration < config.validation.minimum_video_duration_sec:
        raise EpisodeInputError(
            f"episode duration {duration:.3f}s is below minimum", ["video_too_short"]
        )
    if len(infos) > 1:
        duration_spread = max(usable_durations) - min(usable_durations)
        if duration_spread > config.validation.max_multiview_duration_diff_sec:
            raise EpisodeInputError(
                f"multi-view duration difference {duration_spread:.3f}s exceeds threshold",
                ["multi_view_duration_mismatch"],
            )
    return metadata, infos, round(duration, 3)
