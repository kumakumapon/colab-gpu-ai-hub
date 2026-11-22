"""Extract sampled frames with source timestamps and 16 kHz mono WAV."""

import re
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_ai.config import Settings
from gpu_ai.media.validate import MediaError, inspect_media, read_file, run_media


@dataclass(frozen=True)
class VideoFrame:
    timestamp_seconds: float
    png: bytes


@dataclass(frozen=True)
class VideoAssets:
    duration_seconds: float
    audio_wav: bytes | None
    frames: tuple[VideoFrame, ...]


def preprocess_video(
    path: str,
    settings: Settings,
    *,
    frame_interval_seconds: float = 1.0,
    max_frames: int = 120,
) -> VideoAssets:
    if not isfinite(frame_interval_seconds) or frame_interval_seconds <= 0:
        raise ValueError("frame interval must be positive and finite")
    if not 1 <= max_frames <= 1_000:
        raise ValueError("frame limit must be between 1 and 1000")
    if Path(path).suffix.lower() != ".mp4":
        raise MediaError("video preprocessing requires MP4")
    try:
        data = read_file(settings.workspace, path, settings.max_input_bytes)
    except OSError:
        raise MediaError("could not open a regular file inside the workspace") from None
    info = inspect_media(
        path, data, max_duration_seconds=settings.max_media_duration_seconds
    )
    if info.duration_seconds is None:
        raise MediaError("video duration is unavailable")

    with TemporaryDirectory(prefix="gpu-ai-video-") as temporary:
        directory = Path(temporary)
        source = directory / "input.mp4"
        source.write_bytes(data)
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-xerror",
            "-f",
            "mov",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
        ]
        frames = _extract_frames(command, directory, frame_interval_seconds, max_frames)
        audio = directory / "audio.wav"
        if info.has_audio:
            run_media(
                [
                    *command,
                    "-loglevel",
                    "error",
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ],
                timeout=settings.max_runtime_seconds,
            )
        paths = [frame for _, frame in frames]
        if info.has_audio:
            paths.append(audio)
        if sum(output.stat().st_size for output in paths) > settings.max_artifact_bytes:
            raise MediaError("preprocessed media exceeds the configured output limit")
        return VideoAssets(
            duration_seconds=info.duration_seconds,
            audio_wav=audio.read_bytes() if info.has_audio else None,
            frames=tuple(
                VideoFrame(timestamp, frame.read_bytes()) for timestamp, frame in frames
            ),
        )


def _extract_frames(
    command: list[str],
    directory: Path,
    interval: float,
    max_frames: int,
) -> list[tuple[float, Path]]:
    # Selection uses actual frame times. VFR inputs retain their original timing.
    filters = (
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.17g}),"
        "showinfo,scale=w='min(iw,1280)':h='min(ih,720)':force_original_aspect_ratio=decrease"
    )
    output = run_media(
        [
            *command,
            "-loglevel",
            "info",
            "-filter_threads",
            "1",
            "-map",
            "0:V:0",
            "-vf",
            filters,
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(max_frames),
            "-threads",
            "1",
            str(directory / "frame-%06d.png"),
        ],
        timeout=120,
    )
    clock = re.search(r"showinfo.*config in time_base: (\d+)/(\d+)", output.stderr)
    pts = re.findall(r"showinfo[^\n]+?\bn:\s*\d+\s+pts:\s*(-?\d+)", output.stderr)
    files = sorted(directory.glob("frame-*.png"))
    if clock is None or len(pts) < len(files) or not files:
        raise MediaError("could not determine the extracted frame timestamps")
    time_base = Fraction(int(clock[1]), int(clock[2]))
    return [(float(int(value) * time_base), path) for value, path in zip(pts, files)]
