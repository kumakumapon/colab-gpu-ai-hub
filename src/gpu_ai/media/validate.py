"""Read media beneath a workspace and validate its content before transfer."""

import json
import math
import os
import stat
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import monotonic

from gpu_ai.backends.base import job_deadline
from gpu_ai.contracts import validate_relative_path

MEDIA_TYPES = {
    ".mp3": ("audio/mpeg", "mp3", "audio"),
    ".wav": ("audio/wav", "wav", "audio"),
    ".mp4": ("video/mp4", "mov", "video"),
    ".jpg": ("image/jpeg", "jpeg_pipe", "video"),
    ".jpeg": ("image/jpeg", "jpeg_pipe", "video"),
    ".png": ("image/png", "png_pipe", "video"),
    ".pdf": ("application/pdf", "", ""),
}
PRIVATE_DIRECTORIES = {"credentials", "secrets", "cache", "weights"}
PRIVATE_SUFFIXES = {".pem", ".key", ".safetensors", ".pt", ".pth", ".ckpt"}


class MediaError(ValueError):
    """An input failed local media validation or preprocessing."""


@dataclass(frozen=True)
class MediaInfo:
    mime_type: str
    duration_seconds: float | None = None
    has_audio: bool = False


def validate_transfer_path(path: str) -> None:
    validate_relative_path(path)
    parsed = PurePosixPath(path)
    if (
        parsed.as_posix() != path
        or path.startswith("gpu-ai-")
        or any(
            part.startswith(".") or part.lower() in PRIVATE_DIRECTORIES
            for part in parsed.parts
        )
        or parsed.suffix.lower() in PRIVATE_SUFFIXES
    ):
        raise ValueError("invalid or reserved transfer path")


def read_file(workspace: Path, path: str, max_bytes: int) -> bytes:
    validate_transfer_path(path)
    parts = PurePosixPath(path).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    with ExitStack() as stack:
        current = os.open(workspace, directory_flags)
        stack.callback(os.close, current)
        for parent in parts[:-1]:
            current = os.open(parent, directory_flags, dir_fd=current)
            stack.callback(os.close, current)
        fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current
        )
        stack.callback(os.close, fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise MediaError("expected a regular input file with one link")
        with os.fdopen(fd, "rb", closefd=False) as source:
            if metadata.st_size > max_bytes:
                raise MediaError("input size exceeds max_input_bytes")
            data = source.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise MediaError("input size exceeds max_input_bytes")
            return data


def run_media(command: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    deadline = job_deadline.get()
    if deadline is not None:
        timeout = min(timeout, deadline - monotonic())
    if timeout <= 0:
        raise MediaError("media processing timed out")
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise MediaError("ffmpeg and ffprobe must be installed") from None
    except subprocess.TimeoutExpired:
        raise MediaError("media processing timed out") from None
    except subprocess.CalledProcessError:
        raise MediaError("media is invalid or could not be decoded") from None


def probe_media(path: Path, demuxer: str, *, timeout: float = 30) -> dict:
    output = run_media(
        [
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-format_whitelist",
            demuxer,
            "-show_entries",
            "format=format_name,duration:stream=codec_type,width,height:stream_disposition=attached_pic",
            "-of",
            "json",
            str(path.resolve()),
        ],
        timeout=timeout,
    )
    try:
        return json.loads(output.stdout)
    except json.JSONDecodeError:
        raise MediaError("invalid ffprobe metadata") from None


def _matches_header(suffix: str, data: bytes) -> bool:
    match suffix:
        case ".mp3":
            return data.startswith(b"ID3") or (
                len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
            )
        case ".wav":
            return data.startswith((b"RIFF", b"RF64")) and data[8:12] == b"WAVE"
        case ".mp4":
            return data[4:8] == b"ftyp" and data[8:12] != b"qt  "
        case ".jpg" | ".jpeg":
            return data.startswith(b"\xff\xd8\xff")
        case ".png":
            return data.startswith(b"\x89PNG\r\n\x1a\n")
        case ".pdf":
            return data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    return False


def inspect_media(path: str, data: bytes, *, max_duration_seconds: float) -> MediaInfo:
    suffix = PurePosixPath(path).suffix.lower()
    expected = MEDIA_TYPES.get(suffix)
    if expected is None or not _matches_header(suffix, data):
        raise MediaError(
            "unsupported file type or content does not match the extension"
        )
    mime_type, demuxer, required_stream = expected
    if not demuxer:
        return MediaInfo(mime_type)

    with TemporaryDirectory(prefix="gpu-ai-media-") as temporary:
        local = Path(temporary) / f"input{suffix}"
        local.write_bytes(data)
        metadata = probe_media(local, demuxer)
    streams = metadata.get("streams", [])
    if not any(
        stream.get("codec_type") == required_stream
        and not stream.get("disposition", {}).get("attached_pic")
        for stream in streams
    ):
        raise MediaError("required media stream is missing")
    if any(
        stream.get("width", 0) * stream.get("height", 0) > 32_000_000
        for stream in streams
    ):
        raise MediaError("video frame or image dimensions exceed the supported limit")

    duration = None
    if suffix in {".mp3", ".wav", ".mp4"}:
        try:
            duration = float(metadata.get("format", {})["duration"])
        except (KeyError, TypeError, ValueError):
            raise MediaError("media duration is unavailable") from None
        if not math.isfinite(duration) or not 0 < duration <= max_duration_seconds:
            raise MediaError("media duration exceeds the allowed range")
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return MediaInfo(mime_type, duration, has_audio)


def load_media(
    workspace: Path, paths: list[str], *, max_bytes: int, max_duration_seconds: float
) -> tuple[dict[str, bytes], dict[str, MediaInfo]]:
    if len(set(paths)) != len(paths):
        raise MediaError("duplicate input paths")
    files, information = {}, {}
    remaining = max_bytes
    for path in paths:
        try:
            data = read_file(workspace, path, remaining)
        except OSError:
            raise MediaError(
                "could not open a regular file inside the workspace"
            ) from None
        information[path] = inspect_media(
            path, data, max_duration_seconds=max_duration_seconds
        )
        files[path] = data
        remaining -= len(data)
    return files, information
