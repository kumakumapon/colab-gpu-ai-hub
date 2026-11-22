"""Japanese and auto-language transcription using faster-whisper small."""

import ctypes
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ._runtime import (
    WorkerError,
    check_parameters,
    ensure_packages,
    integer,
    request_fields,
    write_json,
)

MODEL_ID = "Systran/faster-whisper-small"
# faster-whisper 1.2.0 passes metadata_errors to av.open. PyAV 18 removed it.
PINS = {"faster-whisper": "1.2.0", "ctranslate2": "4.6.0", "av": "17.1.0"}
ARTIFACTS = ("transcript.json", "transcript.srt")


@dataclass(frozen=True)
class Options:
    path: str
    language: str
    vad_filter: bool
    beam_size: int


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(parameters, {"language", "vad_filter", "beam_size"})
    if len(inputs) != 1 or Path(inputs[0]).suffix.lower() not in {".wav", ".mp3"}:
        raise WorkerError("invalid_worker_input")
    language = parameters.get("language", "ja")
    vad_filter = parameters.get("vad_filter", True)
    if language not in ("ja", "auto") or not isinstance(vad_filter, bool):
        raise WorkerError("invalid_worker_input")
    beam_size = integer(parameters.get("beam_size", 5), 1, 10)
    return Options(inputs[0], language, vad_filter, beam_size)


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def check_cuda(ctranslate2) -> None:
    try:
        if ctranslate2.get_cuda_device_count() < 1:
            raise WorkerError("cuda_unavailable")
        for library in ("libcublas.so.12", "libcudnn.so.9"):
            ctypes.CDLL(library)
        if "float16" not in ctranslate2.get_supported_compute_types("cuda", 0):
            raise WorkerError("incompatible_cuda")
    except (OSError, RuntimeError):
        raise WorkerError("incompatible_cuda") from None


def memory_used_mib() -> int | None:
    try:
        sample = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        return int(sample.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def segments_and_srt(segments, duration: float) -> tuple[list[dict], str]:
    rows = []
    for piece in segments:
        if not math.isfinite(piece.start) or not math.isfinite(piece.end):
            continue
        start = round(max(0, min(duration, piece.start)) * 1000)
        end = round(max(0, min(duration, piece.end)) * 1000)
        text = " ".join(piece.text.split())
        if text and end > start:
            rows.append((start, end, text))
    rows.sort()
    result = [
        {"start": start / 1000, "end": end / 1000, "text": text}
        for start, end, text in rows
    ]
    srt = [
        f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n"
        for index, (start, end, text) in enumerate(rows, 1)
    ]
    return result, "\n".join(srt)


def timestamp(milliseconds: int) -> str:
    seconds, millis = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def run() -> dict:
    settings = options(*request_fields())
    software = ensure_packages(PINS)
    try:
        import ctranslate2
        from av.error import FFmpegError
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        from faster_whisper.utils import download_model
    except (ImportError, OSError):
        raise WorkerError("worker_dependencies") from None

    try:
        audio = decode_audio(settings.path, sampling_rate=16_000)
    except (FFmpegError, OSError, ValueError):
        raise WorkerError("invalid_worker_input") from None
    duration = len(audio) / 16_000
    if duration <= 0:
        raise WorkerError("invalid_worker_input")

    check_cuda(ctranslate2)
    started = perf_counter()
    try:
        model_path = download_model("small")
    except Exception:  # noqa: BLE001 - the Hub may include credentials in errors.
        raise WorkerError("model_download_failed") from None
    download_time = perf_counter() - started
    baseline = memory_used_mib()
    started = perf_counter()
    try:
        model = WhisperModel(model_path, device="cuda", compute_type="float16")
    except (RuntimeError, OSError, ValueError) as error:
        raise WorkerError(cuda_error(error, "model_load_failed")) from None
    load_time = perf_counter() - started
    loaded = memory_used_mib()
    started = perf_counter()
    try:
        segments, info = model.transcribe(
            audio,
            language=None if settings.language == "auto" else settings.language,
            vad_filter=settings.vad_filter,
            beam_size=settings.beam_size,
        )
        rows, srt = segments_and_srt(segments, duration)
    except (RuntimeError, OSError, ValueError) as error:
        raise WorkerError(cuda_error(error, "worker_failed")) from None
    inference_time = perf_counter() - started
    inferred = memory_used_mib()
    metrics = {
        "model_id": MODEL_ID,
        "model_revision": Path(model_path).name,
        "software": software,
        "model_download_time_ms": download_time * 1000,
        "model_load_time_ms": load_time * 1000,
        "inference_time_ms": inference_time * 1000,
        "real_time_factor": inference_time / duration,
        "gpu_memory_delta_mib": (
            max(0, max(loaded, inferred) - baseline)
            if None not in (loaded, inferred, baseline)
            else None
        ),
    }
    write_json(
        ARTIFACTS[0],
        {
            "segments": rows,
            "language": info.language,
            "duration_sec": duration,
            "model": "small",
            "metrics": metrics,
        },
    )
    Path(ARTIFACTS[1]).write_text(srt, encoding="utf-8")
    return metrics


def cuda_error(error: Exception, fallback: str) -> str:
    message = str(error).lower()
    if "out of memory" in message or "cuda_error_out_of_memory" in message:
        return "out_of_memory"
    if any(name in message for name in ("cublas", "cudnn", "cuda driver")):
        return "incompatible_cuda"
    return fallback
