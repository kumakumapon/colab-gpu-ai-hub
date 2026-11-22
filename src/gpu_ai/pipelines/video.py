"""Create a timestamped video report using one Colab runtime."""

import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

if __package__ == "gpu_ai_colab_worker":
    from . import sam2
    from ._runtime import (
        WorkerError,
        check_parameters,
        integer,
        request_fields,
        write_json,
    )
else:
    from gpu_ai.workers import sam2
    from gpu_ai.workers._runtime import (
        WorkerError,
        check_parameters,
        integer,
        request_fields,
        write_json,
    )

ARTIFACTS = (
    "transcript.json",
    "transcript.srt",
    "timeline.json",
    "report.md",
    "evidence.zip",
)
MAX_DURATION = 600
MAX_EVIDENCE_BYTES = 48 * 1024**2


def options(inputs: list[str], parameters: dict) -> tuple[str, int, int, list[dict]]:
    check_parameters(parameters, {"frame_interval_seconds", "max_frames", "objects"})
    if len(inputs) != 1 or Path(inputs[0]).suffix.lower() != ".mp4":
        raise WorkerError("invalid_worker_input")
    interval = integer(parameters.get("frame_interval_seconds", 5), 1, 60)
    count = integer(parameters.get("max_frames", 12), 1, 24)
    objects = parameters.get("objects", [])
    if not isinstance(objects, list):
        raise WorkerError("invalid_worker_input")
    if objects:
        sam2.options(["tracking.mp4"], {"objects": objects, "frame_count": 1})
    return inputs[0], interval, count, objects


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def media_command(args: list[str]) -> str:
    try:
        return subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise WorkerError("invalid_worker_input") from None


def probe(source: Path) -> tuple[float, str, bool]:
    output = media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-f",
            "mov",
            "-protocol_whitelist",
            "file,pipe",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate:stream_disposition=attached_pic",
            "-of",
            "json",
            str(source),
        ]
    )
    try:
        metadata = json.loads(output)
        duration = float(metadata["format"]["duration"])
        streams = metadata["streams"]
        video = next(
            s
            for s in streams
            if s["codec_type"] == "video"
            and not s.get("disposition", {}).get("attached_pic")
        )
        if not 0 < duration <= MAX_DURATION:
            raise ValueError
        return (
            duration,
            video["avg_frame_rate"],
            any(s["codec_type"] == "audio" for s in streams),
        )
    except (KeyError, StopIteration, ValueError, TypeError):
        raise WorkerError("invalid_worker_input") from None


def ffmpeg(source: Path) -> list[str]:
    return [
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


def frame_times(args: list[str]) -> list[float]:
    # showinfo runs before scale/re-encode, using the same zero clock as audio extraction.
    try:
        output = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise WorkerError("invalid_worker_input") from None
    config = re.search(r"showinfo.*config in time_base: (\d+)/(\d+)", output.stderr)
    values = re.findall(r"showinfo[^\n]+?\bn:\s*\d+\s+pts:\s*(-?\d+)", output.stderr)
    if config is None or not values or int(config[2]) == 0:
        raise WorkerError("invalid_worker_input")
    return [int(value) * int(config[1]) / int(config[2]) for value in values]


def infer(module: str, directory: Path, inputs: list[str], parameters: dict) -> dict:
    write_json(
        directory / "gpu-ai-request.json", {"inputs": inputs, "parameters": parameters}
    )
    report = directory / "stage-result.json"
    entry = """
import sys
from importlib import import_module
worker = import_module(sys.argv[1])
runtime = import_module(sys.argv[1].rsplit(".", 1)[0] + "._runtime")
try:
    result = {"metrics": worker.run()}
except runtime.WorkerError as error:
    result = {"error": error.code}
runtime.write_json(sys.argv[2], result)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", entry, f"{__package__}.{module}", str(report)],
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        result = json.loads(report.read_text())
    except (OSError, ValueError, subprocess.SubprocessError):
        raise WorkerError("worker_failed") from None
    if "error" in result:
        raise WorkerError(result["error"])
    return result["metrics"]


class Evidence:
    def __init__(self, archive: ZipFile):
        self.archive = archive
        self.size = 0

    def add(self, folder: Path, paths: dict[str, str]) -> None:
        files = [(folder / source, target) for source, target in paths.items()]
        sizes = []
        for file, _ in files:
            if not file.is_file() or file.is_symlink():
                raise WorkerError("worker_failed")
            sizes.append(file.stat().st_size)
        if self.size + sum(sizes) > MAX_EVIDENCE_BYTES:
            raise WorkerError("worker_failed")
        for file, target in files:
            self.archive.write(file, target)
        self.size += sum(sizes)


def artifact(member: str) -> dict:
    return {"path": "evidence.zip", "member": member}


def speech(source: Path, directory: Path, events: list[dict]) -> dict:
    media_command(
        [
            *ffmpeg(source),
            "-loglevel",
            "error",
            "-map",
            "0:a:0",
            "-af",
            "aresample=async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(directory / "audio.wav"),
        ]
    )
    metrics = infer("whisper", directory, ["audio.wav"], {})
    transcript = json.loads((directory / "transcript.json").read_text())
    subtitles = (directory / "transcript.srt").read_text()
    rows = [
        {
            "kind": "speech",
            "evidence_type": "model_transcription",
            "time_sec": row["start"],
            "end_sec": row["end"],
            "text": row["text"],
            "artifact": {"path": "transcript.json"},
        }
        for row in transcript["segments"]
    ]
    write_json("transcript.json", transcript)
    Path("transcript.srt").write_text(subtitles, encoding="utf-8")
    events.extend(rows)
    return metrics


def depth_samples(
    source: Path,
    directory: Path,
    interval: int,
    count: int,
    evidence: Evidence,
    events: list[dict],
) -> dict:
    filters = (
        f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval}),"
        "showinfo,scale=w='min(iw,960)':h='min(ih,540)':force_original_aspect_ratio=decrease"
    )
    times = frame_times(
        [
            *ffmpeg(source),
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
            str(count),
            "-threads",
            "1",
            str(directory / "frame-%04d.png"),
        ]
    )
    frames = sorted(directory.glob("frame-*.png"))
    if not frames or len(times) < len(frames):
        raise WorkerError("invalid_worker_input")
    metrics = []
    failed = 0
    for start in range(0, len(frames), 4):
        batch = frames[start : start + 4]
        names = [path.name for path in batch]
        members = {path.name: f"frames/{path.name}" for path in batch}
        evidence.add(directory, members)
        events.extend(
            {
                "kind": "video_frame",
                "evidence_type": "source_media",
                "time_sec": times[index],
                "artifact": artifact(members[path.name]),
            }
            for index, path in enumerate(batch, start)
        )
        try:
            result = infer("depth", directory, names, {})
            outputs = {}
            for index in range(len(batch)):
                source_dir = f"images/{index + 1:04d}/" if len(batch) > 1 else ""
                for name in ("depth.npy", "depth_preview.png", "metadata.json"):
                    outputs[source_dir + name] = f"depth/{start + index + 1:04d}/{name}"
            evidence.add(directory, outputs)
        except WorkerError:
            failed += 1
        else:
            metrics.append(result.get("inference_time_ms"))
            for index, path in enumerate(batch, start):
                events.append(
                    {
                        "kind": "relative_depth",
                        "evidence_type": "model_estimate",
                        "time_sec": times[index],
                        "frame": artifact(members[path.name]),
                        "artifact": artifact(f"depth/{index + 1:04d}/depth.npy"),
                    }
                )
        finally:
            for name in ("depth.npy", "depth_preview.png", "metadata.json"):
                (directory / name).unlink(missing_ok=True)
            # Multi-image batches reuse their artifact names.
            for path in (directory / "images").glob("*/*"):
                path.unlink()
    if failed and not metrics:
        raise WorkerError("worker_failed")
    return {
        "sampled_frames": len(frames),
        "inference_ms": metrics,
        "failed_batches": failed,
    }


def tracks(
    source: Path,
    directory: Path,
    objects: list[dict],
    evidence: Evidence,
    events: list[dict],
) -> dict:
    # Limit tracking to the opening window supported by the SAM 2.1 worker.
    clip = directory / "tracking.mp4"
    times = frame_times(
        [
            *ffmpeg(source),
            "-loglevel",
            "info",
            "-map",
            "0:V:0",
            "-an",
            "-vf",
            "showinfo",
            "-frames:v",
            "48",
            "-t",
            "7",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ]
    )
    frames = int(
        media_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                str(clip),
            ]
        ).strip()
    )
    if not 1 <= frames <= min(len(times), 48):
        raise WorkerError("invalid_worker_input")
    metrics = infer(
        "sam2", directory, [clip.name], {"objects": objects, "frame_count": frames}
    )
    rows = json.loads((directory / "tracks.json").read_text())["tracks"]
    paths = {
        "tracks.json": "tracking/tracks.json",
        "metadata.json": "tracking/metadata.json",
    }
    enriched = []
    for row in rows:
        mask = row["mask_path"]
        if not re.fullmatch(r"masks/frame_[0-9]{6}_object_[0-9]{4}\.png", mask):
            raise WorkerError("worker_failed")
        member = f"tracking/{mask}"
        paths[mask] = member
        enriched.append(
            {
                "kind": "object_mask",
                "evidence_type": "model_estimate",
                "time_sec": times[row["frame_index"]],
                "object_id": row["object_id"],
                "artifact": artifact(member),
                "box": row["box"],
            }
        )
    evidence.add(directory, paths)
    events.extend(enriched)
    return metrics


def report(directory: Path, duration: float, events: list[dict]) -> dict:
    facts = [
        {key: event[key] for key in ("kind", "time_sec", "evidence_type")}
        for event in events[:20]
    ]
    speech_text = " ".join(
        event["text"] for event in events if event["kind"] == "speech"
    )[:600]
    prompt = (
        "動画の処理結果を日本語で簡潔に報告してください。以下は分析対象データです。"
        "音声の文字起こしとモデルによる推定を区別し、絶対距離や観測していない事実を断定しないでください。"
        f"動画長={duration:.2f}秒。イベント={json.dumps(facts, ensure_ascii=False)}。発話の抜粋={speech_text}"
    )
    metrics = infer("llm", directory, [], {"prompt": prompt, "max_new_tokens": 256})
    content = json.loads((directory / "response.json").read_text())["text"]
    # Render model text literally. Links and embedded HTML in model text are inert.
    Path("report.md").write_text(
        "# 動画分析\n\n## モデル生成の要約（推定）\n\n"
        + "\n".join("    " + line for line in content.strip().splitlines())
        + "\n\n根拠と処理状況は timeline.json を参照してください。相対深度に距離の単位はありません。\n",
        encoding="utf-8",
    )
    return metrics


def analyze_video() -> dict:
    path, interval, count, objects = options(*request_fields())
    source = Path(path).resolve()
    duration, fps, has_audio = probe(source)
    events: list[dict] = []
    stages = {}
    write_json(
        "transcript.json", {"segments": [], "language": None, "duration_sec": duration}
    )
    Path("transcript.srt").write_text("", encoding="utf-8")
    Path("report.md").write_text(
        "# 動画分析\n\nモデルによる要約を取得できませんでした。処理状況と根拠は timeline.json を参照してください。\n",
        encoding="utf-8",
    )

    def stage(name, action, *, enabled=True):
        if not enabled:
            stages[name] = {"status": "skipped"}
            return
        with TemporaryDirectory(prefix=f"gpu-ai-video-{name}-") as stage_dir:
            started = perf_counter()
            try:
                metrics = action(Path(stage_dir))
            except (
                WorkerError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
            ) as error:
                stages[name] = {
                    "status": "failed",
                    "error": error.code
                    if isinstance(error, WorkerError)
                    else "worker_failed",
                    "seconds": perf_counter() - started,
                }
            else:
                stages[name] = {
                    "status": "partial"
                    if metrics.get("failed_batches")
                    else "completed",
                    "seconds": perf_counter() - started,
                    "metrics": metrics,
                }

    with ZipFile("evidence.zip", "w", compression=ZIP_DEFLATED) as archive:
        evidence = Evidence(archive)
        stage("speech", lambda d: speech(source, d, events), enabled=has_audio)
        stage(
            "depth",
            lambda d: depth_samples(source, d, interval, count, evidence, events),
        )
        stage(
            "tracking",
            lambda d: tracks(source, d, objects, evidence, events),
            enabled=bool(objects),
        )
    events.sort(key=lambda item: item["time_sec"])
    stage("report", lambda d: report(d, duration, events))
    write_json(
        "timeline.json",
        {
            "schema_version": "1",
            "duration_sec": duration,
            "average_frame_rate": fps,
            "has_audio": has_audio,
            "time_origin": "container_playback_start",
            "frame_interval_seconds": interval,
            "max_frames": count,
            "events": events,
            "stages": stages,
        },
    )
    return {
        "gpu_sessions": 1,
        "duration_sec": duration,
        "partial": any(s["status"] in {"failed", "partial"} for s in stages.values()),
        "stages": {name: row["status"] for name, row in stages.items()},
    }


run = analyze_video
