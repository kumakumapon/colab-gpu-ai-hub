import json
import subprocess
from zipfile import ZipFile

import pytest

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.pipelines import video
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts
from gpu_ai.workers._runtime import WorkerError, write_json


@pytest.mark.parametrize(
    ("audio", "tracking", "fail_depth"),
    [(True, True, False), (False, False, False), (True, False, True)],
)
def test_video_pipeline(
    tmp_path, monkeypatch, simulated_colab, audio, tracking, fail_depth
):
    workspace = tmp_path / "input"
    workspace.mkdir()
    source = workspace / "sample.mp4"
    sound = (
        ["-f", "lavfi", "-i", "sine=frequency=440:duration=2.5", "-c:a", "aac"]
        if audio
        else []
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=5:duration=2.5",
            *sound,
            "-vf",
            "select=not(eq(n\\,5))",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    source_times = [
        float(frame["best_effort_timestamp_time"])
        for frame in json.loads(probe.stdout)["frames"]
    ]
    invoked = []

    def worker(module, folder, inputs, parameters):
        invoked.append(module)
        if module == "whisper":
            assert (folder / inputs[0]).read_bytes().startswith(b"RIFF")
            write_json(
                folder / "transcript.json",
                {"segments": [{"start": 0.2, "end": 0.8, "text": "確認"}]},
            )
            (folder / "transcript.srt").write_text(
                "1\n00:00:00,200 --> 00:00:00,800\n確認\n"
            )
        elif module == "depth":
            if fail_depth:
                raise WorkerError("model_load_failed")
            for index, name in enumerate(inputs, 1):
                assert (folder / name).read_bytes().startswith(b"\x89PNG")
                directory = (
                    folder / f"images/{index:04d}" if len(inputs) > 1 else folder
                )
                directory.mkdir(parents=True, exist_ok=True)
                for file in ("depth.npy", "depth_preview.png", "metadata.json"):
                    (directory / file).write_bytes(b"mock depth")
        elif module == "sam2":
            count = parameters["frame_count"]
            assert count == len(source_times)
            (folder / "masks").mkdir()
            rows = []
            for index in range(count):
                mask = f"masks/frame_{index:06d}_object_0001.png"
                (folder / mask).write_bytes(b"mock mask")
                rows.append(
                    {
                        "frame_index": index,
                        "object_id": 1,
                        "mask_path": mask,
                        "box": [2, 2, 16, 16],
                    }
                )
            write_json(folder / "tracks.json", {"tracks": rows})
            write_json(folder / "metadata.json", {})
        elif module == "llm":
            assert len(parameters["prompt"]) < 3500
            write_json(folder / "response.json", {"text": "<b>推定結果</b>"})
        return {"inference_time_ms": 10}

    monkeypatch.setattr(video, "infer", worker)
    calls = simulated_colab(video.analyze_video, "video")
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    request = TaskRequest(
        operation=Operation.ANALYZE_VIDEO,
        inputs=[source.name],
        parameters={
            "frame_interval_seconds": 1,
            "max_frames": 3,
            "objects": [{"object_id": 1, "point": [4, 5]}] if tracking else [],
        },
    )
    runner = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    )
    result = runner.run(request)
    assert result.status is JobStatus.COMPLETED
    assert result.cleanup_status is CleanupStatus.CONFIRMED
    assert result.metrics["partial"] is fail_depth
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert ("whisper" in invoked) is audio and ("sam2" in invoked) is tracking
    root = settings.results_dir / str(request.job_id)
    assert bool((root / "transcript.srt").read_bytes()) is audio
    assert "\n    <b>推定結果</b>" in (root / "report.md").read_text()
    timeline = json.loads((root / "timeline.json").read_text())
    assert timeline["stages"]["speech"]["status"] == (
        "completed" if audio else "skipped"
    )
    assert timeline["stages"]["tracking"]["status"] == (
        "completed" if tracking else "skipped"
    )
    assert timeline["stages"]["depth"]["status"] == (
        "failed" if fail_depth else "completed"
    )
    estimates = [
        event
        for event in timeline["events"]
        if event["evidence_type"] == "model_estimate"
    ]
    depth_times = [
        event["time_sec"] for event in estimates if event["kind"] == "relative_depth"
    ]
    assert depth_times == ([] if fail_depth else [0.0, 1.2, 2.2])
    if tracking:
        assert [
            event["time_sec"] for event in estimates if event["kind"] == "object_mask"
        ] == source_times
    with ZipFile(root / "evidence.zip") as bundle:
        for event in estimates:
            assert event["artifact"]["member"] in bundle.namelist()
            assert event["evidence_type"] == "model_estimate"


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_frames": 25},
        {"frame_interval_seconds": 0},
        {"objects": [{"object_id": 1}]},
    ],
)
def test_invalid_video_parameters(parameters):
    with pytest.raises(WorkerError):
        video.options(["video.mp4"], parameters)
