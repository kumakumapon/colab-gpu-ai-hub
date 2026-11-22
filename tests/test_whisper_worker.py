import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, whisper
from gpu_ai.workers._runtime import WorkerError


@pytest.fixture
def japanese_model(monkeypatch):
    monkeypatch.setattr(whisper, "ensure_packages", lambda pins: pins)
    monkeypatch.setattr(whisper, "check_cuda", lambda _: None)
    monkeypatch.setattr(whisper, "memory_used_mib", lambda: 1200)

    def download(model):
        assert model == "small"
        return "/cache/snapshots/fixture-revision"

    class Model:
        def __init__(self, path, *, device, compute_type):
            assert path.endswith("/fixture-revision")
            assert (device, compute_type) == ("cuda", "float16")

        def transcribe(self, audio, **options):
            assert len(audio) == 32_000
            assert options == {"language": "ja", "vad_filter": True, "beam_size": 3}
            return iter(
                [
                    SimpleNamespace(start=1, end=4, text=" 世界 "),
                    SimpleNamespace(start=-1, end=0.8, text="こんにちは"),
                    SimpleNamespace(start=0.8, end=1, text="  "),
                    SimpleNamespace(start=float("nan"), end=1.1, text="skip"),
                ]
            ), SimpleNamespace(language="ja")

    def decode(path, *, sampling_rate):
        assert path == "sample.wav"
        assert sampling_rate == 16_000
        return range(32_000)

    modules = {
        "ctranslate2": SimpleNamespace(),
        "av.error": SimpleNamespace(FFmpegError=ValueError),
        "faster_whisper": SimpleNamespace(WhisperModel=Model),
        "faster_whisper.audio": SimpleNamespace(decode_audio=decode),
        "faster_whisper.utils": SimpleNamespace(download_model=download),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_japanese_audio_through_colab_wrapper(
    tmp_path, simulated_colab, japanese_model
):
    workspace = tmp_path / "inputs"
    workspace.mkdir()
    with wave.open(str(workspace / "sample.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 32_000)
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    calls = simulated_colab(whisper.run, "whisper")
    backend = ColabCLIBackend(colab_worker_scripts(), settings=settings)
    result = JobRunner(backend, settings).run(
        TaskRequest(
            operation=Operation.TRANSCRIBE_AUDIO,
            inputs=["sample.wav"],
            parameters={"beam_size": 3},
        )
    )
    assert result.status == JobStatus.COMPLETED
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert result.metrics["model_id"] == whisper.MODEL_ID
    output = settings.results_dir / str(result.job_id)
    document = json.loads((output / "transcript.json").read_text(encoding="utf-8"))
    assert document["duration_sec"] == 2
    assert document["language"] == "ja"
    assert document["segments"] == [
        {"start": 0, "end": 0.8, "text": "こんにちは"},
        {"start": 1, "end": 2, "text": "世界"},
    ]
    assert (output / "transcript.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:00,800\nこんにちは\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n世界\n"
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"language": "en"},
        {"beam_size": 0},
        {"beam_size": True},
        {"vad_filter": "no"},
        {"model": "large"},
    ],
)
def test_rejects_invalid_transcription_settings(parameters):
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        whisper.options(["test.wav"], parameters)


def test_empty_audio_is_an_explicit_failure(tmp_path, monkeypatch, japanese_model):
    monkeypatch.chdir(tmp_path)
    options = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["sample.wav"])
    Path("gpu-ai-request.json").write_text(options.model_dump_json(), encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper.audio",
        SimpleNamespace(decode_audio=lambda *a, **kw: []),
    )
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        whisper.run()
    assert whisper.segments_and_srt([], 1) == ([], "")


def test_worker_error_is_reported_and_releases_gpu(tmp_path, simulated_colab):
    def fail():
        raise WorkerError("incompatible_cuda")

    calls = simulated_colab(fail, "whisper")
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    backend = ColabCLIBackend(colab_worker_scripts(), settings=settings)
    result = JobRunner(backend, settings).run(
        TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["sample.wav"]),
        {"sample.wav": b"input"},
    )
    assert result.error == "execute:incompatible_cuda"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "stop"]
    assert not settings.results_dir.exists()


def test_invalid_settings_do_not_launch_colab(tmp_path, simulated_colab):
    calls = simulated_colab(whisper.run, "whisper")
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    backend = ColabCLIBackend(colab_worker_scripts(), settings=settings)
    result = JobRunner(backend, settings).run(
        TaskRequest(
            operation=Operation.TRANSCRIBE_AUDIO,
            inputs=["sample.wav"],
            parameters={"beam_size": 99},
        ),
        {"sample.wav": b"input"},
    )
    assert result.error == "provision:invalid_worker_input"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == []
