from hashlib import sha256

import pytest
from pydantic import ValidationError

from gpu_ai.backends.base import Backend, MockBackend
from gpu_ai.config import Settings, load_settings
from gpu_ai.contracts import CleanupStatus, JobResult, JobStatus, Operation, TaskRequest
from gpu_ai.registry import WorkerRegistry


@pytest.mark.parametrize(
    "override",
    [
        {"operation": "unknown"},
        {"gpu": "invalid"},
        {"inputs": ["../private.wav"]},
        {"inputs": ["/private.wav"]},
        {"inputs": ["C:/private.wav"]},
        {"inputs": ["media\\private.wav"]},
        {"inputs": [""]},
        {"extra": 1},
    ],
)
def test_rejects_invalid_request(override):
    with pytest.raises(ValidationError):
        TaskRequest.model_validate({"operation": "transcribe_audio", **override})


@pytest.mark.parametrize(
    "override",
    [
        {"max_input_bytes": -1},
        {"max_runtime_seconds": 0},
        {"max_concurrent_jobs": 2},
        {"workspace": ""},
        {"extra": 1},
    ],
)
def test_rejects_invalid_settings(override):
    with pytest.raises(ValidationError):
        Settings.model_validate(override)


def test_loads_file_with_environment_overrides(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text('workspace = "media"\ngpu = "L4"\nmax_input_bytes = 1024\n')
    settings = load_settings(config, environ={"GPU_AI_GPU": "T4"})
    assert settings.workspace.name == "media"
    assert settings.gpu == "T4"
    assert settings.max_input_bytes == 1024


def test_mock_job_round_trip():
    request = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["sample.wav"])
    backend: Backend = MockBackend({"transcript.json": b"{}"})
    session_id = backend.provision(request)
    assert backend.provision(request) == str(request.job_id)
    assert backend.status(session_id) == JobStatus.STARTING
    backend.upload(session_id, {"sample.wav": b"mock audio"})

    registry = WorkerRegistry()
    registry.register(request.operation, lambda task: backend.execute(session_id, task))
    result = registry.resolve(request.operation)(request)
    downloads = backend.download(session_id)
    assert result.artifacts[0].sha256 == sha256(downloads["transcript.json"]).hexdigest()
    assert result.status is JobStatus.COMPLETED
    assert result.cleanup_status is CleanupStatus.UNKNOWN

    result.cleanup_status = backend.stop(session_id)
    serialized = JobResult.model_validate_json(result.model_dump_json())
    assert serialized == result
    assert serialized.schema_version == "1"
    assert serialized.cleanup_status is CleanupStatus.CONFIRMED


def test_mock_rejects_incomplete_upload_and_conflicting_job():
    request = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["sample.wav"])
    backend = MockBackend()
    session_id = backend.provision(request)
    with pytest.raises(RuntimeError, match="not been uploaded"):
        backend.execute(session_id, request)
    with pytest.raises(ValueError, match="different request"):
        backend.provision(request.model_copy(update={"operation": Operation.ESTIMATE_DEPTH}))
    assert backend.stop(session_id) is CleanupStatus.CONFIRMED
    assert backend.stop(session_id) is CleanupStatus.CONFIRMED
    assert backend.status(session_id) is JobStatus.CANCELLED


def test_registry_rejects_unregistered_and_duplicate_operations():
    registry = WorkerRegistry()
    operation = Operation.LLM_INFER
    with pytest.raises(LookupError):
        registry.resolve(operation)

    def worker(request):
        return JobResult(job_id=request.job_id, status=JobStatus.COMPLETED)

    registry.register(operation, worker)
    assert registry.resolve(operation) is worker
    with pytest.raises(ValueError, match="already registered"):
        registry.register(operation, worker)
