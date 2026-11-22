import subprocess
from datetime import UTC, datetime
from threading import Event, Thread

import pytest

from gpu_ai.backends.base import MockBackend, job_deadline
from gpu_ai.backends.colab_cli import (
    ColabCLIBackend,
    ColabCLIError,
    ErrorCode,
    WorkerScript,
)
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.jobs import JobBusy, JobRecord
from gpu_ai.runner import JobRunner


@pytest.fixture
def request_data():
    return TaskRequest(operation=Operation.LLM_INFER, inputs=["prompt.txt"])


@pytest.fixture
def runner(tmp_path):
    return JobRunner(
        MockBackend({"result.txt": b"output"}),
        Settings(workspace=tmp_path, results_dir=tmp_path / "results"),
    )


def test_success_and_deduplication(runner, request_data, monkeypatch):
    output = runner.run(request_data, {"prompt.txt": b"prompt"})
    assert output.status == JobStatus.COMPLETED
    assert output.cleanup_status == CleanupStatus.CONFIRMED
    record = runner.ledger.get(request_data.job_id)
    assert record.result == output
    assert len(record.input_hashes["prompt.txt"]) == 64
    assert [status for _, status in record.history] == [
        JobStatus.QUEUED,
        JobStatus.STARTING,
        JobStatus.UPLOADING,
        JobStatus.RUNNING,
        JobStatus.DOWNLOADING,
        JobStatus.STOPPING,
        JobStatus.COMPLETED,
    ]
    assert (
        b'"prompt"'
        not in (runner.ledger.directory / f"{request_data.job_id}.json").read_bytes()
    )
    monkeypatch.setattr(
        runner.backend, "provision_owned", lambda *_: pytest.fail("duplicate")
    )
    assert runner.run(request_data) == output
    request_data.parameters["new"] = True
    with pytest.raises(ValueError, match="different request"):
        runner.run(request_data)


@pytest.mark.parametrize(
    ("method", "error_code"),
    [
        ("provision_owned", "provision_failed"),
        ("upload", "upload_failed"),
        ("execute", "execute_failed"),
        ("download", "download_failed"),
    ],
)
def test_failure_releases_ownership(
    runner, request_data, monkeypatch, method, error_code
):
    recovered = []
    recover = runner.backend.recover_owned

    def fail(*args):
        if method == "provision_owned":
            MockBackend.provision_owned(runner.backend, *args)
        raise RuntimeError("private details")

    def spy(session):
        recovered.append(session)
        return recover(session)

    monkeypatch.setattr(runner.backend, method, fail)
    monkeypatch.setattr(runner.backend, "recover_owned", spy)
    output = runner.run(request_data, {"prompt.txt": b"prompt"})
    assert output.status == JobStatus.FAILED
    assert output.cleanup_status == CleanupStatus.CONFIRMED
    assert output.error == error_code
    assert recovered == [runner.ledger.get(request_data.job_id).session_id]


def test_auth_failure_precedes_new(tmp_path, monkeypatch):
    commands = []

    def no_auth(args, **options):
        commands.append(args[4])
        return subprocess.CompletedProcess(args, 1, "", "invalid_grant")

    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    request = TaskRequest(operation=Operation.LLM_INFER)
    backend = ColabCLIBackend(
        {request.operation: WorkerScript("", ())}, settings=settings
    )
    monkeypatch.setattr(subprocess, "run", no_auth)
    result = JobRunner(backend, settings).run(request)
    assert commands == ["sessions"]
    assert result.error == "provision:auth_required"
    assert result.cleanup_status == CleanupStatus.CONFIRMED


@pytest.mark.parametrize("status", [CleanupStatus.FAILED, CleanupStatus.UNKNOWN])
def test_stop_failure_blocks_new_job_until_reconciled(
    runner, request_data, monkeypatch, status
):
    original = runner.backend.stop
    monkeypatch.setattr(runner.backend, "stop", lambda _: status)
    result = runner.run(request_data, {"prompt.txt": b"prompt"})
    assert result.status == JobStatus.CLEANUP_PENDING
    assert result.cleanup_status == status
    assert result.artifacts
    with pytest.raises(JobBusy):
        runner.run(TaskRequest(operation=request_data.operation))
    assert runner.reconcile()[0].result == result
    monkeypatch.setattr(runner.backend, "stop", original)
    reconciled = runner.reconcile(retry=True)[0]
    assert reconciled.result.status == JobStatus.COMPLETED
    assert reconciled.result.cleanup_status == CleanupStatus.CONFIRMED
    assert reconciled.result.error is None


def test_cancel_during_upload(runner, request_data, monkeypatch):
    cancel = Event()
    monkeypatch.setattr(runner.backend, "upload", lambda *_: cancel.set())
    monkeypatch.setattr(runner.backend, "execute", lambda *_: pytest.fail("cancelled"))
    result = runner.run(request_data, {"prompt.txt": b"prompt"}, cancel=cancel)
    assert result.status == JobStatus.CANCELLED
    assert result.artifacts == []
    assert result.cleanup_status == CleanupStatus.CONFIRMED


def test_deadline_spans_calls_and_cleanup_has_separate_budget(
    runner, request_data, monkeypatch
):
    times = iter([100, 100, 101, 101, 103])
    runner.settings.max_runtime_seconds = 2
    monkeypatch.setattr("gpu_ai.runner.monotonic", lambda: next(times))
    monkeypatch.setattr(runner.backend, "execute", lambda *_: pytest.fail("timed out"))
    original = runner.backend.stop

    def stop(session):
        assert job_deadline.get() is None
        return original(session)

    monkeypatch.setattr(runner.backend, "stop", stop)
    result = runner.run(request_data, {"prompt.txt": b"prompt"})
    assert result.error == "wall_clock_timeout"
    assert result.cleanup_status == CleanupStatus.CONFIRMED


def test_hash_mismatch_fails_before_results_are_saved(
    runner, request_data, monkeypatch
):
    monkeypatch.setattr(runner.backend, "download", lambda _: {"result.txt": b"tamper"})
    result = runner.run(
        request_data,
        {"prompt.txt": b"prompt"},
        save_results=lambda *_: pytest.fail("unverified result"),
    )
    assert result.error == "verify_failed"


def test_concurrent_runners_share_a_gpu_lock(runner, request_data, monkeypatch):
    ready, resume = Event(), Event()
    reserve = runner.backend.provision_owned
    results = []

    def wait_for_second_run(*args):
        ready.set()
        assert resume.wait(timeout=5)
        return reserve(*args)

    monkeypatch.setattr(runner.backend, "provision_owned", wait_for_second_run)
    worker = Thread(
        target=lambda: results.append(
            runner.run(request_data, {"prompt.txt": b"prompt"})
        )
    )
    worker.start()
    try:
        assert ready.wait(timeout=5)
        other = JobRunner(runner.backend, runner.settings)
        with pytest.raises(JobBusy):
            other.run(TaskRequest(operation=request_data.operation))
    finally:
        resume.set()
        worker.join(timeout=5)
    assert results[0].status == JobStatus.COMPLETED
    assert len(runner.backend._sessions) == 1


def test_restart_recovery_uses_only_owned_session(tmp_path, monkeypatch):
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    created = JobRunner(ColabCLIBackend({}, settings=settings), settings)
    task = TaskRequest(operation=Operation.LLM_INFER)
    owned = "gpu-ai-" + "a" * 32
    root = tmp_path / ".colab-cli" / owned
    root.mkdir(parents=True)
    config = root / "sessions.json"
    config.write_text('{"' + owned + '": {"endpoint": "owned-endpoint"}}')
    (root / "allocation-intent").touch()
    now = datetime.now(UTC)
    record = JobRecord(
        job_id=task.job_id,
        backend="colab-cli",
        request_sha256="test",
        status=JobStatus.STARTING,
        session_id=owned,
        created_at=now,
        updated_at=now,
    )
    with created.ledger.exclusive():
        created.ledger.save(record)

    commands = []

    def run(args, **options):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "[colab] Session terminated.\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    restarted = JobRunner(ColabCLIBackend({}, settings=settings), settings)
    assert restarted.reconcile()[0].session_id == owned
    assert not commands
    recovered = restarted.reconcile(retry=True)[0]
    assert commands == [
        ["colab", "--auth=oauth2", "--config", str(config), "stop", "-s", owned]
    ]
    assert recovered.result.status == JobStatus.FAILED
    assert recovered.result.error == "interrupted"
    assert recovered.cleanup_status == CleanupStatus.CONFIRMED


def test_colab_cli_timeout_respects_job_deadline(tmp_path, monkeypatch):
    from gpu_ai.backends.colab_cli import ColabCLI

    def run(args, **options):
        assert options["timeout"] == 0.5
        raise subprocess.TimeoutExpired(args, 0.5)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("gpu_ai.backends.colab_cli.monotonic", lambda: 100)
    token = job_deadline.set(100.5)
    try:
        with pytest.raises(ColabCLIError) as caught:
            ColabCLI().check_auth(tmp_path / "sessions.json")
    finally:
        job_deadline.reset(token)
    assert caught.value.code == ErrorCode.TIMEOUT
