import json
import stat
import subprocess
from hashlib import sha256
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from gpu_ai.artifacts import (
    RESULT_MANIFEST,
    TRANSFER_MANIFEST,
    ArtifactStore,
    read_transfer_archive,
)
from gpu_ai.backends.base import MockBackend
from gpu_ai.backends.colab_cli import ColabCLIBackend, WorkerScript
from gpu_ai.config import Settings
from gpu_ai.contracts import Artifact, CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner


def artifact(data=b"trusted"):
    return Artifact(
        path="reports/result.json",
        size=len(data),
        sha256=sha256(data).hexdigest(),
        mime_type="application/json",
    )


def make_zip(path, member, content, manifest):
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member, content)
        archive.writestr(TRANSFER_MANIFEST, json.dumps([manifest.model_dump()]))


@pytest.mark.parametrize(
    "problem", ["traversal", "symlink", "oversize", "hash", "extra"]
)
def test_rejects_invalid_output_zip(tmp_path, problem):
    archive_path = tmp_path / "output.zip"
    item = artifact()
    entry, data, max_bytes = item.path, b"trusted", 1024
    if problem == "traversal":
        entry = "../escape"
    elif problem == "symlink":
        entry = ZipInfo(item.path)
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    elif problem == "oversize":
        data = b"x" * 1025
        item = artifact(data)
    elif problem == "hash":
        data = b"altered"
    make_zip(archive_path, entry, data, item)
    if problem == "extra":
        with ZipFile(archive_path, "a") as archive:
            archive.writestr("credentials/token.txt", b"token")
    with pytest.raises(ValueError):
        read_transfer_archive(archive_path, (item.path,), max_bytes=max_bytes)


def test_valid_transfer_zip(tmp_path):
    archive_path = tmp_path / "output.zip"
    item = artifact()
    make_zip(archive_path, item.path, b"trusted", item)
    assert read_transfer_archive(archive_path, (item.path,), max_bytes=1024) == (
        [item],
        {item.path: b"trusted"},
    )


def test_results_are_immutable_and_job_scoped(tmp_path):
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    output_path = "nested/result.json"
    backend = MockBackend({output_path: b"{}"})
    runner = JobRunner(backend, settings)
    first = TaskRequest(operation=Operation.LLM_INFER)
    second = TaskRequest(operation=Operation.LLM_INFER)
    result = runner.run(first)
    assert result.status == JobStatus.COMPLETED
    assert runner.run(second).status == JobStatus.COMPLETED
    for request in (first, second):
        target = settings.results_dir / str(request.job_id)
        assert (target / output_path).read_bytes() == b"{}"
        assert (target / RESULT_MANIFEST).is_file()
    store = ArtifactStore(settings.results_dir, max_bytes=1024)
    with pytest.raises(FileExistsError):
        store.save(result, {output_path: b"{}"}, {}, {}, {})
    assert not list(settings.results_dir.glob(".gpu-ai-results-*"))


def test_invalid_archive_fails_before_publish_and_releases_gpu(tmp_path, monkeypatch):
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    item = artifact()
    calls = []

    def run(args, **options):
        command = args[4]
        calls.append(command)
        if command == "sessions":
            result = "[colab] No active sessions found on server.\n"
        elif command == "new":
            result = "[colab] Session READY.\n"
        elif command == "exec":
            from pathlib import Path

            wrapper = Path(args[8]).read_text()
            marker = wrapper.rsplit("print('", 1)[1].split("'", 1)[0]
            result = marker + json.dumps([item.model_dump()]) + "\n"
        elif command == "download":
            make_zip(args[-1], item.path, b"altered", item)
            result = "downloaded\n"
        elif command == "stop":
            result = "[colab] Session terminated.\n"
        else:
            result = ""
        return subprocess.CompletedProcess(args, 0, result, "")

    monkeypatch.setattr(subprocess, "run", run)
    request = TaskRequest(operation=Operation.LLM_INFER)
    backend = ColabCLIBackend(
        {request.operation: WorkerScript("", (item.path,))}, settings=settings
    )
    result = JobRunner(backend, settings).run(request)
    assert result.status == JobStatus.FAILED
    assert result.error == "download:invalid_response"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert not settings.results_dir.exists()
