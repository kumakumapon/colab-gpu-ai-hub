"""Durable job records and a workspace-wide, cross-process GPU lock."""

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from pydantic import Field

from gpu_ai.contracts import CleanupStatus, Contract, JobResult, JobStatus

FINAL_STATES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


class JobBusy(RuntimeError):
    """A job is active or its GPU release is still unconfirmed."""


class JobRecord(Contract):
    job_id: UUID
    backend: str
    request_sha256: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    history: list[tuple[datetime, JobStatus]] = Field(default_factory=list)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None
    result: JobResult | None = None
    outcome_status: JobStatus | None = None
    cleanup_status: CleanupStatus = CleanupStatus.UNKNOWN

    def advance(self, status: JobStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(UTC)
        self.history.append((self.updated_at, status))


class JobLedger:
    def __init__(self, workspace: Path) -> None:
        self.directory = workspace.resolve() / ".gpu-ai-jobs"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError("job ledger must be a real directory")
        fd = os.open(
            self.directory / "gpu.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        with os.fdopen(fd, "ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise JobBusy("another job or reconciliation is active") from None
            yield

    def get(self, job_id: UUID) -> JobRecord | None:
        try:
            data = (self.directory / f"{job_id}.json").read_bytes()
        except FileNotFoundError:
            return None
        return JobRecord.model_validate_json(data)

    def request_cancel(self, job_id: UUID) -> JobRecord:
        record = self.get(job_id)
        if record is None:
            raise LookupError("job_not_found")
        if record.status not in FINAL_STATES | {
            JobStatus.CLEANUP_PENDING,
            JobStatus.STOPPING,
        }:
            marker = self.directory / f"{job_id}.cancel"
            try:
                fd = os.open(
                    marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
            except FileExistsError:
                if marker.is_symlink() or not marker.is_file():
                    raise ValueError("invalid cancellation marker") from None
            else:
                os.close(fd)
        return record

    def cancellation_requested(self, job_id: UUID) -> bool:
        marker = self.directory / f"{job_id}.cancel"
        return not marker.is_symlink() and marker.is_file()

    def unfinished(self) -> Iterator[JobRecord]:
        for path in sorted(self.directory.glob("*.json")):
            record = JobRecord.model_validate_json(path.read_bytes())
            if record.status not in FINAL_STATES:
                yield record

    def save(self, record: JobRecord) -> None:
        with NamedTemporaryFile(dir=self.directory, delete=False) as temporary:
            path = Path(temporary.name)
            try:
                temporary.write(record.model_dump_json().encode())
                temporary.flush()
                os.fsync(temporary.fileno())
                path.replace(self.directory / f"{record.job_id}.json")
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                path.unlink(missing_ok=True)
