"""Backend contract and an in-memory implementation for GPU-free tests."""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Protocol

from gpu_ai.contracts import Artifact, CleanupStatus, JobResult, JobStatus, TaskRequest


class Backend(Protocol):
    def provision(self, request: TaskRequest) -> str: ...
    def upload(self, session_id: str, files: dict[str, bytes]) -> None: ...
    def execute(self, session_id: str, request: TaskRequest) -> JobResult: ...
    def download(self, session_id: str) -> dict[str, bytes]: ...
    def status(self, session_id: str) -> JobStatus: ...
    def stop(self, session_id: str) -> CleanupStatus: ...


@dataclass
class _Session:
    request: TaskRequest
    status: JobStatus = JobStatus.STARTING
    inputs: dict[str, bytes] = field(default_factory=dict)
    result: JobResult | None = None
    stopped: bool = False


class MockBackend:
    def __init__(self, outputs: dict[str, bytes] | None = None) -> None:
        self._outputs = dict(outputs or {})
        self._artifacts = [
            Artifact(
                path=path,
                mime_type="application/octet-stream",
                sha256=sha256(content).hexdigest(),
                size=len(content),
            )
            for path, content in self._outputs.items()
        ]
        self._sessions: dict[str, _Session] = {}

    def provision(self, request: TaskRequest) -> str:
        session_id = str(request.job_id)
        session = self._sessions.get(session_id)
        if session is None:
            self._sessions[session_id] = _Session(request.model_copy(deep=True))
        elif session.request != request:
            raise ValueError("job ID already belongs to a different request")
        return session_id

    def upload(self, session_id: str, files: dict[str, bytes]) -> None:
        session = self._active_session(session_id)
        if session.result is not None:
            raise RuntimeError("cannot upload after execution")
        if set(files) != set(session.request.inputs):
            raise ValueError("uploaded paths must match the job inputs")
        session.inputs = files.copy()
        session.status = JobStatus.UPLOADING

    def execute(self, session_id: str, request: TaskRequest) -> JobResult:
        session = self._active_session(session_id)
        if request != session.request:
            raise ValueError("request does not match the provisioned job")
        if set(session.inputs) != set(request.inputs):
            raise RuntimeError("job inputs have not been uploaded")
        if session.result is None:
            session.status = JobStatus.COMPLETED
            session.result = JobResult(
                job_id=request.job_id, status=session.status, artifacts=self._artifacts
            )
        return session.result.model_copy(deep=True)

    def download(self, session_id: str) -> dict[str, bytes]:
        session = self._active_session(session_id)
        if session.result is None:
            raise RuntimeError("job has not executed")
        return self._outputs.copy()

    def status(self, session_id: str) -> JobStatus:
        return self._sessions[session_id].status

    def stop(self, session_id: str) -> CleanupStatus:
        session = self._sessions[session_id]
        session.stopped = True
        if session.result is None:
            session.status = JobStatus.CANCELLED
        else:
            session.result.cleanup_status = CleanupStatus.CONFIRMED
        return CleanupStatus.CONFIRMED

    def _active_session(self, session_id: str) -> _Session:
        session = self._sessions[session_id]
        if session.stopped:
            raise RuntimeError("session has already stopped")
        return session
