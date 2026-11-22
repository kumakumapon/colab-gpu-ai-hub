"""Run one GPU job and recover owned sessions after interruption."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event
from time import monotonic
from uuid import uuid4

from gpu_ai.artifacts import ArtifactStore, verify_artifacts
from gpu_ai.backends.base import ManagedBackend, job_deadline
from gpu_ai.backends.colab_cli import ColabCLIError
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobResult, JobStatus, Operation, TaskRequest
from gpu_ai.jobs import FINAL_STATES, JobBusy, JobLedger, JobRecord
from gpu_ai.media.validate import (
    MediaInfo,
    load_media,
    read_file,
    validate_transfer_path,
)
from gpu_ai.workers.llm import MAX_PROMPT_BYTES, decode_prompt

ResultSink = Callable[[JobResult, dict[str, bytes], dict[str, str]], None]


class _Cancelled(Exception):
    pass


class _DeadlineExceeded(Exception):
    pass


class JobRunner:
    def __init__(self, backend: ManagedBackend, settings: Settings | None = None):
        self.backend = backend
        self.settings = settings or Settings()
        self.ledger = JobLedger(self.settings.workspace)
        self.artifacts = ArtifactStore(
            self.settings.results_dir, max_bytes=self.settings.max_artifact_bytes
        )

    def run(
        self,
        request: TaskRequest,
        files: dict[str, bytes] | None = None,
        *,
        cancel: Event | None = None,
        save_results: ResultSink | None = None,
    ) -> JobResult:
        """Run a request once. An identical job ID returns the recorded result."""
        request = TaskRequest.model_validate(request.model_dump())
        digest = sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
        with self.ledger.exclusive():
            prior = self.ledger.get(request.job_id)
            if prior is not None:
                if (
                    prior.backend != self.backend.recovery_key
                    or prior.request_sha256 != digest
                ):
                    raise ValueError("job ID belongs to a different request or backend")
                if prior.result is None:
                    raise JobBusy("job was interrupted and needs reconciliation")
                return prior.result
            if next(self.ledger.unfinished(), None) is not None:
                raise JobBusy("an earlier job needs reconciliation")

            now = datetime.now(UTC)
            record = JobRecord(
                job_id=request.job_id,
                backend=self.backend.recovery_key,
                request_sha256=digest,
                created_at=now,
                updated_at=now,
            )
            self._advance(record, JobStatus.QUEUED)
            return self._run(record, request, files, cancel, save_results)

    def _run(
        self,
        record: JobRecord,
        request: TaskRequest,
        files: dict[str, bytes] | None,
        cancel: Event | None,
        save_results: ResultSink | None,
    ) -> JobResult:
        started = monotonic()
        deadline = started + self.settings.max_runtime_seconds
        token = job_deadline.set(deadline)
        outcome = JobResult(job_id=request.job_id, status=JobStatus.FAILED)
        stage = "input_validation"

        def checkpoint() -> None:
            if cancel is not None and cancel.is_set():
                raise _Cancelled
            if monotonic() >= deadline:
                raise _DeadlineExceeded

        try:
            checkpoint()
            prompt_path = (
                request.inputs[0]
                if request.operation is Operation.LLM_INFER
                and len(request.inputs) == 1
                and Path(request.inputs[0]).suffix.lower() == ".txt"
                else None
            )
            if files is None and prompt_path is not None:
                limit = min(self.settings.max_input_bytes, MAX_PROMPT_BYTES)
                files = {
                    prompt_path: read_file(self.settings.workspace, prompt_path, limit)
                }
            if files is None:
                files, media = load_media(
                    self.settings.workspace,
                    request.inputs,
                    max_bytes=self.settings.max_input_bytes,
                    max_duration_seconds=self.settings.max_media_duration_seconds,
                )
            else:
                files = self._validate_inputs(request, files)
                media = {}
                if prompt_path is not None:
                    decode_prompt(files[prompt_path])
                    media[prompt_path] = MediaInfo("text/plain")
            record.input_hashes = {
                path: sha256(data).hexdigest() for path, data in files.items()
            }

            checkpoint()
            stage = "provision"
            session = f"gpu-ai-{uuid4().hex}"
            record.session_id = session
            self._advance(record, JobStatus.STARTING)
            if self.backend.provision_owned(request, session) != session:
                raise ValueError("backend returned an unexpected session name")

            checkpoint()
            stage = "upload"
            self._advance(record, JobStatus.UPLOADING)
            self.backend.upload(session, files)

            checkpoint()
            stage = "execute"
            self._advance(record, JobStatus.RUNNING)
            result = self.backend.execute(session, request)
            if result.status != JobStatus.COMPLETED or result.job_id != request.job_id:
                raise ValueError("backend did not complete the requested job")

            checkpoint()
            stage = "download"
            self._advance(record, JobStatus.DOWNLOADING)
            outputs = self.backend.download(session)
            stage = "verify"
            if sum(map(len, outputs.values())) > self.settings.max_artifact_bytes:
                raise ValueError("artifact size exceeds max_artifact_bytes")
            verify_artifacts(result, outputs)
            checkpoint()
            result.metrics.setdefault("processing_time_seconds", monotonic() - started)
            stage = "store"
            if save_results is None:
                self.artifacts.save(result, outputs, record.input_hashes, files, media)
            else:
                save_results(result, outputs, record.input_hashes)
            outcome = result.model_copy(update={"error": None})
            checkpoint()
        except (_Cancelled, KeyboardInterrupt):
            outcome.status = JobStatus.CANCELLED
            outcome.error = "cancelled"
        except _DeadlineExceeded:
            outcome.status = JobStatus.FAILED
            outcome.error = "wall_clock_timeout"
        except Exception as error:  # noqa: BLE001 - backend and sink boundary.
            if cancel is not None and cancel.is_set():
                outcome.status = JobStatus.CANCELLED
                outcome.error = "cancelled"
            elif monotonic() >= deadline:
                outcome.status = JobStatus.FAILED
                outcome.error = "wall_clock_timeout"
            else:
                # Publish only stable, explicitly safe diagnostic codes.
                outcome.error = (
                    f"{stage}:{error.code}"
                    if isinstance(error, ColabCLIError)
                    else f"{stage}_failed"
                )
        finally:
            job_deadline.reset(token)
            self._finish(record, outcome)
        return outcome

    def _validate_inputs(
        self, request: TaskRequest, files: dict[str, bytes]
    ) -> dict[str, bytes]:
        files = dict(files)
        if len(set(request.inputs)) != len(request.inputs) or set(files) != set(
            request.inputs
        ):
            raise ValueError("files must match the unique request inputs")
        for path in files:
            validate_transfer_path(path)
        if any(not isinstance(content, bytes) for content in files.values()):
            raise ValueError("inputs must be bytes")
        if sum(map(len, files.values())) > self.settings.max_input_bytes:
            raise ValueError("input size exceeds max_input_bytes")
        return files

    def _advance(self, record: JobRecord, status: JobStatus) -> None:
        record.advance(status)
        self.ledger.save(record)

    def _finish(self, record: JobRecord, outcome: JobResult) -> None:
        record.result = outcome
        record.outcome_status = outcome.status
        cleanup = CleanupStatus.CONFIRMED
        if record.session_id is not None:
            # Keep the original outcome durable while release is in progress.
            try:
                self._advance(record, JobStatus.STOPPING)
            finally:
                try:
                    cleanup = self.backend.recover_owned(record.session_id)
                except (Exception, KeyboardInterrupt):  # noqa: BLE001 - cleanup boundary.
                    cleanup = CleanupStatus.UNKNOWN
        record.cleanup_status = cleanup
        status = outcome.status
        if cleanup != CleanupStatus.CONFIRMED:
            status = JobStatus.CLEANUP_PENDING
            if outcome.error is None:
                outcome.error = "cleanup_unconfirmed"
        elif outcome.error == "cleanup_unconfirmed":
            outcome.error = None
        outcome.status = status
        outcome.cleanup_status = cleanup
        self._advance(record, status)

    def reconcile(self, *, retry: bool = False) -> list[JobRecord]:
        """List interrupted jobs or retry release of sessions recorded by this backend."""
        with self.ledger.exclusive():
            records = [
                record
                for record in self.ledger.unfinished()
                if record.backend == self.backend.recovery_key
            ]
            if retry:
                for record in records:
                    outcome = record.result
                    if outcome is None:
                        outcome = JobResult(
                            job_id=record.job_id,
                            status=JobStatus.FAILED,
                            error="interrupted",
                        )
                    elif outcome.status not in FINAL_STATES:
                        outcome = outcome.model_copy(
                            update={"status": record.outcome_status or JobStatus.FAILED}
                        )
                    self._finish(record, outcome)
            return records
