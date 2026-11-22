"""Colab CLI backend for trusted, application-registered worker scripts."""

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from textwrap import dedent
from threading import Lock
from time import monotonic
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from pydantic import JsonValue

from gpu_ai.artifacts import (
    MAX_ARCHIVE_FILES,
    MAX_MANIFEST_BYTES,
    TRANSFER_MANIFEST,
    read_transfer_archive,
)
from gpu_ai.backends.base import job_deadline
from gpu_ai.config import Settings
from gpu_ai.contracts import (
    Artifact,
    CleanupStatus,
    JobResult,
    JobStatus,
    Operation,
    TaskRequest,
)
from gpu_ai.media.validate import validate_transfer_path


class ErrorCode(StrEnum):
    CLI_MISSING = "cli_missing"
    AUTH_REQUIRED = "auth_required"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ALLOCATION_FAILED = "allocation_failed"
    COMMAND_FAILED = "command_failed"
    SESSION_LOST = "session_lost"
    INVALID_RESPONSE = "invalid_response"
    INVALID_WORKER_INPUT = "invalid_worker_input"
    WORKER_DEPENDENCIES = "worker_dependencies"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    INCOMPATIBLE_CUDA = "incompatible_cuda"
    OUT_OF_MEMORY = "out_of_memory"
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    MODEL_LOAD_FAILED = "model_load_failed"
    WORKER_FAILED = "worker_failed"
    UNSAFE_OUTPUT = "unsafe_output"


WORKER_ERROR_CODES = frozenset(
    {
        ErrorCode.INVALID_WORKER_INPUT,
        ErrorCode.WORKER_DEPENDENCIES,
        ErrorCode.CUDA_UNAVAILABLE,
        ErrorCode.INCOMPATIBLE_CUDA,
        ErrorCode.OUT_OF_MEMORY,
        ErrorCode.MODEL_DOWNLOAD_FAILED,
        ErrorCode.MODEL_LOAD_FAILED,
        ErrorCode.WORKER_FAILED,
        ErrorCode.UNSAFE_OUTPUT,
    }
)


class ColabCLIError(RuntimeError):
    def __init__(self, code: ErrorCode, command: str, returncode: int | None = None):
        self.code = code
        self.command = command
        self.returncode = returncode
        self.session_id: str | None = None
        # The subprocess arguments and streams can contain secrets and paths.
        super().__init__(f"Colab CLI {command}: {code}")


class ColabCLI:
    """Subprocess boundary. Raw output stays inside the adapter."""

    def __init__(self, executable: str = "colab"):
        self.executable = executable

    def _run(self, command: str, *args: str, config: Path, timeout: float = 60) -> str:
        deadline = job_deadline.get()
        if deadline is not None and command != "stop":
            timeout = min(timeout, deadline - monotonic())
            if timeout <= 0:
                raise ColabCLIError(ErrorCode.TIMEOUT, command)
        argv = [
            self.executable,
            "--auth=oauth2",
            "--config",
            str(config),
            command,
            *args,
        ]
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            raise ColabCLIError(ErrorCode.CLI_MISSING, command) from None
        except subprocess.TimeoutExpired:
            raise ColabCLIError(ErrorCode.TIMEOUT, command) from None
        except OSError:
            raise ColabCLIError(ErrorCode.COMMAND_FAILED, command) from None

        # Execution output belongs to the worker. Only inspect it on CLI failure.
        if proc.returncode or command == "sessions":
            diagnostic = proc.stderr
            if command != "exec":
                diagnostic += "\n" + proc.stdout
            code = _classify(diagnostic, command)
            if proc.returncode or code != ErrorCode.COMMAND_FAILED:
                raise ColabCLIError(code, command, proc.returncode)
        return proc.stdout

    def check_auth(self, config: Path) -> None:
        output = self._run("sessions", config=config)
        if not (
            "[colab] No active sessions found on server." in output.splitlines()
            or re.search(
                r"^\[[^\]\n]+\] .+ \| Hardware: .+ \| Variant: ", output, re.MULTILINE
            )
        ):
            raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "sessions")

    def version(self, config: Path) -> str:
        output = self._run("version", config=config)
        match = re.search(
            r"^Version: ([0-9][a-zA-Z0-9.+-]{0,40})$", output, re.MULTILINE
        )
        if match is None:
            raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "version")
        return match[1]


def _classify(output: str, command: str) -> ErrorCode:
    message = output.lower()
    if any(
        term in message
        for term in (
            "to authorize colab-cli",
            "enter the authorization code",
            "invalid_grant",
            "unauthenticated",
            "401 client error",
            "status code: 401",
        )
    ):
        return ErrorCode.AUTH_REQUIRED
    if re.search(r"\b429\b|too many requests|rate.limit|resource_exhausted", message):
        return ErrorCode.RATE_LIMITED
    if command == "new" and any(
        term in message
        for term in (
            "allocation refused",
            "backend rejected accelerator",
            "quota",
            "precondition failed",
            "capacity limit",
        )
    ):
        return ErrorCode.ALLOCATION_FAILED
    if any(
        term in message for term in ("timed out", "timeouterror", "deadline exceeded")
    ):
        return ErrorCode.TIMEOUT
    return ErrorCode.COMMAND_FAILED


@dataclass(frozen=True)
class WorkerScript:
    """Python source provided by the application, with declared relative outputs."""

    source: str
    artifacts: tuple[str, ...]
    validate_request: Callable[[TaskRequest], None] | None = None
    artifacts_for: Callable[[TaskRequest], tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        self._validate_artifacts(self.artifacts)
        compile(self.source, "<colab-worker>", "exec")

    def output_paths(self, request: TaskRequest) -> tuple[str, ...]:
        paths = (
            self.artifacts
            if self.artifacts_for is None
            else self.artifacts_for(request)
        )
        self._validate_artifacts(paths)
        return paths

    @staticmethod
    def _validate_artifacts(paths: tuple[str, ...]) -> None:
        for path in paths:
            validate_transfer_path(path)
        if len(set(paths)) != len(paths):
            raise ValueError("duplicate artifact paths")
        if len(paths) > MAX_ARCHIVE_FILES:
            raise ValueError("too many declared artifacts")


@dataclass
class _Session:
    request: TaskRequest
    config: Path
    status: JobStatus = JobStatus.STARTING
    uploaded: bool = False
    cleanup: CleanupStatus = CleanupStatus.UNKNOWN
    result: JobResult | None = None
    outputs: dict[str, bytes] | None = None


class ColabCLIBackend:
    recovery_key = "colab-cli"

    def __init__(
        self,
        scripts: dict[Operation, WorkerScript],
        *,
        settings: Settings | None = None,
        cli: ColabCLI | None = None,
    ):
        self._scripts = scripts.copy()
        self._settings = settings or Settings()
        self._cli = cli or ColabCLI()
        self._sessions: dict[str, _Session] = {}
        self._provision_lock = Lock()

    def provision(self, request: TaskRequest) -> str:
        with self._provision_lock:
            return self._provision(request)

    def provision_owned(self, request: TaskRequest, session_id: str) -> str:
        self._validate_owned_name(session_id)
        with self._provision_lock:
            return self._provision(request, session_id)

    def _provision(self, request: TaskRequest, owned_name: str | None = None) -> str:
        request = TaskRequest.model_validate(request.model_dump())
        if request.operation not in self._scripts:
            raise LookupError("no Colab worker registered for this operation")
        validator = self._scripts[request.operation].validate_request
        if validator is not None:
            try:
                validator(request)
            except ValueError:
                raise ColabCLIError(ErrorCode.INVALID_WORKER_INPUT, "new") from None
        self._scripts[request.operation].output_paths(request)
        for path in request.inputs:
            validate_transfer_path(path)
        if len(set(request.inputs)) != len(request.inputs):
            raise ValueError("duplicate input paths")
        for name, session in self._sessions.items():
            if session.request.job_id == request.job_id:
                if session.request != request:
                    raise ValueError("job ID already belongs to a different request")
                if owned_name is not None and name != owned_name:
                    raise ValueError("job ID already belongs to a different session")
                if session.status not in (JobStatus.FAILED, JobStatus.CLEANUP_PENDING):
                    return name
                raise RuntimeError(
                    "the previous provisioning attempt for this job failed"
                )
            if session.cleanup != CleanupStatus.CONFIRMED:
                raise RuntimeError(
                    "stop the current Colab session before starting another"
                )

        root = self._settings.workspace.resolve() / ".colab-cli"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink():
            raise ValueError("Colab session state must be stored in a real directory")
        name = owned_name or f"gpu-ai-{uuid4().hex}"
        if owned_name is None:
            directory = Path(mkdtemp(prefix=f"{name}-", dir=root))
        else:
            directory = root / name
            directory.mkdir(mode=0o700)
        config = directory / "sessions.json"
        try:
            self._cli.check_auth(config)
        except ColabCLIError:
            if not any(directory.iterdir()):
                directory.rmdir()
            raise

        session = _Session(request, config)
        self._sessions[name] = session
        try:
            if owned_name is not None:
                (directory / "allocation-intent").touch(exist_ok=False)
            output = self._run(name, "new", "--gpu", request.gpu.value, timeout=120)
            if "[colab] Session READY." not in output.splitlines():
                raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "new")
        except ColabCLIError as error:
            if error.code in (ErrorCode.AUTH_REQUIRED, ErrorCode.ALLOCATION_FAILED):
                session.cleanup = CleanupStatus.CONFIRMED
            else:
                # A timed-out or interrupted allocation may have reached Colab.
                self.stop(name)
            if session.cleanup == CleanupStatus.CONFIRMED:
                session.status = JobStatus.FAILED
            error.session_id = name
            raise
        return name

    @staticmethod
    def _validate_owned_name(session_id: str) -> None:
        if re.fullmatch(r"gpu-ai-[a-f0-9]{32}", session_id) is None:
            raise ValueError("invalid job session name")

    def recover_owned(self, session_id: str) -> CleanupStatus:
        self._validate_owned_name(session_id)
        if session_id in self._sessions:
            return self.stop(session_id)
        root = self._settings.workspace.resolve() / ".colab-cli"
        directory = root / session_id
        intent = directory / "allocation-intent"
        config = directory / "sessions.json"
        if root.is_symlink() or directory.is_symlink() or intent.is_symlink():
            return CleanupStatus.UNKNOWN
        if not intent.is_file():
            return CleanupStatus.CONFIRMED
        if config.is_symlink() or not config.is_file():
            # An interrupted allocation might have reached Colab before saving its ID.
            return CleanupStatus.UNKNOWN
        return self._stop_cli(session_id, config)

    def _run(self, name: str, command: str, *args: str, timeout: float = 60) -> str:
        return self._cli._run(
            command,
            "-s",
            name,
            *args,
            config=self._sessions[name].config,
            timeout=timeout,
        )

    def _active(self, name: str) -> _Session:
        session = self._sessions[name]
        if session.cleanup != CleanupStatus.UNKNOWN or session.status in (
            JobStatus.CLEANUP_PENDING,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
        ):
            raise RuntimeError("Colab session is no longer available for work")
        return session

    def upload(self, session_id: str, files: dict[str, bytes]) -> None:
        session = self._active(session_id)
        if session.result is not None or set(files) != set(session.request.inputs):
            raise ValueError("upload must match the inputs of an unexecuted job")
        if sum(map(len, files.values())) > self._settings.max_input_bytes:
            raise ValueError("input size exceeds max_input_bytes")

        session.uploaded = False
        session.status = JobStatus.UPLOADING
        with TemporaryDirectory() as directory:
            archive = Path(directory) / "inputs.zip"
            with ZipFile(archive, "w") as bundle:
                for path, content in files.items():
                    bundle.writestr(path, content)
                bundle.writestr(
                    "gpu-ai-request.json", session.request.model_dump_json()
                )
            self._run(session_id, "upload", str(archive), "content/gpu-ai-inputs.zip")
        session.uploaded = True

    def execute(self, session_id: str, request: TaskRequest) -> JobResult:
        session = self._active(session_id)
        if session.request != request or not session.uploaded:
            raise ValueError(
                "execute requires the provisioned request and uploaded inputs"
            )
        if session.result is not None:
            return session.result.model_copy(deep=True)

        script = self._scripts[request.operation]
        session.status = JobStatus.RUNNING
        try:
            paths = script.output_paths(request)
            artifacts, metrics = self._execute(session_id, script.source, paths)
        except ColabCLIError:
            session.status = JobStatus.FAILED
            raise
        session.status = JobStatus.DOWNLOADING
        session.result = JobResult(
            job_id=request.job_id,
            status=JobStatus.COMPLETED,
            artifacts=artifacts,
            metrics=metrics,
        )
        return session.result.model_copy(deep=True)

    def _execute(
        self, name: str, source: str, paths: tuple[str, ...]
    ) -> tuple[list[Artifact], dict[str, JsonValue]]:
        prefix = f"GPU_AI_COMPLETE_{uuid4().hex} "
        timeout = self._settings.max_runtime_seconds
        with TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text(
                dedent(f"""
                import json
                import mimetypes
                from hashlib import file_digest
                from pathlib import Path
                from zipfile import ZipFile

                with ZipFile('gpu-ai-inputs.zip') as bundle:
                    bundle.extractall()
                namespace = {{'__name__': '__main__'}}
                exec(compile({source!r}, '<colab-worker>', 'exec'), namespace)
                error = namespace.get('gpu_ai_error')
                if error:
                    print({prefix!r} + json.dumps({{'error': error}}))
                else:
                    root = Path('/content/').resolve()
                    manifest = []
                    total = 0
                    with ZipFile(root / 'gpu-ai-outputs.zip', 'w') as bundle:
                        for path in {paths!r}:
                            file = root / path
                            if file.resolve() != file or not file.is_file():
                                raise ValueError('artifact must be a regular file')
                            metadata = file.stat()
                            total += metadata.st_size
                            if total > {self._settings.max_artifact_bytes} or metadata.st_nlink != 1:
                                raise ValueError('artifact limit exceeded or file is linked')
                            with file.open('rb') as source:
                                digest = file_digest(source, 'sha256').hexdigest()
                            mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
                            manifest.append(dict(path=path, size=metadata.st_size,
                                                 sha256=digest, mime_type=mime))
                            bundle.write(file, path)
                        bundle.writestr({TRANSFER_MANIFEST!r}, json.dumps(manifest))
                    report = json.dumps(dict(artifacts=manifest,
                                             metrics=namespace.get('gpu_ai_metrics', {{}})))
                    if len(report) > {MAX_MANIFEST_BYTES}:
                        raise ValueError('execution report exceeds size limit')
                    print({prefix!r} + report)
                """),
                encoding="utf-8",
            )
            output = self._run(
                name,
                "exec",
                "-f",
                str(worker),
                "--timeout",
                str(timeout),
                timeout=timeout,
            )
        manifests = [
            line[len(prefix) :]
            for line in output.splitlines()
            if line.startswith(prefix)
        ]
        if not manifests:
            raise ColabCLIError(ErrorCode.COMMAND_FAILED, "exec")
        try:
            if len(manifests) != 1 or len(manifests[0]) > MAX_MANIFEST_BYTES:
                raise ValueError("invalid execution manifest")
            report = json.loads(manifests[0])
            if isinstance(report, dict):
                if "error" in report:
                    code = ErrorCode(report["error"])
                    if code not in WORKER_ERROR_CODES:
                        raise ValueError("unknown worker error")
                    raise ColabCLIError(code, "exec")
                manifest = report.get("artifacts")
                metrics = report.get("metrics")
            else:
                manifest, metrics = report, {}
            if (
                not isinstance(metrics, dict)
                or not isinstance(manifest, list)
                or len(manifest) != len(paths)
            ):
                raise ValueError("invalid execution manifest")
            artifacts = [Artifact.model_validate(item) for item in manifest]
            if tuple(item.path for item in artifacts) != paths:
                raise ValueError("unexpected artifacts in execution manifest")
            if sum(item.size for item in artifacts) > self._settings.max_artifact_bytes:
                raise ValueError("output exceeds configured limits")
        except ValueError:
            raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "exec") from None
        return artifacts, metrics

    def download(self, session_id: str) -> dict[str, bytes]:
        session = self._sessions[session_id]
        if session.result is None:
            raise RuntimeError("job has not completed")
        if session.outputs is not None:
            return session.outputs.copy()
        self._active(session_id)
        try:
            with TemporaryDirectory() as directory:
                archive = Path(directory) / "outputs.zip"
                self._run(
                    session_id, "download", "content/gpu-ai-outputs.zip", str(archive)
                )
                expected = tuple(item.path for item in session.result.artifacts)
                artifacts, outputs = read_transfer_archive(
                    archive, expected, max_bytes=self._settings.max_artifact_bytes
                )
                if artifacts != session.result.artifacts:
                    raise ValueError("execution and download manifests differ")
        except ColabCLIError:
            session.status = JobStatus.FAILED
            raise
        except (OSError, ValueError, RuntimeError, BadZipFile):
            session.status = JobStatus.FAILED
            raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "download") from None
        session.outputs = outputs
        session.status = JobStatus.COMPLETED
        return outputs.copy()

    def status(self, session_id: str) -> JobStatus:
        session = self._sessions[session_id]
        if session.cleanup == CleanupStatus.CONFIRMED:
            return session.status
        output = self._run(session_id, "status")
        if f"[colab] Session '{session_id}' not found." in output.splitlines():
            raise ColabCLIError(ErrorCode.SESSION_LOST, "status")
        match = re.search(
            rf"^\[{session_id}\] .+ \| Status: (IDLE|BUSY \(.+\))$",
            output,
            re.MULTILINE,
        )
        if match is None:
            raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "status")
        return JobStatus.RUNNING if match[1].startswith("BUSY") else session.status

    def stop(self, session_id: str) -> CleanupStatus:
        session = self._sessions[session_id]
        if session.cleanup == CleanupStatus.CONFIRMED:
            return session.cleanup
        session.cleanup = self._stop_cli(session_id, session.config)
        if session.cleanup != CleanupStatus.CONFIRMED:
            session.status = JobStatus.CLEANUP_PENDING
        elif session.outputs is not None:
            session.status = JobStatus.COMPLETED
        elif session.status not in (JobStatus.FAILED, JobStatus.CLEANUP_PENDING):
            session.status = JobStatus.CANCELLED
        else:
            session.status = JobStatus.FAILED
        if session.result is not None:
            session.result.cleanup_status = session.cleanup
        return session.cleanup

    def _stop_cli(self, name: str, config: Path) -> CleanupStatus:
        try:
            output = self._cli._run("stop", "-s", name, config=config)
        except ColabCLIError as error:
            if error.code == ErrorCode.TIMEOUT:
                return CleanupStatus.UNKNOWN
            return CleanupStatus.FAILED
        if "[colab] Session terminated." in output.splitlines():
            return CleanupStatus.CONFIRMED
        return CleanupStatus.UNKNOWN
