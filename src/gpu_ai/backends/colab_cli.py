"""Colab CLI backend for trusted, application-registered worker scripts."""

import mimetypes
import re
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory, mkdtemp
from threading import Lock
from time import monotonic
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from gpu_ai.backends.base import job_deadline
from gpu_ai.config import Settings
from gpu_ai.contracts import (
    Artifact,
    CleanupStatus,
    JobResult,
    JobStatus,
    Operation,
    TaskRequest,
    validate_relative_path,
)


class ErrorCode(StrEnum):
    CLI_MISSING = "cli_missing"
    AUTH_REQUIRED = "auth_required"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ALLOCATION_FAILED = "allocation_failed"
    COMMAND_FAILED = "command_failed"
    SESSION_LOST = "session_lost"
    INVALID_RESPONSE = "invalid_response"


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


def _file_path(path: str) -> str:
    validate_relative_path(path)
    if PurePosixPath(path).as_posix() != path or path.startswith("gpu-ai-"):
        raise ValueError("expected a canonical, non-reserved relative file path")
    return path


@dataclass(frozen=True)
class WorkerScript:
    """Python source provided by the application, with declared relative outputs."""

    source: str
    artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        for path in self.artifacts:
            _file_path(path)
        if len(set(self.artifacts)) != len(self.artifacts):
            raise ValueError("duplicate artifact paths")
        compile(self.source, "<colab-worker>", "exec")


@dataclass
class _Session:
    request: TaskRequest
    config: Path
    status: JobStatus = JobStatus.STARTING
    uploaded: bool = False
    cleanup: CleanupStatus = CleanupStatus.UNKNOWN
    result: JobResult | None = None
    outputs: dict[str, bytes] = field(default_factory=dict)


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
        for path in request.inputs:
            _file_path(path)
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
        root.mkdir(parents=True, exist_ok=True)
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
            outputs = self._execute(session_id, script)
        except ColabCLIError:
            session.status = JobStatus.FAILED
            raise
        session.outputs = outputs
        session.status = JobStatus.COMPLETED
        session.result = JobResult(
            job_id=request.job_id,
            status=session.status,
            artifacts=[
                Artifact(
                    path=path,
                    mime_type=mimetypes.guess_type(path)[0]
                    or "application/octet-stream",
                    size=len(content),
                    sha256=sha256(content).hexdigest(),
                )
                for path, content in outputs.items()
            ],
        )
        return session.result.model_copy(deep=True)

    def _execute(self, name: str, script: WorkerScript) -> dict[str, bytes]:
        marker = f"GPU_AI_COMPLETE_{uuid4().hex}"
        timeout = self._settings.max_runtime_seconds
        with TemporaryDirectory() as directory:
            worker = Path(directory) / "worker.py"
            worker.write_text(
                "from zipfile import ZipFile\n"
                "with ZipFile('gpu-ai-inputs.zip') as bundle:\n"
                "    bundle.extractall()\n"
                f"exec(compile({script.source!r}, '<colab-worker>', 'exec'),"
                " {'__name__': '__main__'})\n"
                "with ZipFile('/content/gpu-ai-outputs.zip', 'w') as bundle:\n"
                f"    for path in {script.artifacts!r}:\n"
                "        bundle.write('/content/' + path, path)\n"
                f"print({marker!r})\n",
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
            if marker not in output.splitlines():
                raise ColabCLIError(ErrorCode.COMMAND_FAILED, "exec")

            self._sessions[name].status = JobStatus.DOWNLOADING
            archive = Path(directory) / "outputs.zip"
            self._run(name, "download", "content/gpu-ai-outputs.zip", str(archive))
            try:
                with ZipFile(archive) as bundle:
                    return {path: bundle.read(path) for path in script.artifacts}
            except (OSError, KeyError, BadZipFile):
                raise ColabCLIError(ErrorCode.INVALID_RESPONSE, "download") from None

    def download(self, session_id: str) -> dict[str, bytes]:
        session = self._sessions[session_id]
        if session.result is None:
            raise RuntimeError("job has not completed")
        return session.outputs.copy()

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
        elif session.result is not None:
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
