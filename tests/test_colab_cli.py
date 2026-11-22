import re
import subprocess
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest

from gpu_ai.backends.colab_cli import (
    ColabCLI,
    ColabCLIBackend,
    ColabCLIError,
    ErrorCode,
    WorkerScript,
)
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.doctor import DoctorStatus, diagnose

NO_SESSIONS = "[colab] No active sessions found on server.\n"


@pytest.fixture
def backend(tmp_path):
    scripts = {
        Operation.TRANSCRIBE_AUDIO: WorkerScript(
            "from pathlib import Path\n"
            "Path('output.txt').write_text(Path('media/sample.wav').read_text())\n",
            ("output.txt",),
        )
    }
    return ColabCLIBackend(scripts, settings=Settings(workspace=tmp_path))


def test_lifecycle_uses_dedicated_config_and_session(monkeypatch, tmp_path, backend):
    user_config = tmp_path / ".config" / "colab-cli" / "sessions.json"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('{"existing": {}}')
    calls = []
    configs = set()
    work = tmp_path / "remote-content"
    work.mkdir()

    def run(args, **options):
        assert args[:3] == ["colab", "--auth=oauth2", "--config"]
        assert options["shell"] is False
        assert options["stdin"] == subprocess.DEVNULL
        assert options["capture_output"]
        assert options["timeout"] > 0
        config = Path(args[3])
        assert config != user_config
        assert config.is_absolute()
        configs.add(config)
        command, *flags = args[4:]
        calls.append(command)
        if command != "sessions":
            name = flags[1]
            assert flags[:2] == ["-s", name]
            assert re.fullmatch(r"gpu-ai-[a-f0-9]{32}", name)
        match command:
            case "sessions":
                output = NO_SESSIONS
            case "new":
                assert flags[2:] == ["--gpu", "T4"]
                output = "[colab] Session READY.\n"
            case "upload":
                assert flags[3] == "content/gpu-ai-inputs.zip"
                payload = Path(flags[2]).read_bytes()
                (work / "gpu-ai-inputs.zip").write_bytes(payload)
                with ZipFile(flags[2]) as archive:
                    assert archive.read("media/sample.wav") == b"sample"
                    assert (
                        TaskRequest.model_validate_json(
                            archive.read("gpu-ai-request.json")
                        )
                        == request
                    )
                output = "uploaded\n"
            case "exec":
                assert flags[2] == "-f"
                assert flags[4:] == ["--timeout", "1800"]
                script = Path(flags[3]).read_text()
                # Simulate the Colab execution directory without touching /content.
                monkeypatch.chdir(work)
                marker = []
                exec(  # noqa: S102 - runs only the trusted fixture worker.
                    script.replace("/content/", str(work) + "/"),
                    {"print": marker.append},
                )
                output = marker[0] + "\n"
            case "download":
                assert flags[2] == "content/gpu-ai-outputs.zip"
                Path(flags[3]).write_bytes((work / "gpu-ai-outputs.zip").read_bytes())
                output = "downloaded\n"
            case "status":
                output = (
                    f"[{name}] endpoint | Hardware: T4 | Variant: GPU | Status: IDLE\n"
                )
            case "stop":
                output = "[colab] Session terminated.\n"
            case _:
                pytest.fail(command)
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    request = TaskRequest(
        operation=Operation.TRANSCRIBE_AUDIO, inputs=["media/sample.wav"]
    )
    session = backend.provision(request)
    assert backend.provision(request) == session
    with pytest.raises(KeyError):
        backend.stop("existing")
    backend.upload(session, {"media/sample.wav": b"sample"})
    result = backend.execute(session, request)
    assert backend.download(session) == {"output.txt": b"sample"}
    assert result.artifacts[0].sha256 == sha256(b"sample").hexdigest()
    assert backend.status(session) is JobStatus.COMPLETED
    assert backend.stop(session) is CleanupStatus.CONFIRMED
    assert backend.stop(session) is CleanupStatus.CONFIRMED
    assert len(configs) == 1
    assert calls == ["sessions", "new", "upload", "exec", "download", "status", "stop"]
    assert user_config.read_text() == '{"existing": {}}'


@pytest.mark.parametrize(
    ("version", "sessions", "expected"),
    [
        (None, None, DoctorStatus.NOT_INSTALLED),
        ("0.8.0", NO_SESSIONS, DoctorStatus.AUTHENTICATED),
        (
            "0.8.0",
            "To authorize colab-cli, visit this URL",
            DoctorStatus.UNAUTHENTICATED,
        ),
        ("0.8.0", "unknown output", DoctorStatus.CHECK_FAILED),
    ],
)
def test_doctor(monkeypatch, version, sessions, expected):
    def run(args, **options):
        if version is None:
            raise FileNotFoundError("private path")
        stdout = f"Version: {version}\n" if args[4] == "version" else sessions
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", run)
    assert diagnose().status == expected


@pytest.mark.parametrize(
    ("command", "stdout", "stderr", "code"),
    [
        (
            "sessions",
            NO_SESSIONS,
            "Enter the authorization code: ",
            ErrorCode.AUTH_REQUIRED,
        ),
        (
            "new",
            "",
            "Allocation refused (precondition failed).",
            ErrorCode.ALLOCATION_FAILED,
        ),
        ("upload", "429 Client Error /private/path", "", ErrorCode.RATE_LIMITED),
        ("download", "", "colab-runtime-proxy-token=secret", ErrorCode.COMMAND_FAILED),
        ("exec", "", "TimeoutError: kernel timed out", ErrorCode.TIMEOUT),
    ],
)
def test_errors_keep_diagnostics_private(
    monkeypatch, tmp_path, command, stdout, stderr, code
):
    exit_code = 0 if command == "sessions" else 1
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, exit_code, stdout, stderr),
    )
    with pytest.raises(ColabCLIError) as info:
        ColabCLI()._run(command, config=tmp_path / "sessions.json")
    assert info.value.returncode == exit_code
    assert info.value.code is code
    assert str(info.value) == f"Colab CLI {command}: {code}"


def test_missing_auth_never_allocates(monkeypatch, backend):
    def run(args, **options):
        assert args[4] == "sessions"
        return subprocess.CompletedProcess(args, 1, "", "Enter the authorization code:")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ColabCLIError) as info:
        backend.provision(TaskRequest(operation=Operation.TRANSCRIBE_AUDIO))
    assert info.value.code is ErrorCode.AUTH_REQUIRED


def test_timeout_preserves_unknown_cleanup(monkeypatch, backend):
    def run(args, **options):
        match args[4]:
            case "sessions":
                return subprocess.CompletedProcess(args, 0, NO_SESSIONS, "")
            case "new":
                raise subprocess.TimeoutExpired(args, options["timeout"])
            case "stop":
                return subprocess.CompletedProcess(
                    args, 0, f"[colab] Session '{args[6]}' not found.", ""
                )
            case _:
                pytest.fail(args[4])

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ColabCLIError) as info:
        backend.provision(TaskRequest(operation=Operation.TRANSCRIBE_AUDIO))
    assert info.value.code is ErrorCode.TIMEOUT
    assert info.value.session_id
    assert backend.stop(info.value.session_id) is CleanupStatus.UNKNOWN


@pytest.mark.parametrize(
    "path", ["../escape", "gpu-ai-request.json", "dir//file", "dir/./file"]
)
def test_invalid_paths_fail_before_cli(monkeypatch, backend, path):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("unexpected subprocess")
    )
    request = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO)
    request.inputs = [path]
    with pytest.raises(ValueError):
        backend.provision(request)


def test_exec_exit_zero_without_completion_is_failure(monkeypatch, backend):
    def run(args, **options):
        output = {
            "sessions": NO_SESSIONS,
            "new": "[colab] Session READY.",
            "upload": "uploaded",
            "exec": "worker traceback containing private data",
        }[args[4]]
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    request = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO)
    session = backend.provision(request)
    backend.upload(session, {})
    with pytest.raises(ColabCLIError, match="^Colab CLI exec: command_failed$"):
        backend.execute(session, request)
    with pytest.raises(RuntimeError, match="job has not completed"):
        backend.download(session)


def test_stop_retries_after_nonzero_exit(monkeypatch, backend):
    def run(args, **options):
        match args[4]:
            case "sessions":
                return subprocess.CompletedProcess(args, 0, NO_SESSIONS, "")
            case "new":
                return subprocess.CompletedProcess(
                    args, 0, "[colab] Session READY.", ""
                )
            case "stop":
                return next(stop_results)
            case _:
                pytest.fail(args[4])

    stop_results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "429"),
            subprocess.CompletedProcess([], 0, "[colab] Session terminated.", ""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    session = backend.provision(TaskRequest(operation=Operation.TRANSCRIBE_AUDIO))
    assert backend.stop(session) is CleanupStatus.FAILED
    assert backend.stop(session) is CleanupStatus.CONFIRMED
    assert backend.status(session) is JobStatus.FAILED
