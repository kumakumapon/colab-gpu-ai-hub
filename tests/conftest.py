import json
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def simulated_colab(monkeypatch, tmp_path):
    """Exercise the actual CLI ZIP and execution wrappers with a fake GPU worker."""
    remote = tmp_path / "remote"
    remote.mkdir()
    subprocess_run = subprocess.run
    calls = []

    def connect(worker, operation):
        def run(argv, **kwargs):
            if argv[0] == "colab":
                command = argv[4]
                calls.append(command)
                match command:
                    case "sessions":
                        output = "[colab] No active sessions found on server."
                    case "new":
                        output = "[colab] Session READY."
                    case "upload":
                        (remote / "gpu-ai-inputs.zip").write_bytes(
                            Path(argv[7]).read_bytes()
                        )
                    case "exec":
                        wrapper = Path(argv[8]).read_text(encoding="utf-8")
                        messages = []
                        with monkeypatch.context() as context:
                            context.chdir(remote)
                            exec(  # noqa: S102 - only trusted repository code.
                                wrapper.replace("/content/", str(remote) + "/"),
                                {"print": messages.append},
                            )
                        output = "\n".join(messages)
                    case "download":
                        path = remote / "gpu-ai-outputs.zip"
                        with ZipFile(path) as archive:
                            assert not archive.testzip()
                        Path(argv[-1]).write_bytes(path.read_bytes())
                    case "stop":
                        output = "[colab] Session terminated."
                    case _:
                        pytest.fail(command)
                return subprocess.CompletedProcess(
                    argv, 0, locals().get("output", ""), ""
                )
            if argv[1:3] == ["-m", "gpu_ai_colab_worker"]:
                from gpu_ai.workers._runtime import WorkerError

                report = Path(argv[3])
                package = report.parent / "gpu_ai_colab_worker"
                assert (package / "_runtime.py").is_file()
                assert (package / f"{operation}.py").is_file()
                try:
                    result = {"metrics": worker()}
                except WorkerError as error:
                    result = {"error": error.code}
                report.write_text(json.dumps(result), encoding="utf-8")
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess_run(argv, **kwargs)

        monkeypatch.setattr(subprocess, "run", run)
        return calls

    return connect
