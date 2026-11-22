"""Runtime utilities shipped with the trusted worker code."""

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class WorkerError(ValueError):
    """A worker failure with a safe public error code."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def request_fields() -> tuple[list[str], dict]:
    request = json.loads(Path("gpu-ai-request.json").read_text(encoding="utf-8"))
    return request["inputs"], request["parameters"]


def write_json(path: str | Path, data: object) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def check_parameters(parameters: dict, allowed: set[str]) -> None:
    if parameters.keys() - allowed:
        raise WorkerError("invalid_worker_input")


def integer(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WorkerError("invalid_worker_input")
    return value


def ensure_packages(pins: dict[str, str]) -> dict[str, str]:
    needed = []
    for package, expected in pins.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            needed.append(f"{package}=={expected}")
    if needed:
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    *needed,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            raise WorkerError("worker_dependencies") from None
    return pins.copy()
