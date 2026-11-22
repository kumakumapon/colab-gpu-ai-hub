"""Package a worker and its standard-library runtime for Colab."""

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

from gpu_ai.backends.colab_cli import WorkerScript
from gpu_ai.contracts import TaskRequest


def build_script(
    module: str,
    artifacts: tuple[str, ...],
    validate: Callable[[TaskRequest], None],
    *,
    cuda_libraries: tuple[str, ...] = (),
    hub_http_downloads: bool = False,
    artifacts_for: Callable[[TaskRequest], tuple[str, ...]] | None = None,
) -> WorkerScript:
    directory = Path(__file__).parent
    runtime = (directory / "_runtime.py").read_text(encoding="utf-8")
    worker = (directory / f"{module}.py").read_text(encoding="utf-8")
    entry = dedent(f"""
        import sys
        from .{module} import run
        from ._runtime import WorkerError, write_json

        try:
            report = dict(metrics=run())
        except WorkerError as error:
            report = dict(error=error.code)
        write_json(sys.argv[1], report)
    """)
    source = dedent(f"""
        import json
        import os
        import subprocess
        import sys
        from importlib.metadata import PackageNotFoundError, distribution
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix="gpu-ai-worker-") as directory:
            root = Path(directory)
            package = root / "gpu_ai_colab_worker"
            package.mkdir()
            (package / "__init__.py").touch()
            (package / "__main__.py").write_text({entry!r}, encoding="utf-8")
            (package / "_runtime.py").write_text({runtime!r}, encoding="utf-8")
            (package / "{module}.py").write_text({worker!r}, encoding="utf-8")
            environment = os.environ.copy()
            if {hub_http_downloads!r}:
                environment["HF_HUB_DISABLE_XET"] = "1"
                environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "40"
            search_paths = [directory, environment.get("PYTHONPATH", "")]
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, search_paths))
            library_paths = []
            for name in {cuda_libraries!r}:
                try:
                    library = name.split("-")[1]
                    distribution_root = distribution(name)
                except PackageNotFoundError:
                    continue
                lib = distribution_root.locate_file(f"nvidia/{{library}}/lib")
                if lib.is_dir():
                    library_paths.append(str(lib))
            if library_paths:
                library_paths.append(environment.get("LD_LIBRARY_PATH", ""))
                environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, library_paths))
            destination = root / "worker-report.json"
            completed = subprocess.run(
                [sys.executable, "-m", "gpu_ai_colab_worker", str(destination)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode != 0 or not destination.is_file():
                gpu_ai_error = "worker_failed"
            else:
                report = json.loads(destination.read_text(encoding="utf-8"))
                gpu_ai_metrics = report.get("metrics", {{}})
                gpu_ai_error = report.get("error")
    """)
    return WorkerScript(source, artifacts, validate, artifacts_for)
