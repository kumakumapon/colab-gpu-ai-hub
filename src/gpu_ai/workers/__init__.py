"""Registered GPU inference workers."""

from gpu_ai.backends.colab_cli import WorkerScript
from gpu_ai.contracts import Operation

from . import whisper
from ._scripts import build_script


def colab_worker_scripts() -> dict[Operation, WorkerScript]:
    """Build the application-owned workers accepted by ColabCLIBackend."""
    return {
        Operation.TRANSCRIBE_AUDIO: build_script(
            "whisper",
            whisper.ARTIFACTS,
            whisper.validate_request,
            cuda_libraries=("nvidia-cublas-cu12", "nvidia-cudnn-cu12"),
        ),
    }
