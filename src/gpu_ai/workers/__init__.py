"""Registered GPU inference workers."""

from gpu_ai.backends.colab_cli import WorkerScript
from gpu_ai.contracts import Operation

from . import depth, llm, ocr, whisper
from ._scripts import build_script


def colab_worker_scripts() -> dict[Operation, WorkerScript]:
    """Build the application-owned workers accepted by ColabCLIBackend."""
    return {
        Operation.EXTRACT_TEXT: build_script(
            "ocr",
            (),
            ocr.validate_request,
            artifacts_for=ocr.output_paths,
        ),
        Operation.LLM_INFER: build_script("llm", llm.ARTIFACTS, llm.validate_request),
        Operation.ESTIMATE_DEPTH: build_script(
            "depth",
            (),
            depth.validate_request,
            artifacts_for=depth.output_paths,
        ),
        Operation.TRANSCRIBE_AUDIO: build_script(
            "whisper",
            whisper.ARTIFACTS,
            whisper.validate_request,
            cuda_libraries=("nvidia-cublas-cu12", "nvidia-cudnn-cu12"),
        ),
    }
