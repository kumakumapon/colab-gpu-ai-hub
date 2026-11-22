"""Registered GPU inference workers."""

from pathlib import Path

from gpu_ai.backends.colab_cli import WorkerScript
from gpu_ai.contracts import Operation

from . import depth, image_gen, llm, ocr, sam2, whisper
from ._scripts import build_script


def colab_worker_scripts() -> dict[Operation, WorkerScript]:
    """Build the application-owned workers accepted by ColabCLIBackend."""
    from gpu_ai.pipelines import video

    return {
        Operation.ANALYZE_VIDEO: build_script(
            "video",
            video.ARTIFACTS,
            video.validate_request,
            additional_modules=("whisper", "depth", "sam2", "llm"),
            source_path=Path(video.__file__),
            cuda_libraries=("nvidia-cublas-cu12", "nvidia-cudnn-cu12"),
            hub_http_downloads=True,
        ),
        Operation.SEGMENT_OBJECT: build_script(
            "sam2",
            (),
            sam2.validate_request,
            artifacts_for=sam2.output_paths,
        ),
        Operation.GENERATE_IMAGE: build_script(
            "image_gen",
            (),
            image_gen.validate_request,
            hub_http_downloads=True,
            artifacts_for=image_gen.output_paths,
        ),
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
