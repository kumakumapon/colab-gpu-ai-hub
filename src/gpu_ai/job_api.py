"""Shared request validation and job access for the CLI and MCP server."""

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import AfterValidator, Field

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import (
    GPU,
    CleanupStatus,
    Contract,
    JobStatus,
    Operation,
    TaskRequest,
)
from gpu_ai.jobs import JobRecord
from gpu_ai.media.validate import validate_transfer_path
from gpu_ai.pipelines import video
from gpu_ai.runner import JobRunner
from gpu_ai.workers import (
    colab_worker_scripts,
    depth,
    image_gen,
    llm,
    ocr,
    sam2,
    whisper,
)


def _path(value: str) -> str:
    validate_transfer_path(value)
    return value


InputPath = Annotated[str, Field(max_length=1024), AfterValidator(_path)]


class WorkerInput(Contract):
    inputs: list[InputPath] = Field(default_factory=list)
    gpu: GPU | None = None
    job_id: UUID = Field(default_factory=uuid4)


class WhisperInput(WorkerInput):
    inputs: list[InputPath] = Field(min_length=1, max_length=1)
    language: Literal["ja", "auto"] = "ja"
    vad_filter: bool = Field(default=True, strict=True)
    beam_size: int = Field(default=5, ge=1, le=10, strict=True)


class DepthInput(WorkerInput):
    inputs: list[InputPath] = Field(min_length=1, max_length=4)
    input_size: int = Field(default=518, ge=280, le=700, multiple_of=14, strict=True)
    device: Literal["cuda", "cpu"] = "cuda"
    normalization: Literal["imagenet"] = "imagenet"


class LLMInput(WorkerInput):
    inputs: list[InputPath] = Field(default_factory=list, max_length=1)
    prompt: str | None = Field(default=None, min_length=1, max_length=12000)
    model_id: Literal["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"] = (
        llm.DEFAULT_MODEL
    )
    max_new_tokens: int = Field(default=128, ge=1, le=512, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    temperature: float = Field(default=0, ge=0, le=2, strict=True)
    stop: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=4
    )


class OCRInput(WorkerInput):
    inputs: list[InputPath] = Field(min_length=1, max_length=1)
    language: Literal["ja", "en"] = "ja"
    dpi: int = Field(default=144, ge=96, le=200, strict=True)
    markdown: bool = Field(default=False, strict=True)
    device: Literal["cuda", "cpu"] = "cuda"
    allow_cpu_fallback: bool = Field(default=False, strict=True)


class ImageGenInput(WorkerInput):
    inputs: list[InputPath] = Field(default_factory=list, max_length=0)
    prompt: str = Field(min_length=1, max_length=image_gen.MAX_PROMPT_CHARS)
    negative_prompt: str = Field(default="", max_length=image_gen.MAX_PROMPT_CHARS)
    seed: int = Field(default=0, ge=0, le=image_gen.MAX_SEED, strict=True)
    width: int = Field(default=512, ge=256, le=768, multiple_of=64, strict=True)
    height: int = Field(default=512, ge=256, le=768, multiple_of=64, strict=True)
    steps: int = Field(default=20, ge=1, le=40, strict=True)
    guidance_scale: float = Field(default=7.5, ge=1, le=15, strict=True)
    images_per_prompt: int = Field(default=1, ge=1, le=4, strict=True)
    contact_sheet: bool = Field(default=False, strict=True)


PixelCoordinate = Annotated[int, Field(ge=0, le=sam2.MAX_SIDE, strict=True)]


class SamObjectPrompt(Contract):
    object_id: int = Field(ge=1, le=sam2.MAX_OBJECT_ID, strict=True)
    point: list[PixelCoordinate] | None = Field(
        default=None, min_length=2, max_length=2
    )
    box: list[PixelCoordinate] | None = Field(default=None, min_length=4, max_length=4)


class SamInput(WorkerInput):
    inputs: list[InputPath] = Field(min_length=1, max_length=1)
    objects: list[SamObjectPrompt] = Field(min_length=1, max_length=sam2.MAX_OBJECTS)
    start_frame: int | None = Field(default=None, ge=0, lt=sam2.MAX_FRAMES, strict=True)
    frame_count: int | None = Field(default=None, ge=1, le=sam2.MAX_FRAMES, strict=True)
    overlay: bool = Field(default=False, strict=True)


class VideoInput(WorkerInput):
    inputs: list[InputPath] = Field(min_length=1, max_length=1)
    objects: list[SamObjectPrompt] = Field(
        default_factory=list, max_length=sam2.MAX_OBJECTS
    )
    frame_interval_seconds: int = Field(default=5, ge=1, le=60, strict=True)
    max_frames: int = Field(default=12, ge=1, le=24, strict=True)


# Entry points discover supported operations from this registry.
OPERATIONS = {
    Operation.ANALYZE_VIDEO: (
        VideoInput,
        video.validate_request,
        "Analyze video and produce a timestamped report.",
    ),
    Operation.SEGMENT_OBJECT: (
        SamInput,
        sam2.validate_request,
        "Segment prompted objects in an image or short video.",
    ),
    Operation.GENERATE_IMAGE: (
        ImageGenInput,
        image_gen.validate_request,
        "Generate images with Stable Diffusion v1.5.",
    ),
    Operation.EXTRACT_TEXT: (
        OCRInput,
        ocr.validate_request,
        "Recognize text in one image or PDF.",
    ),
    Operation.TRANSCRIBE_AUDIO: (
        WhisperInput,
        whisper.validate_request,
        "Transcribe audio into JSON and SRT.",
    ),
    Operation.ESTIMATE_DEPTH: (
        DepthInput,
        depth.validate_request,
        "Estimate relative depth from images.",
    ),
    Operation.LLM_INFER: (LLMInput, llm.validate_request, "Generate text with Qwen."),
}


def make_request(
    operation: Operation, value: WorkerInput, settings: Settings
) -> TaskRequest:
    _, validate, _ = OPERATIONS[operation]
    request = TaskRequest(
        operation=operation,
        inputs=value.inputs,
        parameters=value.model_dump(
            exclude={"inputs", "gpu", "job_id"}, exclude_none=True
        ),
        gpu=value.gpu or settings.gpu,
        job_id=value.job_id,
    )
    validate(request)
    return request


def make_runner(settings: Settings) -> JobRunner:
    settings = settings.model_copy(
        update={
            "workspace": settings.workspace.absolute(),
            "results_dir": settings.results_dir.absolute(),
        }
    )
    return JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    )


class JobSummary(Contract):
    job_id: UUID
    status: JobStatus
    artifact_paths: list[str] = Field(default_factory=list)
    cleanup_status: CleanupStatus = CleanupStatus.UNKNOWN
    error: str | None = None


def summarize(record: JobRecord, results_dir: Path) -> JobSummary:
    result = record.result
    paths = []
    if result and record.outcome_status == JobStatus.COMPLETED:
        paths = [
            str(results_dir.absolute() / str(record.job_id) / a.path)
            for a in result.artifacts
        ]
    return JobSummary(
        job_id=record.job_id,
        status=record.status,
        artifact_paths=paths,
        cleanup_status=record.cleanup_status,
        error=result.error if result else None,
    )
