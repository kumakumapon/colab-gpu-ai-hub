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
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, depth, llm, whisper


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


# Entry points discover supported operations from this registry.
OPERATIONS = {
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
