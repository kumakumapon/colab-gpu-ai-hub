"""Versioned request and result models shared by callers and backends."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue


class Operation(StrEnum):
    TRANSCRIBE_AUDIO = "transcribe_audio"
    GENERATE_IMAGE = "generate_image"
    EXTRACT_TEXT = "extract_text"
    SEGMENT_OBJECT = "segment_object"
    ESTIMATE_DEPTH = "estimate_depth"
    LLM_INFER = "llm_infer"
    ANALYZE_VIDEO = "analyze_video"


class GPU(StrEnum):
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"


class JobStatus(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    UPLOADING = "uploading"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLEANUP_PENDING = "cleanup_pending"


class CleanupStatus(StrEnum):
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


def validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or "\\" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("expected a workspace-relative POSIX file path")
    return value


RelativePath = Annotated[str, AfterValidator(validate_relative_path)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TaskRequest(Contract):
    operation: Operation
    inputs: list[RelativePath] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    gpu: GPU = GPU.T4
    job_id: UUID = Field(default_factory=uuid4)


class Artifact(Contract):
    path: RelativePath
    mime_type: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)


class JobResult(Contract):
    schema_version: Literal["1"] = "1"
    job_id: UUID
    status: JobStatus
    artifacts: list[Artifact] = Field(default_factory=list)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    error: str | None = None
    cleanup_status: CleanupStatus = CleanupStatus.UNKNOWN
