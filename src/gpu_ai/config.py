"""Load application settings from TOML and GPU_AI_* environment variables."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, field_validator

from gpu_ai.contracts import GPU, Contract


class Settings(Contract):
    workspace: Path = Path("workspace")
    results_dir: Path = Path("results")
    gpu: GPU = GPU.T4
    max_concurrent_jobs: int = Field(default=1, ge=1, le=1)
    max_input_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_artifact_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    max_media_duration_seconds: float = Field(default=300, gt=0)
    max_runtime_seconds: int = Field(default=1800, gt=0)

    @field_validator("workspace", "results_dir", mode="before")
    @classmethod
    def validate_directory(cls, value: object) -> object:
        if isinstance(value, str) and (not value.strip() or "\x00" in value):
            raise ValueError("expected a non-empty directory path")
        return value


def load_settings(
    path: Path | None = None, *, environ: Mapping[str, str] | None = None
) -> Settings:
    values = {}
    if path is not None:
        with path.open("rb") as config_file:
            values.update(tomllib.load(config_file))

    environment = os.environ if environ is None else environ
    for field in Settings.model_fields:
        key = f"GPU_AI_{field.upper()}"
        if key in environment:
            values[field] = environment[key]
    return Settings.model_validate(values)
