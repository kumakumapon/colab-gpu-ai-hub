"""Validate transferred artifacts and publish immutable per-job results."""

import fcntl
import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from shutil import rmtree
from struct import Struct
from tempfile import mkdtemp
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from gpu_ai.contracts import Artifact, JobResult
from gpu_ai.media.validate import MediaInfo, validate_transfer_path

TRANSFER_MANIFEST = "gpu-ai-transfer.json"
RESULT_MANIFEST = "gpu-ai-manifest.json"
MAX_ARCHIVE_FILES = 1_024
MAX_MANIFEST_BYTES = 1_048_576
_END_OF_DIRECTORY = Struct("<4s4H2LH")


def verify_artifacts(result: JobResult, outputs: dict[str, bytes]) -> None:
    declared = {artifact.path: artifact for artifact in result.artifacts}
    if len(declared) != len(result.artifacts) or declared.keys() != outputs.keys():
        raise ValueError("downloaded artifacts do not match the manifest")
    for path, content in outputs.items():
        validate_transfer_path(path)
        artifact = declared[path]
        if (
            len(content) != artifact.size
            or sha256(content).hexdigest() != artifact.sha256
        ):
            raise ValueError("downloaded artifact failed integrity verification")


def read_transfer_archive(
    path: Path, expected: tuple[str, ...], *, max_bytes: int
) -> tuple[list[Artifact], dict[str, bytes]]:
    expected_names = {*expected, TRANSFER_MANIFEST}
    if len(expected_names) != len(expected) + 1 or len(expected) > MAX_ARCHIVE_FILES:
        raise ValueError("output archive exceeds configured limits")
    with path.open("rb") as source:
        size = source.seek(0, os.SEEK_END)
        if not _END_OF_DIRECTORY.size <= size <= max_bytes + 4 * MAX_MANIFEST_BYTES:
            raise ValueError("output archive exceeds configured limits")
        source.seek(-_END_OF_DIRECTORY.size, os.SEEK_END)
        marker, disk, directory_disk, disk_count, count, length, offset, comment = (
            _END_OF_DIRECTORY.unpack(source.read(_END_OF_DIRECTORY.size))
        )
        if (
            marker != b"PK\x05\x06"
            or disk
            or directory_disk
            or comment
            or count != disk_count
            or count != len(expected_names)
            or length > 2 * MAX_MANIFEST_BYTES
            or offset + length + _END_OF_DIRECTORY.size != size
        ):
            raise ValueError("unexpected or oversized archive directory")
        with ZipFile(source) as archive:
            return _read_entries(archive, expected, expected_names, max_bytes)


def _read_entries(
    archive: ZipFile,
    expected: tuple[str, ...],
    expected_names: set[str],
    max_bytes: int,
) -> tuple[list[Artifact], dict[str, bytes]]:
    entries = archive.infolist()
    if (
        len(entries) != len(expected_names)
        or {entry.filename for entry in entries} != expected_names
    ):
        raise ValueError("unexpected or duplicate archive entries")
    total_size = 0
    for entry in entries:
        file_type = stat.S_IFMT(entry.external_attr >> 16)
        if (
            entry.is_dir()
            or file_type not in {0, stat.S_IFREG}
            or entry.orig_filename != entry.filename
            or entry.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
            or entry.flag_bits & 1
        ):
            raise ValueError("archive entries must be regular files")
        if entry.filename != TRANSFER_MANIFEST:
            validate_transfer_path(entry.filename)
            total_size += entry.file_size
    manifest_entry = archive.getinfo(TRANSFER_MANIFEST)
    if total_size > max_bytes or manifest_entry.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("archive expands past configured limits")
    with archive.open(manifest_entry) as manifest_file:
        manifest = json.loads(manifest_file.read(MAX_MANIFEST_BYTES + 1))
    if not isinstance(manifest, list) or len(manifest) != len(expected):
        raise ValueError("invalid transfer manifest")
    artifacts = [Artifact.model_validate(entry) for entry in manifest]
    declared = {entry.path: entry for entry in artifacts}
    if declared.keys() != set(expected):
        raise ValueError("transfer manifest does not match expected artifacts")

    outputs = {}
    for name in expected:
        entry = archive.getinfo(name)
        artifact = declared[name]
        if entry.file_size != artifact.size:
            raise ValueError("archive entry size does not match its manifest")
        with archive.open(entry) as file:
            content = file.read(artifact.size + 1)
        if (
            len(content) != artifact.size
            or sha256(content).hexdigest() != artifact.sha256
        ):
            raise ValueError("archive entry failed integrity verification")
        outputs[name] = content
    return artifacts, outputs


class ArtifactStore:
    def __init__(self, directory: Path, *, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = max_bytes

    def save(
        self,
        result: JobResult,
        outputs: dict[str, bytes],
        input_hashes: dict[str, str],
        files: dict[str, bytes],
        media: dict[str, MediaInfo],
    ) -> Path:
        if sum(map(len, outputs.values())) > self.max_bytes:
            raise ValueError("artifact size exceeds max_artifact_bytes")
        verify_artifacts(result, outputs)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.directory / str(result.job_id)
        lock_fd = os.open(
            self.directory / ".store.lock",
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(lock_fd, "wb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if os.path.lexists(destination):
                raise FileExistsError("job output directory already exists")
            stage = Path(mkdtemp(prefix=".gpu-ai-results-", dir=self.directory))
            try:
                for relative, content in outputs.items():
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as file:
                        file.write(content)

                manifest = {
                    "schema_version": "1",
                    "job_id": str(result.job_id),
                    "model_id": result.metrics.get("model_id"),
                    "processing_time_seconds": result.metrics.get(
                        "processing_time_seconds"
                    ),
                    "inputs": [
                        {
                            "path": path,
                            "size": len(files[path]),
                            "sha256": digest,
                            "mime_type": info.mime_type
                            if (info := media.get(path))
                            else "application/octet-stream",
                            "duration_seconds": info.duration_seconds if info else None,
                        }
                        for path, digest in input_hashes.items()
                    ],
                    "artifacts": [
                        artifact.model_dump(mode="json")
                        for artifact in result.artifacts
                    ],
                }
                (stage / RESULT_MANIFEST).write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                stage.rename(destination)
            finally:
                if stage.exists():
                    rmtree(stage)
        return destination
