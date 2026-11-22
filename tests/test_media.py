import io
import json
import subprocess
import wave
from hashlib import sha256

import pytest

from gpu_ai.artifacts import RESULT_MANIFEST
from gpu_ai.backends.base import MockBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import JobStatus, Operation, TaskRequest
from gpu_ai.media.preprocess import preprocess_video
from gpu_ai.media.validate import MediaError, load_media, read_file
from gpu_ai.runner import JobRunner


def create_wav(path):
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * 3200)
    return path.read_bytes()


def test_validated_input_and_artifact_manifest(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = create_wav(workspace / "sample.wav")
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    request = TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["sample.wav"])

    class AudioBackend(MockBackend):
        def upload(self, session_id, files):
            assert files == {"sample.wav": data}
            super().upload(session_id, files)

        def execute(self, session_id, submitted):
            result = super().execute(session_id, submitted)
            result.metrics["model_id"] = "test-model"
            return result

    result = JobRunner(AudioBackend({"transcript.json": b"{}"}), settings).run(request)
    assert result.status == JobStatus.COMPLETED
    result_path = settings.results_dir / str(request.job_id)
    manifest = json.loads((result_path / RESULT_MANIFEST).read_text())
    assert manifest["inputs"] == [
        {
            "path": "sample.wav",
            "sha256": sha256(data).hexdigest(),
            "size": len(data),
            "mime_type": "audio/wav",
            "duration_seconds": pytest.approx(0.2),
        }
    ]
    assert manifest["artifacts"][0]["sha256"] == sha256(b"{}").hexdigest()
    assert manifest["model_id"] == "test-model"
    assert manifest["processing_time_seconds"] >= 0
    assert sorted(path.name for path in result_path.iterdir()) == [
        RESULT_MANIFEST,
        "transcript.json",
    ]


def test_rejects_invalid_media_before_provision(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "wrong.wav").write_bytes(b"not audio")
    short = create_wav(workspace / "short.wav")
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    backend = MockBackend()
    result = JobRunner(backend, settings).run(
        TaskRequest(operation=Operation.TRANSCRIBE_AUDIO, inputs=["wrong.wav"])
    )
    assert result.error == "input_validation_failed"
    assert backend._sessions == {}
    with pytest.raises(MediaError, match="size"):
        load_media(
            workspace, ["short.wav"], max_bytes=len(short) - 1, max_duration_seconds=10
        )
    with pytest.raises(MediaError, match="duration"):
        load_media(
            workspace, ["short.wav"], max_bytes=len(short), max_duration_seconds=0.1
        )


@pytest.mark.parametrize(
    "path",
    [
        "../outside.wav",
        "a//b.wav",
        ".colab-cli/sessions.wav",
        "secrets/sample.wav",
        "weights/example.wav",
    ],
)
def test_rejects_unsafe_input_paths(tmp_path, path):
    with pytest.raises(ValueError):
        read_file(tmp_path, path, 1024)


def test_rejects_links(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "outside.wav"
    external.touch()
    (workspace / "link.wav").symlink_to(external)
    (workspace / "outside-directory").symlink_to(tmp_path, target_is_directory=True)
    (workspace / "hardlink.wav").hardlink_to(external)
    for relative in ("link.wav", "outside-directory/outside.wav", "hardlink.wav"):
        with pytest.raises(MediaError):
            load_media(workspace, [relative], max_bytes=1024, max_duration_seconds=1)


@pytest.mark.parametrize("with_audio", [True, False])
def test_preprocess_mp4_with_variable_frame_timestamps(tmp_path, with_audio):
    path = tmp_path / "video.mp4"
    audio_input = (
        ["-f", "lavfi", "-i", "sine=frequency=1000:duration=2"] if with_audio else []
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=2",
            *audio_input,
            "-vf",
            r"select=not(eq(n\,2)+eq(n\,5))",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        timeout=20,
    )
    output = preprocess_video(
        path.name,
        Settings(workspace=tmp_path),
        frame_interval_seconds=0.4,
    )
    assert output.duration_seconds == pytest.approx(2)
    assert [frame.timestamp_seconds for frame in output.frames] == pytest.approx(
        [0, 0.75, 1.5]
    )
    assert all(frame.png.startswith(b"\x89PNG") for frame in output.frames)
    if with_audio:
        with wave.open(io.BytesIO(output.audio_wav)) as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16_000
            assert wav.getnframes() > 30_000
    else:
        assert output.audio_wav is None


def test_corrupt_mp4_is_reported(tmp_path):
    (tmp_path / "broken.mp4").write_bytes(b"\0\0\0\x10ftypisom\0\0\0\0")
    with pytest.raises(MediaError):
        preprocess_video("broken.mp4", Settings(workspace=tmp_path))


@pytest.mark.parametrize(
    ("filename", "source_options", "output_options", "mime"),
    [
        ("frame.png", ["color=size=16x16"], ["-frames:v", "1"], "image/png"),
        ("frame.jpg", ["color=size=16x16"], ["-frames:v", "1"], "image/jpeg"),
        ("frame.jpeg", ["color=size=16x16"], ["-frames:v", "1"], "image/jpeg"),
        ("sound.mp3", ["sine=duration=0.25"], [], "audio/mpeg"),
    ],
)
def test_supported_media_types(
    tmp_path, filename, source_options, output_options, mime
):
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            *source_options,
            *output_options,
            "-threads",
            "1",
            str(tmp_path / filename),
        ],
        check=True,
        timeout=20,
    )
    _, information = load_media(
        tmp_path, [filename], max_bytes=100_000, max_duration_seconds=1
    )
    assert information[filename].mime_type == mime


def test_pdf_requires_content_signature(tmp_path):
    (tmp_path / "document.pdf").write_bytes(b"%PDF-1.4\nstartxref\n0\n%%EOF\n")
    _, information = load_media(
        tmp_path, ["document.pdf"], max_bytes=1000, max_duration_seconds=1
    )
    assert information["document.pdf"].mime_type == "application/pdf"
