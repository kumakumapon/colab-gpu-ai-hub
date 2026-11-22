import json
import sys
from hashlib import sha256
from types import SimpleNamespace

import pypdfium2
import pytest
from PIL import Image, ImageDraw

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, ocr
from gpu_ai.workers._runtime import WorkerError


def write_pdf(path, count=2, size=(115, 86)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, 70, 20), outline="black")
    draw.line((32, 1, 32, 20), fill="black")
    draw.text((5, 4), "Test", fill="black")
    image.save(path, save_all=True, append_images=[image] * (count - 1))


@pytest.fixture
def fake_ocr(monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, "ensure_packages", lambda pins, **kwargs: pins.copy())
    monkeypatch.setattr(ocr.platform, "machine", lambda: "test")

    class OCR:
        def __init__(self, **options):
            calls.append(options)
            assert options["text_detection_model_name"] == ocr.DETECTOR

        def predict(self, *, input):
            with Image.open(input) as image:
                width, height = image.size
            return iter(
                [
                    SimpleNamespace(
                        json={
                            "res": {
                                "rec_texts": [
                                    "日本語",
                                    "English",
                                    "<script>table</script>",
                                ],
                                "rec_scores": [0.99, 0.79, 0.89],
                                "rec_polys": [
                                    [[-1, 1], [15, 1], [15, 5], [-1, 5]],
                                    [[20, 6], [30, 6], [30, 10], [20, 10]],
                                    [
                                        [20, 10],
                                        [width + 1, 10],
                                        [width + 1, height],
                                        [20, height],
                                    ],
                                ],
                            }
                        }
                    )
                ]
            )

    paddle = SimpleNamespace(
        __version__=ocr.PADDLE_VERSION,
        is_compiled_with_cuda=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=OCR))
    return calls


@pytest.mark.parametrize("filename", ["page.png", "page.jpg", "pages.pdf"])
def test_ocr_documents_through_colab(tmp_path, simulated_colab, fake_ocr, filename):
    workspace = tmp_path / "input"
    workspace.mkdir()
    if filename.endswith(".pdf"):
        write_pdf(workspace / filename)
    else:
        Image.new("RGB", (80, 40), "white").save(workspace / filename)
    calls = simulated_colab(ocr.run, "ocr")
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    result = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    ).run(
        TaskRequest(
            operation=Operation.EXTRACT_TEXT,
            inputs=[filename],
            parameters={"allow_cpu_fallback": True, "markdown": True},
        )
    )
    assert result.status == JobStatus.COMPLETED, result
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert fake_ocr[0]["device"] == "cpu"
    assert fake_ocr[0]["text_recognition_model_name"] == ocr.RECOGNIZERS["ja"]
    assert result.metrics["cpu_fallback"]
    assert result.metrics["fallback_reason"] == "cuda_unavailable"
    folder = settings.results_dir / str(result.job_id)
    data = json.loads((folder / "ocr.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert len(data["pages"]) == (2 if filename.endswith(".pdf") else 1)
    for page_number, page in enumerate(data["pages"], 1):
        assert page["page"] == page_number
        assert page["lines"][0]["bbox"][0] == [0, 1]
        assert page["lines"][2]["bbox"][2] == [page["width"] - 1, page["height"] - 1]
        assert [line["reading_order"] for line in page["lines"]] == [1, 2, 3]
        assert [line["text"] for line in page["lines"]][:2] == ["日本語", "English"]
    assert "\n    <script>table</script>" in (folder / "ocr.md").read_text(
        encoding="utf-8"
    )


def test_cpu_fallback_requires_permission(tmp_path, simulated_colab, fake_ocr):
    calls = simulated_colab(ocr.run, "ocr")
    path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(path)
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    result = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    ).run(TaskRequest(operation=Operation.EXTRACT_TEXT, inputs=[path.name]))
    assert result.error == "execute:cuda_unavailable"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert not fake_ocr
    assert calls[-1] == "stop"


@pytest.mark.parametrize("variant", ["damaged", "pages", "pixels", "size", "encrypted"])
def test_invalid_pdf_is_rejected(tmp_path, monkeypatch, variant):
    source = tmp_path / "input.pdf"
    match variant:
        case "damaged":
            source.write_bytes(b"%PDF-1.4\ninvalid internal data\n%%EOF")
        case "pages":
            write_pdf(source, count=ocr.MAX_PAGES + 1)
        case "pixels":
            write_pdf(source, count=1, size=(2050, 5))
        case "size":
            write_pdf(source)
            monkeypatch.setattr(ocr, "MAX_PDF_BYTES", 100)
        case "encrypted":
            write_pdf(source)
            monkeypatch.setattr(
                pypdfium2.raw, "FPDF_GetSecurityHandlerRevision", lambda _: 6
            )
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        ocr.rasterize_pdf(source, tmp_path, 144, pypdfium2)


@pytest.mark.parametrize(
    "inputs,parameters",
    [
        ([], {}),
        (["input.wav"], {}),
        (["input.pdf"], {"dpi": 300}),
        (["input.jpg"], {"language": "it"}),
        (["input.png"], {"allow_cpu_fallback": 1}),
    ],
)
def test_input_options(inputs, parameters):
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        ocr.options(inputs, parameters)


def test_paddle_wheel_requires_exact_ranges_and_digest(tmp_path, monkeypatch):
    from contextlib import nullcontext

    payload = b"official wheel fixture"
    chunk = 7
    monkeypatch.setattr(ocr, "COLAB_WHEEL_SIZE", len(payload))
    monkeypatch.setattr(ocr, "WHEEL_CHUNK_SIZE", chunk)
    monkeypatch.setattr(ocr, "COLAB_WHEEL_SHA256", sha256(payload).hexdigest())
    requests = []

    def read_range(request, *, timeout):
        interval = request.get_header("Range").removeprefix("bytes=")
        start, end = map(int, interval.split("-"))
        requests.append(start)
        assert request.full_url == ocr.COLAB_WHEEL_URL and timeout == 25
        return nullcontext(
            SimpleNamespace(
                status=206,
                headers={"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
                read=lambda size: payload[start : end + 1][:size],
            )
        )

    monkeypatch.setattr(ocr, "urlopen", read_range)
    target = tmp_path / "paddle.whl"
    ocr._download_wheel(target)
    assert target.read_bytes() == payload
    assert sorted(requests) == list(range(0, len(payload), chunk))

    monkeypatch.setattr(ocr, "COLAB_WHEEL_SHA256", "bad-digest")
    with pytest.raises(ValueError, match="digest mismatch"):
        ocr._download_wheel(tmp_path / "tampered.whl")


def test_paddle_wheel_rejects_mismatched_range(monkeypatch):
    from contextlib import nullcontext

    calls = []

    def read_range(request, *, timeout):
        calls.append(request.get_header("Range"))
        return nullcontext(
            SimpleNamespace(status=206, headers={"Content-Range": "bytes 5-9/10"})
        )

    monkeypatch.setattr(ocr, "COLAB_WHEEL_SIZE", 10)
    monkeypatch.setattr(ocr, "WHEEL_CHUNK_SIZE", 5)
    monkeypatch.setattr(ocr, "urlopen", read_range)
    with pytest.raises(OSError, match="range could not be retrieved"):
        ocr._wheel_chunk(0)
    assert calls == ["bytes=0-4"] * 3


def test_colab_installs_verified_gpu_wheel_in_isolated_environment(monkeypatch):
    from pathlib import Path

    calls = []

    def download(wheel):
        assert wheel.name == ocr.COLAB_WHEEL
        wheel.write_bytes(b"verified")
        calls.append("download")

    def run(args, **kwargs):
        assert kwargs["check"]
        if args[3] == "venv":
            calls.append("venv")
        elif args[3:6] == ["pip", "install", "--python"]:
            assert args[6].endswith("/venv/bin/python")
            assert args[7].endswith(ocr.COLAB_WHEEL)
            assert "paddleocr==3.6.0" in args[8:]
            calls.append("install")
        else:
            assert args[0].endswith("/venv/bin/python") and args[1] == "-c"
            assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
            assert len(kwargs["env"]["PYTHONPATH"].split(":")) == 1
            Path(args[4]).write_text('{"metrics":{"device":"cuda"}}')
            calls.append("recognize")

    monkeypatch.setattr(ocr, "ensure_packages", lambda pins: calls.append(pins))
    monkeypatch.setattr(ocr, "_download_wheel", download)
    monkeypatch.setattr(ocr.subprocess, "run", run)
    assert ocr._run_isolated() == {"device": "cuda"}
    assert calls == [{"uv": ocr.UV_VERSION}, "venv", "download", "install", "recognize"]
