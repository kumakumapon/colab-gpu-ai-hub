import json
import sys
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
