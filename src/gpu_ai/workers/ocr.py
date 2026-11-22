"""Bounded image and PDF recognition with the PP-OCRv5 mobile models."""

import json
import math
import os
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPException
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.request import Request, urlopen

from ._runtime import (
    WorkerError,
    check_parameters,
    ensure_packages,
    integer,
    request_fields,
    write_json,
)

PINS = {
    "paddleocr": "3.6.0",
    "paddlex": "3.6.0",
    "pypdfium2": "5.15.2",
    "pillow": "13.0.0",
}
PADDLE_VERSION = "3.3.1"
PADDLE_CUDA_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
COLAB_WHEEL = "paddlepaddle_gpu-3.3.1-cp313-cp313-linux_x86_64.whl"
COLAB_WHEEL_URL = (
    "https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/" + COLAB_WHEEL
)
COLAB_WHEEL_SIZE = 2_021_397_999
COLAB_WHEEL_SHA256 = "d3116dac42e3bfa024aed3a4f8344dd46df28695d7b35ce5ffff964a50d21aed"
WHEEL_CHUNK_SIZE = 16 * 1024**2
WHEEL_DOWNLOAD_WORKERS = 8
UV_VERSION = "0.11.21"
DETECTOR = "PP-OCRv5_mobile_det"
RECOGNIZERS = {"ja": "PP-OCRv5_mobile_rec", "en": "en_PP-OCRv5_mobile_rec"}
MAX_PAGES = 8
MAX_SIDE = 4096
MAX_TOTAL_PIXELS = 32_000_000
MAX_PDF_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Options:
    path: str
    language: str
    dpi: int
    markdown: bool
    device: str
    allow_cpu_fallback: bool


@dataclass(frozen=True)
class Raster:
    path: Path
    width: int
    height: int


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(
        parameters, {"language", "dpi", "markdown", "device", "allow_cpu_fallback"}
    )
    if len(inputs) != 1 or Path(inputs[0]).suffix.lower() not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".pdf",
    }:
        raise WorkerError("invalid_worker_input")
    language = parameters.get("language", "ja")
    dpi = integer(parameters.get("dpi", 144), 96, 200)
    markdown = parameters.get("markdown", False)
    device = parameters.get("device", "cuda")
    fallback = parameters.get("allow_cpu_fallback", False)
    if (
        language not in ("ja", "en")
        or device not in ("cuda", "cpu")
        or not isinstance(markdown, bool)
        or not isinstance(fallback, bool)
    ):
        raise WorkerError("invalid_worker_input")
    return Options(inputs[0], language, dpi, markdown, device, fallback)


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def output_paths(request) -> tuple[str, ...]:
    selected = options(request.inputs, request.parameters)
    return ("ocr.json", "ocr.md") if selected.markdown else ("ocr.json",)


def check_dimensions(width: int, height: int, pixels: int = 0) -> int:
    pixels += width * height
    if not (
        0 < width <= MAX_SIDE and 0 < height <= MAX_SIDE and pixels <= MAX_TOTAL_PIXELS
    ):
        raise WorkerError("invalid_worker_input")
    return pixels


def rasterize_pdf(source: Path, directory: Path, dpi: int, pdfium) -> list[Raster]:
    result = []
    scale = dpi / 72
    try:
        if source.stat().st_size > MAX_PDF_BYTES:
            raise WorkerError("invalid_worker_input")
        with pdfium.PdfDocument(source) as document:
            if (
                pdfium.raw.FPDF_GetSecurityHandlerRevision(document.raw) != -1
                or not 1 <= len(document) <= MAX_PAGES
            ):
                raise WorkerError("invalid_worker_input")
            pixels = 0
            for index in range(len(document)):
                page = document[index]
                try:
                    width, height = page.get_size()
                    pixels = check_dimensions(
                        math.ceil(width * scale), math.ceil(height * scale), pixels
                    )
                    bitmap = page.render(scale=scale, draw_annots=False)
                    try:
                        path = directory / f"{index + 1:04d}.png"
                        image = bitmap.to_pil().convert("RGB")
                        image.save(path)
                        result.append(Raster(path, image.width, image.height))
                    finally:
                        bitmap.close()
                finally:
                    page.close()
    except (OSError, OverflowError, ValueError, pdfium.PdfiumError):
        raise WorkerError("invalid_worker_input") from None
    return result


def rasterize(source: Path, directory: Path, dpi: int) -> list[Raster]:
    try:
        import pypdfium2
        from PIL import Image
    except (ImportError, OSError):
        raise WorkerError("worker_dependencies") from None
    if source.suffix.lower() == ".pdf":
        return rasterize_pdf(source, directory, dpi, pypdfium2)
    try:
        with Image.open(source, formats=("PNG", "JPEG")) as image:
            check_dimensions(*image.size)
            path = directory / "0001.png"
            image.convert("RGB").save(path)
            return [Raster(path, image.width, image.height)]
    except (OSError, ValueError, Image.DecompressionBombError):
        raise WorkerError("invalid_worker_input") from None


def recognition_lines(result, width: int, height: int) -> list[dict]:
    try:
        payload = result.json
        result_fields = payload.get("res", payload)
        rows = []
        for text, score, polygon in zip(
            result_fields["rec_texts"],
            result_fields["rec_scores"],
            result_fields["rec_polys"],
            strict=True,
        ):
            if not isinstance(text, str) or len(text) > 2048 or len(rows) >= 4000:
                raise ValueError("invalid recognition text")
            score = float(score)
            points = [(float(x), float(y)) for x, y in polygon]
            if (
                not 0 <= score <= 1
                or len(points) != 4
                or not all(math.isfinite(x) and math.isfinite(y) for x, y in points)
            ):
                raise ValueError("invalid recognition coordinates")
            bbox = [
                [max(0, min(width - 1, round(x))), max(0, min(height - 1, round(y)))]
                for x, y in points
            ]
            if text.strip():
                rows.append(
                    {
                        "reading_order": len(rows) + 1,
                        "text": text,
                        "score": score,
                        "bbox": bbox,
                    }
                )
        return rows
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        raise WorkerError("worker_failed") from None


def model_on(device: str, language: str, constructor):
    return constructor(
        device="gpu:0" if device == "cuda" else "cpu",
        text_detection_model_name=DETECTOR,
        text_recognition_model_name=RECOGNIZERS[language],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        enable_mkldnn=False,
    )


def available_cuda(paddle) -> bool:
    try:
        return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except (RuntimeError, OSError):
        return False


def use_model(settings: Options, paddle, constructor):
    device, fallback = settings.device, None
    if device == "cuda" and not available_cuda(paddle):
        if not settings.allow_cpu_fallback:
            raise WorkerError("cuda_unavailable")
        device, fallback = "cpu", "cuda_unavailable"
    try:
        return model_on(device, settings.language, constructor), device, fallback
    except Exception as error:  # noqa: BLE001 - suppress vendor errors and credentials.
        if device != "cuda" or not settings.allow_cpu_fallback:
            raise paddle_error(error, "model_load_failed") from None
        fallback = paddle_error(error, "model_load_failed").code
        if fallback == "model_load_failed":
            raise WorkerError(fallback) from None
        try:
            return model_on("cpu", settings.language, constructor), "cpu", fallback
        except Exception as cpu_error:  # noqa: BLE001 - vendor boundary.
            raise paddle_error(cpu_error, "model_load_failed") from None


def paddle_error(error: Exception, fallback: str) -> WorkerError:
    message = str(error).lower()
    if "out of memory" in message or "resource exhausted" in message:
        return WorkerError("out_of_memory")
    if any(marker in message for marker in ("cuda", "cudnn", "cublas", "gpu driver")):
        return WorkerError("incompatible_cuda")
    return WorkerError(fallback)


def read_pages(model, rasters: list[Raster]) -> list[dict]:
    pages = []
    try:
        for index, page in enumerate(rasters, 1):
            start = perf_counter()
            results = list(model.predict(input=str(page.path)))
            if len(results) != 1:
                raise WorkerError("worker_failed")
            pages.append(
                {
                    "page": index,
                    "width": page.width,
                    "height": page.height,
                    "inference_time_ms": (perf_counter() - start) * 1000,
                    "lines": recognition_lines(results[0], page.width, page.height),
                }
            )
    except Exception as error:  # noqa: BLE001 - vendor diagnostics can contain text and paths.
        raise paddle_error(error, "worker_failed") from None
    return pages


def write_markdown(pages: list[dict]) -> None:
    sections = ["# OCR"]
    for page in pages:
        lines = [
            "    " + text for row in page["lines"] for text in row["text"].splitlines()
        ]
        sections.append(f"## Page {page['page']}\n\n" + "\n".join(lines))
    Path("ocr.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _wheel_chunk(start: int) -> bytes:
    end = min(COLAB_WHEEL_SIZE, start + WHEEL_CHUNK_SIZE) - 1
    size = end - start + 1
    request = Request(COLAB_WHEEL_URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{COLAB_WHEEL_SIZE}"
                ):
                    raise OSError("invalid wheel range")
                data = response.read(size + 1)
            if len(data) != size:
                raise OSError("incomplete wheel range")
            return data
        except (OSError, HTTPException):
            if attempt == 2:
                raise OSError("wheel range could not be retrieved") from None
    raise AssertionError("unreachable download attempt")


def _download_wheel(wheel: Path) -> None:
    digest = sha256()
    offsets = range(0, COLAB_WHEEL_SIZE, WHEEL_CHUNK_SIZE)
    with (
        wheel.open("xb") as output,
        ThreadPoolExecutor(max_workers=WHEEL_DOWNLOAD_WORKERS) as pool,
    ):
        for batch in range(0, len(offsets), WHEEL_DOWNLOAD_WORKERS):
            for data in pool.map(
                _wheel_chunk, offsets[batch : batch + WHEEL_DOWNLOAD_WORKERS]
            ):
                output.write(data)
                digest.update(data)
    if digest.hexdigest() != COLAB_WHEEL_SHA256:
        raise ValueError("wheel digest mismatch")


# The child imports the same packaged worker but has its own site-packages.
_ISOLATED_ENTRY = """
import sys
from importlib import import_module

worker = import_module(sys.argv[1])
packages = {**worker.PINS, "paddlepaddle-gpu": worker.PADDLE_VERSION, "uv": worker.UV_VERSION}
try:
    metrics = worker._recognize(
        worker.options(*worker.request_fields()), packages,
        float(sys.argv[3]), float(sys.argv[4]),
    )
    report = {"metrics": metrics}
except worker.WorkerError as error:
    report = {"error": error.code}
worker.write_json(sys.argv[2], report)
"""


def _run_isolated() -> dict:
    ensure_packages({"uv": UV_VERSION})
    uv = [sys.executable, "-m", "uv"]
    quiet = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    with TemporaryDirectory(prefix="gpu-ai-paddle-wheel-") as directory:
        root = Path(directory)
        wheel = root / COLAB_WHEEL
        venv = root / "venv"
        python = venv / "bin" / "python"
        try:
            subprocess.run(
                [*uv, "venv", "--python", sys.executable, str(venv)],
                timeout=60,
                check=True,
                **quiet,
            )
            started = perf_counter()
            _download_wheel(wheel)
            download_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            pins = [f"{package}=={release}" for package, release in PINS.items()]
            subprocess.run(
                [*uv, "pip", "install", "--python", str(python), str(wheel), *pins],
                timeout=420,
                check=True,
                **quiet,
            )
            install_ms = (perf_counter() - started) * 1000
        except (OSError, ValueError, subprocess.SubprocessError):
            raise WorkerError("worker_dependencies") from None

        report = root / "report.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[__name__.count(".")]
        )
        environment["PYTHONNOUSERSITE"] = "1"
        try:
            subprocess.run(
                [
                    str(python),
                    "-c",
                    _ISOLATED_ENTRY,
                    __name__,
                    str(report),
                    str(download_ms),
                    str(install_ms),
                ],
                env=environment,
                check=True,
                **quiet,
            )
            result = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError, subprocess.SubprocessError):
            raise WorkerError("worker_failed") from None
        if "error" in result:
            raise WorkerError(result["error"])
        return result["metrics"]


def run() -> dict:
    settings = options(*request_fields())
    if (
        settings.device == "cuda"
        and sys.version_info[:2] == (3, 13)
        and sys.platform == "linux"
        and platform.machine() == "x86_64"
    ):
        return _run_isolated()
    software = ensure_packages(PINS)
    requested = "paddlepaddle-gpu" if settings.device == "cuda" else "paddlepaddle"
    index = PADDLE_CUDA_INDEX if requested == "paddlepaddle-gpu" else None
    software.update(ensure_packages({requested: PADDLE_VERSION}, index_url=index))
    return _recognize(settings, software)


def _recognize(
    settings: Options,
    software: dict[str, str],
    wheel_download_ms: float | None = None,
    paddle_install_ms: float | None = None,
) -> dict:
    try:
        import paddle
        from paddleocr import PaddleOCR
    except (ImportError, OSError, RuntimeError):
        raise WorkerError("worker_dependencies") from None
    software["paddlepaddle_runtime"] = str(paddle.__version__)
    software["numpy"] = version("numpy")
    started = perf_counter()
    with TemporaryDirectory(prefix="gpu-ai-ocr-") as temporary:
        rasters = rasterize(Path(settings.path), Path(temporary), settings.dpi)
        load_started = perf_counter()
        model, device, fallback = use_model(settings, paddle, PaddleOCR)
        load_ms = (perf_counter() - load_started) * 1000
        try:
            pages = read_pages(model, rasters)
        except WorkerError as error:
            if (
                device != "cuda"
                or not settings.allow_cpu_fallback
                or error.code not in {"incompatible_cuda", "out_of_memory"}
            ):
                raise
            fallback = error.code
            # Releasing the Paddle object before CPU reinitialization keeps one active model.
            del model
            load_started = perf_counter()
            try:
                model = model_on("cpu", settings.language, PaddleOCR)
            except Exception as cpu_error:  # noqa: BLE001 - vendor boundary.
                raise paddle_error(cpu_error, "model_load_failed") from None
            load_ms += (perf_counter() - load_started) * 1000
            device = "cpu"
            pages = read_pages(model, rasters)
    metrics = {
        "model_id": RECOGNIZERS[settings.language],
        "detector_model_id": DETECTOR,
        "model_version": "PP-OCRv5",
        "software": software,
        "requested_device": settings.device,
        "device": device,
        "cpu_fallback": fallback is not None,
        "fallback_reason": fallback,
        "paddle_cuda_version": str(paddle.version.cuda()) if device == "cuda" else None,
        "wheel_download_time_ms": wheel_download_ms,
        "paddle_install_time_ms": paddle_install_ms,
        "model_load_time_ms": load_ms,
        "inference_time_ms": sum(page["inference_time_ms"] for page in pages),
        "processing_time_ms": (perf_counter() - started) * 1000,
        "page_count": len(pages),
    }
    write_json(
        "ocr.json",
        {
            "schema_version": "1",
            "source": settings.path,
            "language": settings.language,
            "dpi": settings.dpi
            if Path(settings.path).suffix.lower() == ".pdf"
            else None,
            "reading_order": "page_then_paddleocr_detection_order",
            "pages": pages,
            "metrics": metrics,
        },
    )
    if settings.markdown:
        write_markdown(pages)
    return metrics
