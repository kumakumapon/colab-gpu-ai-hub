"""Relative inverse-depth estimation with Depth Anything V2 Small."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ._runtime import (
    TRANSFORMERS_VERSION,
    WorkerError,
    check_parameters,
    cuda_metrics,
    ensure_packages,
    integer,
    peak_memory_mib,
    request_fields,
    synchronize,
    torch_failure,
    write_json,
)

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
ARTIFACTS = ("depth.npy", "depth_preview.png", "metadata.json")
MAX_IMAGES = 4
MAX_TOTAL_PIXELS = 16_000_000


@dataclass(frozen=True)
class Options:
    inputs: list[str]
    size: int
    device: str


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(parameters, {"input_size", "device", "normalization"})
    if not 1 <= len(inputs) <= MAX_IMAGES or any(
        Path(path).suffix.lower() not in {".jpg", ".jpeg", ".png"} for path in inputs
    ):
        raise WorkerError("invalid_worker_input")
    size = integer(parameters.get("input_size", 518), 280, 700)
    device = parameters.get("device", "cuda")
    if (
        size % 14
        or device not in ("cuda", "cpu")
        or parameters.get("normalization", "imagenet") != "imagenet"
    ):
        raise WorkerError("invalid_worker_input")
    return Options(inputs, size, device)


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def output_directory(index: int, count: int) -> Path:
    return Path(".") if count == 1 else Path("images") / f"{index + 1:04d}"


def output_paths(request) -> tuple[str, ...]:
    count = len(request.inputs)
    return tuple(
        (output_directory(index, count) / path).as_posix()
        for index in range(count)
        for path in ARTIFACTS
    )


def load_images(inputs: list[str], image_module) -> list:
    images = []
    total = 0
    for path in inputs:
        try:
            with image_module.open(path, formats=("JPEG", "PNG")) as image:
                total += image.width * image.height
                if total > MAX_TOTAL_PIXELS or max(image.size) > 12 * min(image.size):
                    raise WorkerError("invalid_worker_input")
                images.append(image.convert("RGB"))
        except (OSError, ValueError, image_module.DecompressionBombError):
            raise WorkerError("invalid_worker_input") from None
    return images


def save_depth(directory: Path, depth, metadata: dict, *, numpy, image_module) -> None:
    depth = numpy.asarray(depth, dtype=numpy.float32)
    if depth.shape != (metadata["height"], metadata["width"]):
        raise WorkerError("worker_failed")
    finite = depth[numpy.isfinite(depth)]
    if finite.size == 0:
        raise WorkerError("worker_failed")
    minimum, maximum = float(finite.min()), float(finite.max())
    depth = numpy.nan_to_num(depth, nan=minimum, neginf=minimum, posinf=maximum)
    preview = numpy.zeros(depth.shape, dtype=numpy.uint16)
    if maximum > minimum:
        normalized = (depth.astype(numpy.float64) - minimum) / (maximum - minimum)
        preview = numpy.rint(normalized * 65535).astype(numpy.uint16)
    directory.mkdir(parents=True, exist_ok=True)
    numpy.save(directory / ARTIFACTS[0], depth, allow_pickle=False)
    image_module.fromarray(preview).save(directory / ARTIFACTS[1])
    write_json(
        directory / ARTIFACTS[2],
        {
            **metadata,
            "depth_kind": "relative_inverse_depth",
            "preview_normalization": "per_image_minmax",
            "min_relative_depth": minimum,
            "max_relative_depth": maximum,
        },
    )


def run() -> dict:
    settings = options(*request_fields())
    software = ensure_packages(
        {"transformers": TRANSFORMERS_VERSION, "pillow": "13.0.0"}
    )
    try:
        import numpy
        import PIL
        import torch
        from PIL import Image
        from transformers import AutoModelForDepthEstimation
        from transformers.models.dpt.image_processing_pil_dpt import (
            DPTImageProcessorPil,
        )
    except (ImportError, OSError, RuntimeError):
        raise WorkerError("worker_dependencies") from None
    software.update(
        numpy=numpy.__version__, pillow=PIL.__version__, torch=str(torch.__version__)
    )
    images = load_images(settings.inputs, Image)
    cuda_metrics(torch, settings.device)
    started = perf_counter()
    try:
        processor = DPTImageProcessorPil.from_pretrained(
            MODEL_ID,
            revision=REVISION,
        )
        model = (
            AutoModelForDepthEstimation.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                use_safetensors=True,
                trust_remote_code=False,
            )
            .to(settings.device)
            .eval()
        )
        synchronize(torch, settings.device)
    except Exception as error:  # noqa: BLE001 - suppress untrusted upstream diagnostics.
        raise torch_failure(torch, error, "model_load_failed") from None
    load_time = perf_counter() - started
    try:
        depths, batch_times, inference_time = predict(
            torch, processor, model, images, settings
        )
    except (RuntimeError, ValueError) as error:
        raise torch_failure(torch, error, "worker_failed") from None
    peak = peak_memory_mib(torch, settings.device)
    metrics = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "software": software,
        "device": settings.device,
        "input_size": settings.size,
        "normalization": "imagenet",
        "model_load_time_ms": load_time * 1000,
        "inference_time_ms": inference_time,
        "peak_vram_mb": peak,
        "image_count": len(images),
    }
    for index, image in enumerate(images):
        save_depth(
            output_directory(index, len(images)),
            depths[index],
            {
                "source": settings.inputs[index],
                "width": image.width,
                "height": image.height,
                "model_id": MODEL_ID,
                "model_revision": REVISION,
                "inference_time_ms": batch_times[index],
                "peak_vram_mb": peak,
            },
            numpy=numpy,
            image_module=Image,
        )
    return metrics


def predict(
    torch, processor, model, images: list, settings: Options
) -> tuple[dict, dict[int, float], float]:
    groups = defaultdict(list)
    for index, image in enumerate(images):
        pixels = processor(
            images=image,
            size={"height": settings.size, "width": settings.size},
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            return_tensors="pt",
        ).pixel_values
        groups[tuple(pixels.shape[-2:])].append((index, pixels))

    depths, durations = {}, {}
    total_time = 0.0
    with torch.inference_mode():
        for group in groups.values():
            indices, pixels = zip(*group)
            inputs = torch.cat(pixels, dim=0).to(settings.device)
            synchronize(torch, settings.device)
            started = perf_counter()
            predictions = model(pixel_values=inputs)
            outputs = processor.post_process_depth_estimation(
                predictions,
                target_sizes=[
                    (images[index].height, images[index].width) for index in indices
                ],
            )
            synchronize(torch, settings.device)
            elapsed = (perf_counter() - started) * 1000
            total_time += elapsed
            for index, output in zip(indices, outputs, strict=True):
                depths[index] = output["predicted_depth"].cpu().numpy()
                durations[index] = elapsed
    return depths, durations, total_time
