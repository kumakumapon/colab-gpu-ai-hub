"""Bounded Stable Diffusion v1.5 inference with its safety checker enabled."""

import math
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

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
FP16_MODEL_FILES = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "safety_checker/config.json",
    "safety_checker/model.fp16.safetensors",
)
PINS = {"diffusers": "0.43.0", "transformers": TRANSFORMERS_VERSION, "pillow": "13.0.0"}
MAX_PROMPT_CHARS = 600
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_SEED = 2**32 - 1


@dataclass(frozen=True)
class Options:
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    images_per_prompt: int
    contact_sheet: bool


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(
        parameters,
        {
            "prompt",
            "negative_prompt",
            "seed",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "images_per_prompt",
            "contact_sheet",
        },
    )
    prompt = parameters.get("prompt")
    negative = parameters.get("negative_prompt", "")
    width = integer(parameters.get("width", 512), 256, 768)
    height = integer(parameters.get("height", 512), 256, 768)
    steps = integer(parameters.get("steps", 20), 1, 40)
    count = integer(parameters.get("images_per_prompt", 1), 1, 4)
    seed = integer(parameters.get("seed", 0), 0, MAX_SEED)
    scale = parameters.get("guidance_scale", 7.5)
    sheet = parameters.get("contact_sheet", False)
    if (
        inputs
        or not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(negative, str)
        or any(
            len(text) > MAX_PROMPT_CHARS or "\x00" in text
            for text in (prompt, negative)
        )
        or width % 64
        or height % 64
        or width * height > 768 * 768
        or type(scale) not in (int, float)
        or not 1 <= scale <= 15
        or (negative.strip() and scale <= 1)
        or not isinstance(sheet, bool)
    ):
        raise WorkerError("invalid_worker_input")
    return Options(
        prompt, negative, seed, width, height, steps, float(scale), count, sheet
    )


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def image_paths(settings: Options) -> tuple[str, ...]:
    return tuple(
        f"image_{index:04d}.png" for index in range(1, settings.images_per_prompt + 1)
    )


def output_paths(request) -> tuple[str, ...]:
    settings = options(request.inputs, request.parameters)
    extra = ("contact_sheet.png",) if settings.contact_sheet else ()
    return (*image_paths(settings), "metadata.json", *extra)


def check_token_limits(tokenizer, settings: Options) -> None:
    for prompt in (settings.prompt, settings.negative_prompt):
        if (
            len(tokenizer(prompt, truncation=False, verbose=False).input_ids)
            > tokenizer.model_max_length
        ):
            raise WorkerError("invalid_worker_input")


def generate(torch, pipeline, settings: Options, device: str):
    results = []
    for index in range(settings.images_per_prompt):
        seed = (settings.seed + index) & MAX_SEED
        generator = torch.Generator(device=device).manual_seed(seed)
        started = perf_counter()
        try:
            with torch.inference_mode():
                output = pipeline(
                    prompt=settings.prompt,
                    negative_prompt=settings.negative_prompt,
                    width=settings.width,
                    height=settings.height,
                    num_inference_steps=settings.steps,
                    guidance_scale=settings.guidance_scale,
                    num_images_per_prompt=1,
                    generator=generator,
                    output_type="pil",
                )
            synchronize(torch, device)
        except (RuntimeError, ValueError) as error:
            raise torch_failure(torch, error, "worker_failed") from None
        flags = output.nsfw_content_detected
        if flags is None or len(flags) != 1 or len(output.images) != 1:
            raise WorkerError("worker_failed")
        if flags[0]:
            raise WorkerError("unsafe_output")
        image = output.images[0]
        if image.size != (settings.width, settings.height):
            raise WorkerError("worker_failed")
        results.append((image.convert("RGB"), seed, (perf_counter() - started) * 1000))
    return results


def save_png(image, path: str, max_bytes: int) -> None:
    image.save(path, format="PNG")
    if Path(path).stat().st_size > max_bytes:
        raise WorkerError("worker_failed")


def save_images(image_module, results, settings: Options) -> list[dict]:
    metadata = []
    for (image, seed, elapsed), path in zip(
        results, image_paths(settings), strict=True
    ):
        save_png(image, path, MAX_IMAGE_BYTES)
        metadata.append({"path": path, "seed": seed, "generation_time_ms": elapsed})
    if settings.contact_sheet:
        columns = min(2, len(results))
        rows = math.ceil(len(results) / columns)
        sheet = image_module.new(
            "RGB", (settings.width * columns, settings.height * rows)
        )
        for index, (image, _, _) in enumerate(results):
            sheet.paste(
                image,
                (index % columns * settings.width, index // columns * settings.height),
            )
        save_png(sheet, "contact_sheet.png", MAX_IMAGE_BYTES * 4)
    return metadata


def run() -> dict:
    settings = options(*request_fields())
    software = ensure_packages(PINS)
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from huggingface_hub import snapshot_download
        from PIL import Image
    except (ImportError, OSError, RuntimeError):
        raise WorkerError("worker_dependencies") from None
    device = "cuda"
    cuda_metrics(torch, device)
    started = perf_counter()
    try:
        model_path = snapshot_download(
            MODEL_ID, revision=REVISION, allow_patterns=FP16_MODEL_FILES, max_workers=4
        )
    except Exception:  # noqa: BLE001 - the Hub may include credentials in errors.
        raise WorkerError("model_download_failed") from None
    download_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
            requires_safety_checker=True,
        ).to(device)
        synchronize(torch, device)
    except Exception as error:  # noqa: BLE001 - never expose vendor or model download details.
        raise torch_failure(torch, error, "model_load_failed") from None
    if pipeline.safety_checker is None or pipeline.feature_extractor is None:
        raise WorkerError("model_load_failed")
    load_ms = (perf_counter() - started) * 1000
    pipeline.set_progress_bar_config(disable=True)
    check_token_limits(pipeline.tokenizer, settings)
    results = generate(torch, pipeline, settings, device)
    images = save_images(Image, results, settings)
    metrics = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "software": {**software, "torch": str(torch.__version__)},
        "scheduler": type(pipeline.scheduler).__name__,
        "width": settings.width,
        "height": settings.height,
        "steps": settings.steps,
        "guidance_scale": settings.guidance_scale,
        "seed": settings.seed,
        "image_count": len(images),
        "safety_checker": "passed",
        "model_download_time_ms": download_ms,
        "model_load_time_ms": load_ms,
        "generation_time_ms": sum(item["generation_time_ms"] for item in images),
        "peak_vram_mb": peak_memory_mib(torch, device),
    }
    write_json("metadata.json", {"schema_version": "1", **metrics, "images": images})
    return metrics
