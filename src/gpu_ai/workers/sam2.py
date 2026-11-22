"""Prompted masks and short-video tracking with SAM 2.1 Hiera Tiny."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any

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

MODEL_ID = "facebook/sam2.1-hiera-tiny"
REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
PINS = {"transformers": TRANSFORMERS_VERSION, "pillow": "13.0.0", "av": "17.1.0"}
MAX_FRAMES = 48
MAX_DURATION_SECONDS = 8
MAX_OBJECTS = 4
MAX_OBJECT_ID = 9999
MAX_SIDE = 1280
MAX_FRAME_PIXELS = 1280 * 720
MAX_TOTAL_PIXELS = 32_000_000
COLORS = ((255, 80, 80), (70, 180, 255), (50, 220, 130), (255, 205, 60))


@dataclass(frozen=True)
class ObjectPrompt:
    object_id: int
    point: tuple[int, ...] | None
    box: tuple[int, ...] | None

    def processor_arguments(self) -> dict:
        if self.point is not None:
            return {"input_points": [[[list(self.point)]]], "input_labels": [[[1]]]}
        return {"input_boxes": [[list(self.box)]]}

    def check_bounds(self, width: int, height: int) -> None:
        if self.point is not None:
            x, y = self.point
            valid = x < width and y < height
        else:
            _, _, right, bottom = self.box
            valid = right <= width and bottom <= height
        if not valid:
            raise WorkerError("invalid_worker_input")


@dataclass(frozen=True)
class Options:
    path: str
    video: bool
    objects: tuple[ObjectPrompt, ...]
    start_frame: int
    frame_count: int
    overlay: bool


@dataclass(frozen=True)
class Frame:
    index: int
    image: Any
    time_sec: float
    pts: int | None = None
    time_base: str | None = None


@dataclass(frozen=True)
class Clip:
    frames: tuple[Frame, ...]
    fps: float | None = None
    source_start_sec: float | None = None
    duration_sec: float | None = None


def coordinate_list(value, count: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise WorkerError("invalid_worker_input")
    return tuple(integer(number, 0, MAX_SIDE) for number in value)


def object_prompt(data: dict) -> ObjectPrompt:
    if not isinstance(data, dict):
        raise WorkerError("invalid_worker_input")
    check_parameters(data, {"point", "box", "object_id"})
    object_id = integer(data.get("object_id"), 1, MAX_OBJECT_ID)
    point, box = data.get("point"), data.get("box")
    if (point is None) == (box is None):
        raise WorkerError("invalid_worker_input")
    if point is not None:
        return ObjectPrompt(object_id, coordinate_list(point, 2), None)
    bounds = coordinate_list(box, 4)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise WorkerError("invalid_worker_input")
    return ObjectPrompt(object_id, None, bounds)


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(parameters, {"objects", "start_frame", "frame_count", "overlay"})
    if len(inputs) != 1 or Path(inputs[0]).suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".mp4",
    }:
        raise WorkerError("invalid_worker_input")
    objects = parameters.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= MAX_OBJECTS:
        raise WorkerError("invalid_worker_input")
    prompts = tuple(object_prompt(item) for item in objects)
    if len({item.object_id for item in prompts}) != len(prompts):
        raise WorkerError("invalid_worker_input")

    video = Path(inputs[0]).suffix.lower() == ".mp4"
    overlay = parameters.get("overlay", False)
    if not isinstance(overlay, bool):
        raise WorkerError("invalid_worker_input")
    if video:
        start = integer(parameters.get("start_frame", 0), 0, MAX_FRAMES - 1)
        count = integer(parameters.get("frame_count"), 1, MAX_FRAMES)
        if start + count > MAX_FRAMES:
            raise WorkerError("invalid_worker_input")
    else:
        start, count = 0, 1
        if overlay or "start_frame" in parameters or "frame_count" in parameters:
            raise WorkerError("invalid_worker_input")
    return Options(inputs[0], video, prompts, start, count, overlay)


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def mask_path(frame_index: int, object_id: int) -> str:
    return f"masks/frame_{frame_index:06d}_object_{object_id:04d}.png"


def output_paths(request) -> tuple[str, ...]:
    selected = options(request.inputs, request.parameters)
    masks = tuple(
        mask_path(index, prompt.object_id)
        for index in range(
            selected.start_frame, selected.start_frame + selected.frame_count
        )
        for prompt in selected.objects
    )
    overlay = ("overlay.mp4",) if selected.overlay else ()
    return (*masks, "tracks.json", "metadata.json", *overlay)


def check_dimensions(width: int, height: int) -> None:
    if not (
        0 < width <= MAX_SIDE
        and 0 < height <= MAX_SIDE
        and width * height <= MAX_FRAME_PIXELS
    ):
        raise WorkerError("invalid_worker_input")


def load_image(path: str, image_module) -> Clip:
    try:
        with image_module.open(path, formats=("PNG", "JPEG")) as image:
            check_dimensions(*image.size)
            return Clip((Frame(0, image.convert("RGB"), 0.0),))
    except (ValueError, OSError, image_module.DecompressionBombError):
        raise WorkerError("invalid_worker_input") from None


def load_video(settings: Options, av) -> Clip:
    frames = []
    total_pixels = 0
    first_pts, previous_pts, frame_size = None, None, None
    end = settings.start_frame + settings.frame_count
    try:
        with av.open(
            settings.path, mode="r", format="mov", options={"enable_drefs": "0"}
        ) as container:
            if not container.streams.video:
                raise WorkerError("invalid_worker_input")
            stream = container.streams.video[0]
            check_dimensions(stream.width, stream.height)
            if stream.frames and stream.frames > MAX_FRAMES:
                raise WorkerError("invalid_worker_input")
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else None
            )
            if duration is not None and not 0 < duration <= MAX_DURATION_SECONDS:
                raise WorkerError("invalid_worker_input")
            fps = float(stream.average_rate) if stream.average_rate else None

            for index, frame in enumerate(container.decode(stream)):
                if index >= MAX_FRAMES or frame.pts is None or frame.time_base is None:
                    raise WorkerError("invalid_worker_input")
                pts = frame.pts * frame.time_base
                if first_pts is None:
                    first_pts = pts
                    frame_size = (frame.width, frame.height)
                    check_dimensions(*frame_size)
                elapsed = float(pts - first_pts)
                total_pixels += frame.width * frame.height
                if (
                    (previous_pts is not None and pts <= previous_pts)
                    or not 0 <= elapsed <= MAX_DURATION_SECONDS
                    or (frame.width, frame.height) != frame_size
                    or total_pixels > MAX_TOTAL_PIXELS
                ):
                    raise WorkerError("invalid_worker_input")
                previous_pts = pts
                if settings.start_frame <= index < end:
                    frames.append(
                        Frame(
                            index,
                            frame.to_image().convert("RGB"),
                            elapsed,
                            frame.pts,
                            str(frame.time_base),
                        )
                    )
    except (av.error.FFmpegError, OSError, ValueError, OverflowError):
        raise WorkerError("invalid_worker_input") from None
    if len(frames) != settings.frame_count:
        raise WorkerError("invalid_worker_input")
    elapsed = float(previous_pts - first_pts)
    return Clip(tuple(frames), fps, float(first_pts), max(elapsed, duration or 0))


def bounding_box(mask, np) -> list[int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    return [int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1]


@contextmanager
def overlay_writer(av, frame: Frame):
    # The tracker keeps the original time base. The preview uses a 30 kHz clock.
    clock = Fraction(1, 30_000)
    width, height = frame.image.size
    dimensions = (width + width % 2, height + height % 2)
    with av.open("overlay.mp4", mode="w", format="mp4") as output:
        stream = output.add_stream("mpeg4", rate=30)
        stream.width, stream.height = dimensions
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = clock
        previous = -1

        def write(image, frame_time: float) -> None:
            nonlocal previous
            video_frame = av.VideoFrame.from_image(image)
            if image.size != dimensions:
                video_frame = video_frame.reformat(
                    width=dimensions[0], height=dimensions[1]
                )
            video_frame.pts = max(
                previous + 1, round((frame_time - frame.time_sec) / clock)
            )
            video_frame.time_base = clock
            previous = video_frame.pts
            for packet in stream.encode(video_frame):
                output.mux(packet)

        yield write
        for packet in stream.encode():
            output.mux(packet)


def write_masks(
    raw_masks,
    frame: Frame,
    prompts: tuple[ObjectPrompt, ...],
    np,
    image_module,
    overlay,
):
    width, height = frame.image.size
    masks = raw_masks.cpu().numpy()
    if masks.shape != (len(prompts), 1, height, width):
        raise WorkerError("worker_failed")
    view = frame.image.copy() if overlay is not None else None
    records = []
    for index, prompt in enumerate(prompts):
        mask = np.asarray(masks[index, 0], dtype=np.bool_)
        path = mask_path(frame.index, prompt.object_id)
        pixels = mask.astype(np.uint8)
        image_module.fromarray(pixels * 255).save(path, format="PNG")
        if view is not None:
            color = image_module.new("RGB", (width, height), COLORS[index])
            view = image_module.composite(
                color, view, image_module.fromarray(pixels * 110)
            )
        records.append(
            {
                "frame_index": frame.index,
                "time_sec": frame.time_sec,
                "pts": frame.pts,
                "time_base": frame.time_base,
                "object_id": prompt.object_id,
                "mask_path": path,
                "box": bounding_box(mask, np),
            }
        )
    if view is not None:
        overlay(view, frame.time_sec)
    return records


def track(
    model,
    processor,
    torch,
    np,
    image_module,
    av,
    clip: Clip,
    settings: Options,
    device: str,
) -> list[dict]:
    width, height = clip.frames[0].image.size
    Path("masks").mkdir(exist_ok=True)
    session = processor.init_video_session(
        video=[frame.image for frame in clip.frames],
        inference_device=device,
        processing_device="cpu",
        inference_state_device="cpu",
        video_storage_device="cpu",
        dtype=torch.float32,
    )
    for prompt in settings.objects:
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=prompt.object_id,
            **prompt.processor_arguments(),
        )
    if list(session.obj_ids) != [prompt.object_id for prompt in settings.objects]:
        raise WorkerError("worker_failed")
    context = overlay_writer(av, clip.frames[0]) if settings.overlay else nullcontext()
    records = []
    processed = 0
    with torch.inference_mode(), context as overlay:
        for prediction in model.propagate_in_video_iterator(session, start_frame_idx=0):
            if prediction.frame_idx != processed or processed >= len(clip.frames):
                raise WorkerError("worker_failed")
            resized = processor.post_process_masks(
                [prediction.pred_masks.cpu()],
                original_sizes=[[height, width]],
                binarize=True,
            )[0]
            records.extend(
                write_masks(
                    resized,
                    clip.frames[processed],
                    settings.objects,
                    np,
                    image_module,
                    overlay,
                )
            )
            processed += 1
    if processed != len(clip.frames):
        raise WorkerError("worker_failed")
    return records


def run() -> dict:
    settings = options(*request_fields())
    software = ensure_packages(PINS)
    try:
        import av
        import numpy as np
        import torch
        from PIL import Image
        from transformers import Sam2VideoModel, Sam2VideoProcessor
    except (ImportError, OSError, RuntimeError):
        raise WorkerError("worker_dependencies") from None

    clip = (
        load_video(settings, av) if settings.video else load_image(settings.path, Image)
    )
    width, height = clip.frames[0].image.size
    for prompt in settings.objects:
        prompt.check_bounds(width, height)
    device = "cuda"
    cuda_metrics(torch, device)
    started = perf_counter()
    try:
        processor = Sam2VideoProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        model = (
            Sam2VideoModel.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                dtype=torch.float32,
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        synchronize(torch, device)
    except Exception as error:  # noqa: BLE001 - vendor boundary.
        raise torch_failure(torch, error, "model_load_failed") from None
    load_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    try:
        tracks = track(model, processor, torch, np, Image, av, clip, settings, device)
        synchronize(torch, device)
    except (RuntimeError, OSError, ValueError, av.error.FFmpegError) as error:
        if isinstance(error, WorkerError):
            raise
        raise torch_failure(torch, error, "worker_failed") from None
    metrics = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "software": {**software, "torch": str(torch.__version__)},
        "width": width,
        "height": height,
        "source_fps": clip.fps,
        "source_start_sec": clip.source_start_sec,
        "source_duration_sec": clip.duration_sec,
        "start_frame": settings.start_frame,
        "frame_count": len(clip.frames),
        "object_count": len(settings.objects),
        "model_load_time_ms": load_ms,
        "tracking_time_ms": (perf_counter() - started) * 1000,
        "peak_vram_mb": peak_memory_mib(torch, device),
    }
    write_json(
        "tracks.json",
        {"schema_version": "1", "source": settings.path, "tracks": tracks},
    )
    write_json("metadata.json", {"schema_version": "1", **metrics})
    return metrics
