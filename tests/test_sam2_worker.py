import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, sam2
from gpu_ai.workers._runtime import WorkerError

POINT = {"object_id": 6, "point": [8, 8]}
BOX = {"object_id": 3, "box": [4, 4, 40, 30]}


@pytest.fixture
def fake_sam2(monkeypatch):
    monkeypatch.setattr(sam2, "ensure_packages", lambda pins: pins)
    prompts = []

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert args == (sam2.MODEL_ID,)
            assert kwargs["revision"] == sam2.REVISION
            return cls()

        def init_video_session(self, *, video, **kwargs):
            assert kwargs == {
                "inference_device": "cuda",
                "processing_device": "cpu",
                "inference_state_device": "cpu",
                "video_storage_device": "cpu",
                "dtype": "fp32",
            }
            return SimpleNamespace(video=video, obj_ids=[])

        def add_inputs_to_inference_session(
            self, *, inference_session, frame_idx, obj_ids, **kwargs
        ):
            assert frame_idx == 0
            assert kwargs in (
                {"input_points": [[[[8, 8]]]], "input_labels": [[[1]]]},
                {"input_boxes": [[[4, 4, 40, 30]]]},
            )
            prompts.append((obj_ids, kwargs))
            inference_session.obj_ids.append(obj_ids)

        def post_process_masks(self, masks, *, original_sizes, binarize):
            assert binarize
            assert tuple(original_sizes[0]) == masks[0].numpy().shape[-2:]
            return masks

    class Model:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            assert model == sam2.MODEL_ID
            assert kwargs == {
                "revision": sam2.REVISION,
                "dtype": "fp32",
                "use_safetensors": True,
            }
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def propagate_in_video_iterator(self, session, *, start_frame_idx):
            assert start_frame_idx == 0
            for index, frame in enumerate(session.video):
                masks = np.zeros(
                    (len(session.obj_ids), 1, frame.height, frame.width), dtype=np.bool_
                )
                masks[0, 0, 3:10, 2 + index : 11 + index] = True
                if len(session.obj_ids) > 1 and index != 2:
                    masks[1, 0, 7:22, 8:14] = True

                def output(mask):
                    return SimpleNamespace(cpu=lambda: output(mask), numpy=lambda: mask)

                yield SimpleNamespace(frame_idx=index, pred_masks=output(masks))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32="fp32",
            __version__="mock",
            OutOfMemoryError=MemoryError,
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                reset_peak_memory_stats=lambda: None,
                synchronize=lambda: None,
                max_memory_allocated=lambda: 40 * 1024 * 1024,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Sam2VideoProcessor=Processor,
            Sam2VideoModel=Model,
        ),
    )
    return prompts


def generate_video(path, *, fps=4, variable=True):
    selection = (
        ["-vf", r"select=not(eq(n\,2)+eq(n\,5))", "-fps_mode", "passthrough"]
        if variable
        else []
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
            f"testsrc2=size=64x48:rate={fps}:duration=2",
            *selection,
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        timeout=20,
    )


@pytest.mark.parametrize("video", [False, True])
def test_sam2_preserves_source_frame_indices_and_shapes(
    tmp_path, simulated_colab, fake_sam2, video
):
    requests = {"objects": [POINT, BOX]}
    if video:
        filename = "clip.mp4"
        generate_video(tmp_path / filename)
        requests.update(start_frame=1, frame_count=4, overlay=True)
    else:
        filename = "frame.png"
        Image.new("RGB", (64, 48), "blue").save(tmp_path / filename)
    calls = simulated_colab(sam2.run, "sam2")
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    runner = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    )
    job = TaskRequest(
        operation=Operation.SEGMENT_OBJECT, inputs=[filename], parameters=requests
    )
    result = runner.run(job)
    assert result.status is JobStatus.COMPLETED, result
    assert result.cleanup_status is CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert [obj_id for obj_id, _ in fake_sam2] == [6, 3]
    assert tuple(artifact.path for artifact in result.artifacts) == sam2.output_paths(
        job
    )
    folder = settings.results_dir / str(result.job_id)
    tracks = json.loads((folder / "tracks.json").read_text(encoding="utf-8"))["tracks"]
    assert len(tracks) == (8 if video else 2)
    assert [row["object_id"] for row in tracks] == [6, 3] * (4 if video else 1)
    assert [row["frame_index"] for row in tracks[::2]] == (
        [1, 2, 3, 4] if video else [0]
    )
    assert [row["time_sec"] for row in tracks[::2]] == pytest.approx(
        [0.25, 0.75, 1, 1.5] if video else [0]
    )
    assert tracks[0]["box"] == [2, 3, 11, 10]
    for row in tracks:
        with Image.open(folder / row["mask_path"]) as mask:
            assert mask.size == (64, 48)
            assert set(np.unique(np.asarray(mask))) <= {0, 255}
    if video:
        assert tracks[0]["pts"] * float(sam2.Fraction(tracks[0]["time_base"])) == 0.25
        assert tracks[5]["box"] is None
        assert result.metrics["source_fps"] > 0
        with av.open(str(folder / "overlay.mp4")) as output:
            assert [
                float(frame.pts * frame.time_base) for frame in output.decode(video=0)
            ] == (pytest.approx([0.0, 0.5, 0.75, 1.25], abs=1e-3))


@pytest.mark.parametrize(
    ("filename", "parameters"),
    [
        ("sample.png", {"objects": []}),
        ("sample.png", {"objects": [{"object_id": 1, "box": [14, 4, 10, 40]}]}),
        (
            "sample.png",
            {"objects": [{"object_id": 1, "point": [1, 1], "box": [1, 1, 3, 3]}]},
        ),
        ("sample.png", {"objects": [{"object_id": 1, "name": "tree"}]}),
        ("sample.png", {"objects": [POINT, POINT]}),
        ("sample.png", {"objects": [POINT], "frame_count": 1}),
        ("sample.mp4", {"objects": [BOX]}),
        ("sample.mp4", {"objects": [BOX], "start_frame": 45, "frame_count": 4}),
        ("sample.mp4", {"objects": [BOX], "frame_count": 49}),
    ],
)
def test_invalid_prompts_rejected(filename, parameters):
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        sam2.options([filename], parameters)


def test_rejects_full_video_over_frame_limit(tmp_path):
    video = tmp_path / "long.mp4"
    generate_video(video, fps=25, variable=False)
    settings = sam2.options([str(video)], {"objects": [BOX], "frame_count": 4})
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        sam2.load_video(settings, av)


def test_prompts_must_fit_source_image():
    prompt = sam2.object_prompt(BOX)
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        prompt.check_bounds(20, 30)
