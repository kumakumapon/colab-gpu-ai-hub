import json
import sys
from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import pytest
from PIL import Image

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, image_gen
from gpu_ai.workers._runtime import WorkerError


@pytest.fixture
def diffusion(monkeypatch):
    state = SimpleNamespace(seeds=[], flags=[False], token_count=10, checker=True)
    monkeypatch.setattr(image_gen, "ensure_packages", lambda pins: pins)

    class Generator:
        def __init__(self, *, device):
            assert device == "cuda"

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class Pipeline:
        scheduler = SimpleNamespace()
        tokenizer = SimpleNamespace()
        feature_extractor = object()

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            assert model_path == "/local/model"
            assert kwargs == {
                "local_files_only": True,
                "variant": "fp16",
                "torch_dtype": "fp16",
                "use_safetensors": True,
                "requires_safety_checker": True,
            }
            obj = cls()
            obj.safety_checker = object() if state.checker else None
            obj.tokenizer = SimpleNamespace(model_max_length=77)
            return obj

        def to(self, device):
            self.tokenizer = Tokenizer()
            assert device == "cuda"
            return self

        def set_progress_bar_config(self, *, disable):
            assert disable

        def __call__(self, **kwargs):
            assert kwargs["num_images_per_prompt"] == 1
            assert kwargs["num_inference_steps"] == 4
            assert kwargs["negative_prompt"] == "blur"
            seed = kwargs["generator"].seed
            state.seeds.append(seed)
            png = Image.new(
                "RGB", (kwargs["width"], kwargs["height"]), (seed % 255, 22, 50)
            )
            return SimpleNamespace(images=[png], nsfw_content_detected=state.flags)

    class Tokenizer:
        model_max_length = 77

        def __call__(self, text, **kwargs):
            assert text in {"test landscape", "blur"}
            return SimpleNamespace(input_ids=range(state.token_count))

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Generator=Generator,
            float16="fp16",
            OutOfMemoryError=MemoryError,
            inference_mode=nullcontext,
            __version__="mock",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                reset_peak_memory_stats=lambda: None,
                synchronize=lambda: None,
                max_memory_allocated=lambda: 12 * 1024 * 1024,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(StableDiffusionPipeline=Pipeline)
    )

    def download(model_id, *, revision, allow_patterns, max_workers):
        assert model_id == image_gen.MODEL_ID and revision == image_gen.REVISION
        assert "safety_checker/model.fp16.safetensors" in allow_patterns
        assert max_workers == 4
        return "/local/model"

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    return state


def request(**options):
    parameters = {
        "prompt": "test landscape",
        "negative_prompt": "blur",
        "seed": 19,
        "width": 256,
        "height": 320,
        "steps": 4,
        **options,
    }
    return TaskRequest(operation=Operation.GENERATE_IMAGE, parameters=parameters)


def make_runner(tmp_path):
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    return JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    )


def test_image_bytes_are_repeatable_and_registered(
    tmp_path, simulated_colab, diffusion
):
    calls = simulated_colab(image_gen.run, "image_gen")
    runner = make_runner(tmp_path)
    first, second = [
        runner.run(request(images_per_prompt=3, contact_sheet=True)) for _ in range(2)
    ]
    assert first.status == second.status == JobStatus.COMPLETED
    assert first.cleanup_status == second.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"] * 2
    assert diffusion.seeds == [19, 20, 21] * 2
    assert [artifact.path for artifact in first.artifacts] == [
        "image_0001.png",
        "image_0002.png",
        "image_0003.png",
        "metadata.json",
        "contact_sheet.png",
    ]
    second_artifacts = {item.path: item.sha256 for item in second.artifacts}
    for item in first.artifacts:
        location = runner.settings.results_dir / str(first.job_id) / item.path
        assert sha256(location.read_bytes()).hexdigest() == item.sha256
        if item.path.endswith(".png"):
            assert item.sha256 == second_artifacts[item.path]
            with Image.open(location) as image:
                assert image.format == "PNG"
                image.load()
                expected = (
                    (512, 640) if item.path == "contact_sheet.png" else (256, 320)
                )
                assert image.size == expected
        else:
            metadata = json.loads(location.read_text(encoding="utf-8"))
            assert metadata["safety_checker"] == "passed"
            assert metadata["model_id"] == image_gen.MODEL_ID
            assert metadata["peak_vram_mb"] == 12
            assert metadata["model_download_time_ms"] >= 0
            assert [image["seed"] for image in metadata["images"]] == [19, 20, 21]
            assert "prompt" not in metadata


@pytest.mark.parametrize(
    ("flags", "checker", "token_count", "failure"),
    [
        ([True], True, 10, "unsafe_output"),
        (None, True, 10, "worker_failed"),
        ([False], False, 10, "model_load_failed"),
        ([False], True, 78, "invalid_worker_input"),
    ],
)
def test_image_worker_rejects_unsafe_or_invalid_outputs(
    tmp_path,
    simulated_colab,
    diffusion,
    flags,
    checker,
    token_count,
    failure,
):
    diffusion.flags, diffusion.checker, diffusion.token_count = (
        flags,
        checker,
        token_count,
    )
    calls = simulated_colab(image_gen.run, "image_gen")
    runner = make_runner(tmp_path)
    result = runner.run(request())
    assert result.error == f"execute:{failure}"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert not runner.settings.results_dir.exists()
    assert calls == ["sessions", "new", "upload", "exec", "stop"]


@pytest.mark.parametrize(
    "options",
    [
        {"prompt": ""},
        {"negative_prompt": "\0"},
        {"width": 257},
        {"height": 832},
        {"steps": 41},
        {"images_per_prompt": 5},
        {"guidance_scale": 1},
        {"seed": -1},
        {"contact_sheet": "true"},
    ],
)
def test_image_generation_limits(options):
    job = request(**options)
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        image_gen.validate_request(job)
