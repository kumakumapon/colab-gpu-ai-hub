import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy
import pytest
from PIL import Image

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, depth
from gpu_ai.workers._runtime import WorkerError


class Tensor:
    def __init__(self, value):
        self.value = value
        self.shape = value.shape

    def to(self, device):
        assert device == "cpu"
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


@pytest.fixture
def relative_model(monkeypatch):
    groups = []
    monkeypatch.setattr(depth, "ensure_packages", lambda pins: pins)

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == depth.MODEL_ID and kwargs["revision"] == depth.REVISION
            return cls()

        def __call__(self, *, images, size, **kwargs):
            assert size == {"height": 518, "width": 518}
            assert kwargs["ensure_multiple_of"] == 14
            assert kwargs["keep_aspect_ratio"]
            target = (14, 14) if images.width == images.height else (14, 28)
            return SimpleNamespace(pixel_values=Tensor(numpy.zeros((1, 3, *target))))

        def post_process_depth_estimation(self, outputs, target_sizes):
            assert len(target_sizes) == outputs
            results = []
            for height, width in target_sizes:
                data = numpy.arange(height * width, dtype=numpy.float32).reshape(
                    height, width
                )
                data[0, :3] = [numpy.nan, -numpy.inf, numpy.inf]
                results.append({"predicted_depth": Tensor(data)})
            return results

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == depth.MODEL_ID and kwargs["use_safetensors"]
            return cls()

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            count = pixel_values.shape[0]
            groups.append(count)
            return count

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="mock",
            OutOfMemoryError=RuntimeError,
            cat=lambda tensors, dim: Tensor(
                numpy.concatenate([t.value for t in tensors], axis=dim)
            ),
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoImageProcessor=Processor,
            AutoModelForDepthEstimation=Model,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.dpt.image_processing_pil_dpt",
        SimpleNamespace(DPTImageProcessorPil=Processor),
    )
    return groups


@pytest.mark.parametrize("sizes", [[(8, 8)], [(8, 8), (12, 12), (15, 8)]])
def test_relative_depth_preserves_shapes_and_batches(
    tmp_path,
    simulated_colab,
    relative_model,
    sizes,
):
    workspace = tmp_path / "inputs"
    workspace.mkdir()
    names = []
    for index, size in enumerate(sizes):
        name = f"frame{index}.png"
        Image.new("RGB", size, "white").save(workspace / name)
        names.append(name)
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    calls = simulated_colab(depth.run, "depth")
    backend = ColabCLIBackend(colab_worker_scripts(), settings=settings)
    request = TaskRequest(
        operation=Operation.ESTIMATE_DEPTH, inputs=names, parameters={"device": "cpu"}
    )
    result = JobRunner(backend, settings).run(request)
    assert result.status == JobStatus.COMPLETED, result
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert relative_model == ([1] if len(sizes) == 1 else [2, 1])
    assert tuple(item.path for item in result.artifacts) == depth.output_paths(request)
    assert result.metrics["image_count"] == len(sizes)
    for index, (width, height) in enumerate(sizes):
        output = (
            settings.results_dir
            / str(result.job_id)
            / depth.output_directory(index, len(sizes))
        )
        array = numpy.load(output / "depth.npy", allow_pickle=False)
        assert array.shape == (height, width) and array.dtype == numpy.float32
        assert numpy.isfinite(array).all()
        with Image.open(output / "depth_preview.png") as preview:
            assert preview.size == (width, height)
            assert numpy.asarray(preview).max() == 65535
        meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert (meta["width"], meta["height"]) == (width, height)
        assert meta["depth_kind"] == "relative_inverse_depth"
        assert meta["min_relative_depth"] == array.min()
        assert meta["max_relative_depth"] == array.max()
        assert meta["model_revision"] == depth.REVISION
        assert meta["source"] == names[index]
        assert meta["inference_time_ms"] >= 0
        assert meta["peak_vram_mb"] is None


@pytest.mark.parametrize(
    ("inputs", "parameters"),
    [
        ([], {}),
        (["a.png"] * 5, {}),
        (["a.pdf"], {}),
        (["a.jpg"], {"input_size": 519}),
        (["a.jpg"], {"device": "mps"}),
        (["a.jpg"], {"normalization": "unit"}),
    ],
)
def test_rejects_unsupported_depth_options(inputs, parameters):
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        depth.options(inputs, parameters)


def test_nonfinite_only_depth_is_rejected(tmp_path):
    with pytest.raises(WorkerError, match="worker_failed"):
        depth.save_depth(
            tmp_path,
            [[numpy.nan]],
            {"height": 1, "width": 1},
            numpy=numpy,
            image_module=Image,
        )
    assert not list(tmp_path.iterdir())
