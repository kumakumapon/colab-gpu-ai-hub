import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy
import pytest

from gpu_ai.backends.colab_cli import ColabCLIBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import CleanupStatus, JobStatus, Operation, TaskRequest
from gpu_ai.runner import JobRunner
from gpu_ai.workers import colab_worker_scripts, llm
from gpu_ai.workers._runtime import WorkerError

PROMPT = "挨拶だけを書いてください。"
LARGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class ArtificialOOM(RuntimeError):
    pass


@pytest.fixture
def qwen_model(monkeypatch):
    state = SimpleNamespace(
        tokens=3,
        model_error=None,
        generated=[100, 151645],
        response="こんにちは。",
        loaded=[],
        seeds=[],
        criteria_inputs=[],
    )
    monkeypatch.setattr(llm, "ensure_packages", lambda pins: pins)

    class Batch(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    class Tokenizer:
        pad_token_id = 151643

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert len(kwargs["revision"]) == 40 and not kwargs["trust_remote_code"]
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": PROMPT}]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            return Batch(input_ids=numpy.zeros((1, state.tokens), dtype="int64"))

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens
            assert tokens == state.generated
            return state.response

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[151645, 151643])

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            state.loaded.append(model_id)
            assert kwargs == {
                "revision": llm.MODELS[model_id],
                "dtype": "fp16",
                "use_safetensors": True,
                "trust_remote_code": False,
            }
            if state.model_error:
                raise state.model_error
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def generate(self, *, input_ids, **kwargs):
            if kwargs["do_sample"]:
                assert kwargs["temperature"] == 0.75
            else:
                assert kwargs["temperature"] is None
            assert kwargs["max_new_tokens"] == 12
            assert kwargs["pad_token_id"] == Tokenizer.pad_token_id
            result = numpy.concatenate((input_ids, [state.generated]), axis=1)
            for stop in kwargs.get("stopping_criteria", []):
                assert stop(result, None)
            return result

    class StopStringCriteria:
        def __init__(self, tokenizer, stops):
            assert stops == ["<stop>"]

        def __call__(self, input_ids, scores, **kwargs):
            state.criteria_inputs.append(input_ids.tolist())
            return True

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="mock",
            OutOfMemoryError=ArtificialOOM,
            float16="fp16",
            manual_seed=state.seeds.append,
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                is_available=lambda: True,
                reset_peak_memory_stats=lambda: None,
                synchronize=lambda: None,
                max_memory_allocated=lambda: 8 * 1024 * 1024,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=Tokenizer,
            AutoModelForCausalLM=Model,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
            StopStringCriteria=StopStringCriteria,
        ),
    )
    return state


@pytest.mark.parametrize("prompt_file", [False, True])
def test_qwen_greedy_inference_and_response_schema(
    tmp_path,
    simulated_colab,
    qwen_model,
    prompt_file,
):
    workspace = tmp_path / "inputs"
    workspace.mkdir()
    parameters = {"max_new_tokens": 12, "seed": 9, "temperature": 0}
    inputs = []
    if prompt_file:
        (workspace / "prompt.txt").write_text(PROMPT, encoding="utf-8")
        inputs.append("prompt.txt")
    else:
        parameters["prompt"] = PROMPT
    settings = Settings(workspace=workspace, results_dir=tmp_path / "results")
    calls = simulated_colab(llm.run, "llm")
    backend = ColabCLIBackend(colab_worker_scripts(), settings=settings)
    result = JobRunner(backend, settings).run(
        TaskRequest(operation=Operation.LLM_INFER, inputs=inputs, parameters=parameters)
    )
    assert result.status == JobStatus.COMPLETED, result
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == ["sessions", "new", "upload", "exec", "download", "stop"]
    assert qwen_model.loaded == [llm.DEFAULT_MODEL]
    assert qwen_model.seeds == [9]
    assert "text" not in result.metrics and "prompt" not in result.metrics
    folder = settings.results_dir / str(result.job_id)
    response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
    assert response["text"] == "こんにちは。"
    assert response["model_id"] == llm.DEFAULT_MODEL
    assert response["input_tokens"] == 3
    assert response["output_tokens"] == 2
    assert response["finish_reason"] == "eos"
    assert response["peak_vram_mb"] == 8
    assert response["model_load_time_ms"] >= 0
    assert response["generation_time_ms"] >= 0
    assert response["tokens_per_sec"] > 0
    manifest = json.loads((folder / "gpu-ai-manifest.json").read_text(encoding="utf-8"))
    assert [item["mime_type"] for item in manifest["inputs"]] == (
        ["text/plain"] if prompt_file else []
    )


def test_custom_stop_is_applied_only_to_generated_tokens(
    tmp_path,
    simulated_colab,
    qwen_model,
):
    qwen_model.generated = [100, 111, 102]
    qwen_model.response = "こんにちは。<stop>余計な文字"
    calls = simulated_colab(llm.run, "llm")
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    result = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    ).run(
        TaskRequest(
            operation=Operation.LLM_INFER,
            parameters={
                "prompt": PROMPT,
                "model_id": LARGE_MODEL,
                "temperature": 0.75,
                "stop": ["<stop>"],
                "max_new_tokens": 12,
            },
        ),
    )
    assert result.status == JobStatus.COMPLETED, result
    assert calls[-1] == "stop"
    assert qwen_model.criteria_inputs == [[qwen_model.generated]]
    assert qwen_model.loaded == [LARGE_MODEL]
    output = settings.results_dir / str(result.job_id) / "response.json"
    response = json.loads(output.read_text(encoding="utf-8"))
    assert response["text"] == "こんにちは。"
    assert response["finish_reason"] == "stop_string"


@pytest.mark.parametrize(
    ("overlong", "error"),
    [(True, "execute:invalid_worker_input"), (False, "execute:out_of_memory")],
)
def test_token_cap_and_oom_do_not_switch_model(
    tmp_path,
    simulated_colab,
    qwen_model,
    overlong,
    error,
):
    if overlong:
        qwen_model.tokens = llm.MAX_INPUT_TOKENS + 1
    else:
        qwen_model.model_error = ArtificialOOM("secret hardware details")
    calls = simulated_colab(llm.run, "llm")
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    result = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    ).run(
        TaskRequest(
            operation=Operation.LLM_INFER,
            parameters={"prompt": PROMPT, "model_id": LARGE_MODEL},
        )
    )
    assert result.error == error
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert qwen_model.loaded == ([] if overlong else [LARGE_MODEL])
    assert calls[-2:] == ["exec", "stop"]


@pytest.mark.parametrize(
    ("inputs", "parameters"),
    [
        ([], {}),
        ([], {"prompt": ""}),
        ([], {"prompt": " " * 10}),
        ([], {"prompt": "あ" * (llm.MAX_INPUT_CHARS + 1)}),
        (["input.txt"], {"prompt": PROMPT}),
        (["image.png"], {}),
        ([], {"prompt": PROMPT, "model_id": "other"}),
        ([], {"prompt": PROMPT, "temperature": 3}),
        ([], {"prompt": PROMPT, "max_new_tokens": 513}),
        ([], {"prompt": PROMPT, "seed": True}),
        ([], {"prompt": PROMPT, "stop": [""]}),
    ],
)
def test_rejects_invalid_generation_limits(inputs, parameters):
    with pytest.raises(WorkerError, match="invalid_worker_input"):
        llm.options(inputs, parameters)


@pytest.mark.parametrize("content", [b"\xff", b" ", b"a" * (llm.MAX_INPUT_CHARS + 1)])
def test_bad_prompt_file_is_rejected_before_colab(tmp_path, simulated_colab, content):
    (tmp_path / "prompt.txt").write_bytes(content)
    settings = Settings(workspace=tmp_path, results_dir=tmp_path / "results")
    calls = simulated_colab(llm.run, "llm")
    result = JobRunner(
        ColabCLIBackend(colab_worker_scripts(), settings=settings), settings
    ).run(TaskRequest(operation=Operation.LLM_INFER, inputs=["prompt.txt"]))
    assert result.error == "input_validation_failed"
    assert result.cleanup_status == CleanupStatus.CONFIRMED
    assert calls == []


def test_packaged_worker_runs_in_a_real_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "gpu-ai-request.json").write_text(
        json.dumps({"inputs": [], "parameters": {"prompt": " "}}), encoding="utf-8"
    )
    namespace = {"__name__": "__main__"}
    script = colab_worker_scripts()[Operation.LLM_INFER]
    exec(compile(script.source, "<test-colab-worker>", "exec"), namespace)  # noqa: S102
    assert namespace["gpu_ai_error"] == "invalid_worker_input"


def test_finish_at_generation_limit():
    assert llm.finish("本文", [100, 101], (), [151645], 2) == ("本文", "length")
