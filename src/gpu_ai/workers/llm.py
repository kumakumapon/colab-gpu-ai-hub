"""Bounded text generation with Qwen2.5 Instruct."""

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

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS = {
    DEFAULT_MODEL: "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
}
ARTIFACTS = ("response.json",)
MAX_INPUT_CHARS = 12_000
MAX_PROMPT_BYTES = MAX_INPUT_CHARS * 4 + 3
MAX_INPUT_TOKENS = 4_096


@dataclass(frozen=True)
class Options:
    model_id: str
    prompt: str | None
    path: str | None
    max_new_tokens: int
    seed: int
    temperature: float
    stops: tuple[str, ...]


def validate_prompt(prompt: object) -> str:
    if (
        not isinstance(prompt, str)
        or not 0 < len(prompt) <= MAX_INPUT_CHARS
        or not prompt.strip()
        or "\x00" in prompt
    ):
        raise WorkerError("invalid_worker_input")
    return prompt


def decode_prompt(data: bytes) -> str:
    if len(data) > MAX_PROMPT_BYTES:
        raise WorkerError("invalid_worker_input")
    try:
        return validate_prompt(data.decode("utf-8-sig"))
    except UnicodeError:
        raise WorkerError("invalid_worker_input") from None


def options(inputs: list[str], parameters: dict) -> Options:
    check_parameters(
        parameters,
        {"model_id", "prompt", "max_new_tokens", "seed", "temperature", "stop"},
    )
    path = None
    prompt = None
    if "prompt" in parameters and not inputs:
        prompt = validate_prompt(parameters["prompt"])
    elif (
        "prompt" not in parameters
        and len(inputs) == 1
        and Path(inputs[0]).suffix.lower() == ".txt"
    ):
        path = inputs[0]
    else:
        raise WorkerError("invalid_worker_input")
    model = parameters.get("model_id", DEFAULT_MODEL)
    if not isinstance(model, str) or model not in MODELS:
        raise WorkerError("invalid_worker_input")
    max_new = integer(parameters.get("max_new_tokens", 128), 1, 512)
    seed = integer(parameters.get("seed", 0), 0, 2**32 - 1)
    temperature = parameters.get("temperature", 0)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise WorkerError("invalid_worker_input")
    stop = parameters.get("stop", [])
    if (
        not isinstance(stop, list)
        or len(stop) > 4
        or any(
            not isinstance(text, str) or not 0 < len(text) <= 64 or "\x00" in text
            for text in stop
        )
    ):
        raise WorkerError("invalid_worker_input")
    if len(stop) != len(set(stop)):
        raise WorkerError("invalid_worker_input")
    return Options(model, prompt, path, max_new, seed, float(temperature), tuple(stop))


def validate_request(request) -> None:
    options(request.inputs, request.parameters)


def read_prompt(settings: Options) -> str:
    if settings.prompt is not None:
        return settings.prompt
    if settings.path is None:
        raise WorkerError("invalid_worker_input")
    try:
        with Path(settings.path).open("rb") as source:
            return decode_prompt(source.read(MAX_PROMPT_BYTES + 1))
    except OSError:
        raise WorkerError("invalid_worker_input") from None


def finish(
    text: str, token_ids: list[int], stops: tuple[str, ...], eos_ids, max_new: int
):
    positions = [text.find(stop) for stop in stops]
    first = min((position for position in positions if position >= 0), default=None)
    if first is not None:
        return text[:first], "stop_string"
    end = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or ())
    if token_ids and token_ids[-1] in end:
        return text, "eos"
    return text, "length" if len(token_ids) >= max_new else "unknown"


def run() -> dict:
    settings = options(*request_fields())
    prompt = read_prompt(settings)
    software = ensure_packages({"transformers": TRANSFORMERS_VERSION})
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
        )
    except (ImportError, OSError, RuntimeError):
        raise WorkerError("worker_dependencies") from None

    device = "cuda"
    cuda_metrics(torch, device)
    revision = MODELS[settings.model_id]
    model_args = {"revision": revision, "trust_remote_code": False}
    started = perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, **model_args)
    except Exception as error:  # noqa: BLE001 - upstream diagnostics may contain secrets.
        raise torch_failure(torch, error, "model_load_failed") from None
    tokenizer_time = (perf_counter() - started) * 1000
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_tokens = encoded["input_ids"].shape[-1]
    if not 0 < input_tokens <= MAX_INPUT_TOKENS:
        raise WorkerError("invalid_worker_input")

    started = perf_counter()
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                settings.model_id,
                **model_args,
                dtype=torch.float16,
                use_safetensors=True,
            )
            .to(device)
            .eval()
        )
        synchronize(torch, device)
    except Exception as error:  # noqa: BLE001 - upstream diagnostics may contain secrets.
        raise torch_failure(torch, error, "model_load_failed") from None
    model_time = (perf_counter() - started) * 1000

    sample = settings.temperature > 0
    generation = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": sample,
        "temperature": settings.temperature if sample else None,
        "top_p": 1.0 if sample else None,
        "top_k": 0 if sample else None,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if settings.stops:
        matcher = StopStringCriteria(tokenizer, list(settings.stops))

        class GeneratedTextStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return matcher(input_ids[:, input_tokens:], scores, **kwargs)

        generation["stopping_criteria"] = StoppingCriteriaList([GeneratedTextStop()])

    try:
        inputs = encoded.to(device)
        torch.manual_seed(settings.seed)
        synchronize(torch, device)
        started = perf_counter()
        with torch.inference_mode():
            sequences = model.generate(**inputs, **generation)
        synchronize(torch, device)
        duration = (perf_counter() - started) * 1000
        tokens = sequences[0, input_tokens:].tolist()
    except (RuntimeError, ValueError) as error:
        raise torch_failure(torch, error, "worker_failed") from None
    raw_text = tokenizer.decode(tokens, skip_special_tokens=True)
    text, reason = finish(
        raw_text,
        tokens,
        settings.stops,
        model.generation_config.eos_token_id,
        settings.max_new_tokens,
    )
    rate = len(tokens) * 1000 / duration if duration > 0 else None
    metrics = {
        "model_id": settings.model_id,
        "model_revision": revision,
        "software": {**software, "torch": str(torch.__version__)},
        "input_tokens": input_tokens,
        "output_tokens": len(tokens),
        "tokenizer_load_time_ms": tokenizer_time,
        "model_load_time_ms": model_time,
        "generation_time_ms": duration,
        "tokens_per_sec": rate if rate is not None and math.isfinite(rate) else None,
        "peak_vram_mb": peak_memory_mib(torch, device),
        "finish_reason": reason,
    }
    write_json(ARTIFACTS[0], {"text": text, **metrics})
    return metrics
