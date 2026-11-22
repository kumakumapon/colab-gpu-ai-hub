import asyncio
import json
import sys
import wave
from datetime import timedelta
from threading import Event
from uuid import uuid4

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image

from gpu_ai import cli
from gpu_ai.backends.base import MockBackend
from gpu_ai.config import Settings
from gpu_ai.contracts import JobResult, JobStatus
from gpu_ai.mcp_server import create_server
from gpu_ai.runner import JobRunner

TOOLS = {
    "transcribe_audio",
    "extract_text",
    "estimate_depth",
    "llm_infer",
    "get_job_status",
    "get_job_result",
    "cancel_job",
}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    runner = JobRunner(
        MockBackend({"result.txt": b"artifact contents"}),
        Settings(workspace=tmp_path, results_dir=tmp_path / "results"),
    )
    monkeypatch.setattr(cli, "make_runner", lambda _: runner)
    return runner


@pytest.mark.parametrize(
    "operation, path",
    [
        ("transcribe_audio", "sample.wav"),
        ("estimate_depth", "sample.png"),
        ("extract_text", "sample.png"),
        ("llm_infer", "prompt.txt"),
    ],
)
def test_cli_runs_each_worker(operation, path, runner, capsys):
    target = runner.settings.workspace / path
    if path.endswith(".wav"):
        with wave.open(str(target), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x00" * 1600)
    elif path.endswith(".png"):
        Image.new("RGB", (16, 16), "white").save(target)
    else:
        target.write_text("hello")
    option = "--prompt-file" if operation == "llm_infer" else "--input"
    assert cli.main(["run", operation, option, path]) == 0
    result = JobResult.model_validate_json(capsys.readouterr().out)
    assert result.status is JobStatus.COMPLETED
    assert runner.ledger.get(result.job_id).result == result
    assert cli.main(["results", str(result.job_id)]) == 0
    assert JobResult.model_validate_json(capsys.readouterr().out) == result


def test_shared_contract_and_private_errors(runner, capsys, caplog):
    job_id = str(uuid4())
    request = {"prompt": "hello", "job_id": job_id, "max_new_tokens": 8}
    assert (
        cli.main(
            [
                "run",
                "llm_infer",
                "--job-id",
                job_id,
                "--prompt",
                "hello",
                "--parameters",
                '{"max_new_tokens":8}',
            ]
        )
        == 0
    )
    cli_result = JobResult.model_validate_json(capsys.readouterr().out)

    async def scenario():
        async with create_connected_server_and_client_session(
            create_server(runner), read_timeout_seconds=timedelta(seconds=5)
        ) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            assert tools == TOOLS
            for arguments in [
                {"request": {"prompt": "hello", "extra": "credential_value"}},
                {"request": {"inputs": ["../credential_value.txt"]}},
                {"request": {"prompt": "hello"}, "extra": "credential_value"},
                {"request": '{"prompt":"credential_value"}'},
            ]:
                rejected = await client.call_tool("llm_infer", arguments)
                assert rejected.isError and "credential_value" not in str(rejected)
            replay = await client.call_tool("llm_infer", {"request": request})
            assert not replay.isError
            response = await client.call_tool("get_job_result", {"job_id": job_id})
            result = response.structuredContent
            assert result["job_id"] == job_id
            assert result["status"] == cli_result.status
            assert result["cleanup_status"] == cli_result.cleanup_status
            assert result["artifact_paths"][0].endswith("/result.txt")
            assert "artifact contents" not in str(response)

    asyncio.run(scenario())
    assert "credential_value" not in caplog.text


def test_async_submission_and_cli_cancellation(runner, monkeypatch, capsys):
    released = Event()
    original = runner.backend.execute

    def slow_execute(session, request):
        if not released.wait(5):
            raise TimeoutError
        return original(session, request)

    monkeypatch.setattr(runner.backend, "execute", slow_execute)

    async def scenario():
        async with create_connected_server_and_client_session(
            create_server(runner), read_timeout_seconds=timedelta(seconds=5)
        ) as client:
            response = await client.call_tool(
                "llm_infer", {"request": {"prompt": "hello"}}
            )
            assert not response.isError
            job_id = response.structuredContent["job_id"]
            assert response.structuredContent["status"] == "queued"
            assert cli.main(["cancel", job_id]) == 0
            assert json.loads(capsys.readouterr().out)["job_id"] == job_id
            released.set()
            async with asyncio.timeout(5):
                while True:
                    status = await client.call_tool(
                        "get_job_status", {"job_id": job_id}
                    )
                    if status.structuredContent["status"] == "cancelled":
                        assert status.structuredContent["cleanup_status"] == "confirmed"
                        break
                    await asyncio.sleep(0.01)

    try:
        asyncio.run(scenario())
    finally:
        released.set()


def test_stdio_lists_tools(tmp_path):
    async def scenario():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "gpu_ai.mcp_server"], cwd=tmp_path
        )
        async with (
            stdio_client(params) as (reader, writer),
            ClientSession(reader, writer, timedelta(seconds=5)) as client,
        ):
            await client.initialize()
            assert {tool.name for tool in (await client.list_tools()).tools} == TOOLS

    asyncio.run(scenario())
