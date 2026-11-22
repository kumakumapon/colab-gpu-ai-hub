"""Opt-in T4 smoke for the public video tool and real Colab file transfer."""

import asyncio
import json
import os
import subprocess
import sys
from datetime import timedelta
from hashlib import sha256
from zipfile import ZipFile

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gpu_ai.backends.colab_cli import ColabCLI
from gpu_ai.config import Settings
from gpu_ai.job_api import make_runner

pytestmark = pytest.mark.colab_live


def test_video_analysis_t4(tmp_path):
    if os.environ.get("CI") or os.environ.get("GPU_AI_T4_SMOKE") != "1":
        pytest.skip("requires explicit GPU_AI_T4_SMOKE=1 outside CI")
    budget = os.environ.get("GPU_AI_T4_SMOKE_MAX_SECONDS", "")
    if not budget.isascii() or not budget.isdecimal() or not 1 <= int(budget) <= 900:
        pytest.fail("set GPU_AI_T4_SMOKE_MAX_SECONDS to a value between 1 and 900")
    budget = int(budget)
    settings = Settings(
        workspace=tmp_path / "workspace",
        results_dir=tmp_path / "results",
        max_runtime_seconds=budget,
    )
    settings.workspace.mkdir()
    source = settings.workspace / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=yellow:size=192x128:rate=5:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-vf",
            "drawbox=x=45:y=35:w=70:h=60:color=blue:t=fill",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
        timeout=30,
    )
    local_config = tmp_path / "settings.toml"
    local_config.write_text(
        f'workspace = "{settings.workspace.as_posix()}"\n'
        f'results_dir = "{settings.results_dir.as_posix()}"\n'
        f"max_runtime_seconds = {budget}\n",
        encoding="utf-8",
    )
    listing = tmp_path / "listing.json"
    colab = ColabCLI()

    def assert_empty():
        response = colab._run("sessions", config=listing)
        assert "[colab] No active sessions found on server." in response.splitlines(), (
            "Colab runtime list is not empty; check the owned job before continuing"
        )

    assert_empty()

    async def execute():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "gpu_ai.mcp_server", "--config", str(local_config)],
            env={
                key: value
                for key, value in os.environ.items()
                if key != "PYTHONPATH" and not key.startswith("GPU_AI_")
            },
        )
        async with (
            stdio_client(params) as (reader, writer),
            ClientSession(reader, writer, timedelta(seconds=30)) as client,
        ):
            await client.initialize()
            job = await client.call_tool(
                "analyze_video",
                {
                    "request": {
                        "inputs": [source.name],
                        "gpu": "T4",
                        "max_frames": 2,
                        "frame_interval_seconds": 1,
                        "objects": [{"object_id": 1, "point": [70, 55]}],
                    }
                },
            )
            assert not job.isError and job.structuredContent
            job_id = job.structuredContent["job_id"]
            print(f"Job {job_id}; evidence directory {tmp_path}", flush=True)
            async with asyncio.timeout(budget + 90):
                while True:
                    status = await client.call_tool(
                        "get_job_status", {"job_id": job_id}
                    )
                    assert not status.isError and status.structuredContent
                    if status.structuredContent["status"] in {
                        "completed",
                        "failed",
                        "cancelled",
                        "cleanup_pending",
                    }:
                        break
                    await asyncio.sleep(3)
            result = await client.call_tool("get_job_result", {"job_id": job_id})
            assert not result.isError and result.structuredContent
            return result.structuredContent

    try:
        result = asyncio.run(execute())
    finally:
        # Recover only sessions recorded by this smoke if the normal stop failed.
        runner = make_runner(settings)
        if runner.reconcile():
            runner.reconcile(retry=True)
        assert_empty()

    assert result["status"] == "completed", result["error"]
    assert result["cleanup_status"] == "confirmed"
    directory = settings.results_dir / result["job_id"]
    manifest = json.loads((directory / "gpu-ai-manifest.json").read_text())
    declared = {item["path"]: item for item in manifest["artifacts"]}
    assert {
        "report.md",
        "transcript.srt",
        "transcript.json",
        "timeline.json",
        "evidence.zip",
    } == set(declared)
    assert {str(directory / name) for name in declared} == set(result["artifact_paths"])
    for name, artifact in declared.items():
        assert sha256((directory / name).read_bytes()).hexdigest() == artifact["sha256"]
    timeline = json.loads((directory / "timeline.json").read_text())
    statuses = {name: value["status"] for name, value in timeline["stages"].items()}
    assert set(statuses.values()) == {"completed"}, timeline["stages"]
    assert timeline["has_audio"]
    assert {e["kind"] for e in timeline["events"]} >= {
        "video_frame",
        "relative_depth",
        "object_mask",
    }
    assert all(
        0 <= e["time_sec"] <= timeline["duration_sec"] for e in timeline["events"]
    )
    with ZipFile(directory / "evidence.zip") as evidence:
        for event in timeline["events"]:
            if event["artifact"]["path"] == "evidence.zip":
                assert event["artifact"]["member"] in evidence.namelist()
    assert "モデル生成の要約（推定）" in (directory / "report.md").read_text()
    print(
        "T4 video stages, SHA-256, automatic cleanup and no remaining runtimes verified."
    )
