"""Local stdio MCP server for allowlisted Colab workers."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Lock
from uuid import UUID

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from gpu_ai.config import load_settings
from gpu_ai.contracts import JobResult, JobStatus, Operation
from gpu_ai.job_api import (
    OPERATIONS,
    JobSummary,
    WorkerInput,
    make_request,
    make_runner,
    summarize,
)
from gpu_ai.jobs import JobBusy
from gpu_ai.runner import JobRunner


class LocalMCP(FastMCP):
    async def call_tool(self, name, arguments):
        # FastMCP accepts additional outer keys and its default errors can echo inputs.
        worker = name in OPERATIONS
        fields = {"request"} if worker else {"job_id"}
        if (
            not isinstance(arguments, dict)
            or arguments.keys() != fields
            or (worker and not isinstance(arguments["request"], dict))
        ):
            raise ToolError("invalid_arguments")
        try:
            return await super().call_tool(name, arguments)
        except Exception:  # noqa: BLE001 - never return unfiltered SDK/tool errors.
            raise ToolError("tool_request_failed") from None


def create_server(runner: JobRunner) -> FastMCP:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-ai")
    lock = Lock()
    active: Future[JobResult] | None = None
    cancel = Event()

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield
        finally:
            cancel.set()
            executor.shutdown(wait=True)

    server = LocalMCP("gpu-ai", lifespan=lifespan)

    def submission(operation: Operation, model: type[WorkerInput]):
        def submit(request: WorkerInput) -> JobSummary:
            nonlocal active
            task = make_request(operation, request, runner.settings)
            with lock:
                if active is not None and not active.done():
                    raise JobBusy("another submission is active")
                cancel.clear()
                queued = Event()
                active = executor.submit(runner.run, task, cancel=cancel, queued=queued)
                active.add_done_callback(lambda _: queued.set())
                queued.wait()
                if active.done():
                    active.result()
            return JobSummary(job_id=task.job_id, status=JobStatus.QUEUED)

        submit.__annotations__["request"] = model
        return submit

    for operation, (model, _, description) in OPERATIONS.items():
        server.add_tool(
            submission(operation, model),
            name=operation.value,
            description=f"{description} Submit an asynchronous job. Poll get_job_status.",
        )

    def get_job_status(job_id: UUID) -> JobSummary:
        """Read the current status and cleanup state of a submitted job."""
        record = runner.ledger.get(job_id)
        if record is None:
            raise LookupError("job_not_found")
        return summarize(record, runner.settings.results_dir)

    def get_job_result(job_id: UUID) -> JobSummary:
        """Get verified local artifact paths for a job."""
        return get_job_status(job_id)

    def cancel_job(job_id: UUID) -> JobSummary:
        """Request cancellation at the next job checkpoint. Poll for cleanup."""
        record = runner.ledger.request_cancel(job_id)
        return summarize(record, runner.settings.results_dir)

    for tool in (get_job_status, get_job_result, cancel_job):
        server.add_tool(tool)
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        runner = make_runner(load_settings(args.config))
    except (ValueError, OSError):
        parser.exit(1, "gpu-ai: invalid local configuration\n")
    create_server(runner).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
