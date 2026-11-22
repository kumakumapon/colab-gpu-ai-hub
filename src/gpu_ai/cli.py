"""Local command-line entry point for GPU AI Hub."""

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from gpu_ai import doctor
from gpu_ai.config import load_settings
from gpu_ai.contracts import GPU, JobStatus, Operation
from gpu_ai.job_api import OPERATIONS, make_request, make_runner, summarize
from gpu_ai.jobs import JobBusy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-ai")
    parser.add_argument("--config", type=Path, help="Trusted local TOML settings file")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    run = commands.add_parser("run")
    run.add_argument("operation", choices=OPERATIONS)
    run.add_argument(
        "--input", action="append", default=[], help="Workspace-relative path"
    )
    run.add_argument("--prompt-file", help="Workspace-relative UTF-8 prompt file")
    run.add_argument("--prompt")
    run.add_argument(
        "--parameters", default="{}", help="Worker options as a JSON object"
    )
    run.add_argument("--gpu", choices=GPU)
    run.add_argument("--job-id", type=UUID)
    for name in ("status", "results", "cancel"):
        commands.add_parser(name).add_argument("job_id", type=UUID)
    commands.add_parser("reconcile").add_argument("--retry", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        return doctor.main()
    try:
        runner = make_runner(load_settings(args.config))
        ledger = runner.ledger
        if args.command == "run":
            operation = Operation(args.operation)
            parameters = json.loads(args.parameters)
            if (
                not isinstance(parameters, dict)
                or {"inputs", "gpu", "job_id"} & parameters.keys()
            ):
                raise ValueError("invalid parameters")
            if args.prompt_file is not None:
                if operation != Operation.LLM_INFER or args.input:
                    raise ValueError("invalid prompt file")
                args.input = [args.prompt_file]
            if args.prompt is not None:
                if "prompt" in parameters:
                    raise ValueError("duplicate prompt")
                parameters["prompt"] = args.prompt
            data = dict(parameters, inputs=args.input, gpu=args.gpu)
            if args.job_id is not None:
                data["job_id"] = args.job_id
            model, _, _ = OPERATIONS[operation]
            result = runner.run(
                make_request(operation, model.model_validate(data), runner.settings)
            )
            print(result.model_dump_json())
            return int(result.status != JobStatus.COMPLETED)
        if args.command == "reconcile":
            print(
                json.dumps(
                    [
                        summarize(record, runner.settings.results_dir).model_dump(
                            mode="json"
                        )
                        for record in runner.reconcile(retry=args.retry)
                    ]
                )
            )
            return 0
        record = ledger.get(args.job_id)
        if record is None:
            print('{"error":"job_not_found"}', file=sys.stderr)
            return 1
        if args.command == "cancel":
            record = ledger.request_cancel(args.job_id)
        if args.command == "results" and record.result is not None:
            print(record.result.model_dump_json())
        else:
            print(summarize(record, runner.settings.results_dir).model_dump_json())
        return 0
    except JobBusy:
        print('{"error":"job_busy"}', file=sys.stderr)
    except (ValueError, OSError):
        print('{"error":"invalid_request_or_configuration"}', file=sys.stderr)
    return 1
