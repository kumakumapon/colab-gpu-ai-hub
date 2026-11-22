"""Resolve operations to their registered workers."""

from collections.abc import Callable

from gpu_ai.contracts import JobResult, Operation, TaskRequest

Worker = Callable[[TaskRequest], JobResult]


class WorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[Operation, Worker] = {}

    def register(self, operation: Operation, worker: Worker) -> None:
        operation = Operation(operation)
        if operation in self._workers:
            raise ValueError(f"worker already registered for {operation.value}")
        self._workers[operation] = worker

    def resolve(self, operation: Operation) -> Worker:
        operation = Operation(operation)
        try:
            return self._workers[operation]
        except KeyError:
            raise LookupError(f"no worker registered for {operation.value}") from None
