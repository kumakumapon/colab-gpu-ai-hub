"""Read-only Colab CLI and OAuth diagnostics."""

import platform
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_ai.backends.colab_cli import ColabCLI, ColabCLIError, ErrorCode


class DoctorStatus(StrEnum):
    NOT_INSTALLED = "not_installed"
    UNAUTHENTICATED = "unauthenticated"
    AUTHENTICATED = "authenticated"
    CHECK_FAILED = "check_failed"


@dataclass(frozen=True)
class DoctorReport:
    status: DoctorStatus
    version: str | None = None
    error: ErrorCode | None = None


def diagnose(cli: ColabCLI | None = None) -> DoctorReport:
    cli = cli or ColabCLI()
    version = None
    with TemporaryDirectory() as directory:
        config = Path(directory) / "sessions.json"
        try:
            version = cli.version(config)
            cli.check_auth(config)
        except ColabCLIError as error:
            status = {
                ErrorCode.CLI_MISSING: DoctorStatus.NOT_INSTALLED,
                ErrorCode.AUTH_REQUIRED: DoctorStatus.UNAUTHENTICATED,
            }.get(error.code, DoctorStatus.CHECK_FAILED)
            return DoctorReport(status, version, error.code)
    return DoctorReport(DoctorStatus.AUTHENTICATED, version)


def main() -> int:
    report = diagnose()
    print(f"Colab CLI: {report.version or 'version unavailable'}")
    print(f"OAuth: {report.status.value}")
    if report.error:
        print(f"Diagnostic: {report.error.value}")
    kernel = platform.release().lower()
    if platform.system() == "Linux" and ("microsoft" in kernel or "wsl" in kernel):
        print("WSL: use Ubuntu's colab executable and an unlocked Linux keyring.")
    else:
        print("WSL2 setup: install and run colab inside Ubuntu.")
    print(
        "GPU eligibility: check the Colab account plan, then run the opt-in T4 smoke."
    )
    print("See docs/setup-wsl2.md for setup and the T4 smoke procedure.")
    return 0 if report.status == DoctorStatus.AUTHENTICATED else 1


if __name__ == "__main__":
    raise SystemExit(main())
