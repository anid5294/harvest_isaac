"""Keep the task's exit status even if native Kit shutdown exits with zero."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def resolve_exit(child_code, status):
    if child_code:
        return 128 - child_code if child_code < 0 else child_code
    if status is None:
        return 1
    return int(status["exit_code"])


def main():
    with tempfile.TemporaryDirectory(prefix="orchard-run-status-") as directory:
        status_file = Path(directory) / "status.json"
        env = dict(os.environ, VLA_RUN_STATUS_FILE=str(status_file))
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("run_env.py")),
                                 *sys.argv[1:]], env=env)
        status = json.loads(status_file.read_text()) if status_file.exists() else None
        code = resolve_exit(result.returncode, status)
        if status is None and result.returncode == 0:
            print("Runner exited without a completion status; treating this as failure.", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
