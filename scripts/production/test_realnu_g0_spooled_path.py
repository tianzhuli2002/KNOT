#!/usr/bin/env python3
"""Exercise the direct G0 wrapper from an unrelated working directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve(strict=True)
    expected_sha = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_root = project_root / "runtime/source" / expected_sha
    wrapper = source_root / "scripts/production/run_realnu_gpu_g0.sh"
    if not wrapper.is_file():
        raise FileNotFoundError(f"Immutable G0 wrapper is missing: {wrapper}")

    with tempfile.TemporaryDirectory(prefix="realnu-g0-spool-regression-") as temporary:
        root = Path(temporary)
        attempt_root = root / "attempt"
        attempt_root.mkdir()
        g0_root = attempt_root / "G0"
        contract_path = attempt_root / "resolved_execution_contract.json"
        contract_path.write_text(
            json.dumps(
                {
                    "schema": "realnu_g0_execution_contract_v1",
                    "source_sha": expected_sha,
                    "source_root": str(source_root),
                    "project_root": str(project_root),
                    "g0_root": str(g0_root),
                }
            )
        )
        python_stub = root / "python-stub"
        call_log = root / "python-calls.txt"
        python_stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$@\" >> \"$JER_REALNU_WRAPPER_TEST_LOG\"\n"
            "if [[ \"${1:-}\" == '-' ]]; then cat >/dev/null; fi\n"
        )
        python_stub.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "SLURM_JOB_ID": "spool-regression",
                "JER_REALNU_SOURCE_ROOT": str(source_root),
                "JER_REALNU_PROJECT_ROOT": str(project_root),
                "JER_REALNU_EXPECTED_SHA": expected_sha,
                "JER_REALNU_G0_ROOT": str(g0_root),
                "JER_REALNU_PREFLIGHT_CONTRACT": str(contract_path),
                "JER_PYTHON": str(python_stub),
                "JER_REALNU_WRAPPER_TEST_LOG": str(call_log),
            }
        )
        completed = subprocess.run(
            ["/bin/bash", str(wrapper)], cwd=root, env=env,
            check=False, text=True, capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Spool-path wrapper regression failed: {completed.stdout}\n{completed.stderr}"
            )
        calls = call_log.read_text().splitlines()
        if str(source_root / "scripts/production/run_realnu_gpu_g0.py") not in calls:
            raise RuntimeError(f"G0 runner was not resolved under the explicit source root: {calls}")
        if any("/var/spool" in value or str(root / "scripts") in value for value in calls):
            raise RuntimeError(f"G0 wrapper derived its runner from a spool/unrelated path: {calls}")
    print("realnu_g0_spooled_path=PASS")


if __name__ == "__main__":
    main()
