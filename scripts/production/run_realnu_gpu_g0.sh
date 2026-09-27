#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Realnu G0 requires its allocated Slurm GPU job." >&2
    exit 2
fi

: "${JER_REALNU_SOURCE_ROOT:?JER_REALNU_SOURCE_ROOT is required}"
: "${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}"
: "${JER_REALNU_EXPECTED_SHA:?JER_REALNU_EXPECTED_SHA is required}"
: "${JER_REALNU_G0_ROOT:?JER_REALNU_G0_ROOT is required}"
: "${JER_REALNU_PREFLIGHT_CONTRACT:?JER_REALNU_PREFLIGHT_CONTRACT is required}"

source_root="$(realpath -e "$JER_REALNU_SOURCE_ROOT")"
project_root="$(realpath -e "$JER_REALNU_PROJECT_ROOT")"
g0_root="$(realpath -m "$JER_REALNU_G0_ROOT")"
contract_path="$(realpath -e "$JER_REALNU_PREFLIGHT_CONTRACT")"
python_bin="${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}"
runner="$source_root/scripts/production/run_realnu_gpu_g0.py"

if [[ "$source_root" != "$project_root/runtime/source/$JER_REALNU_EXPECTED_SHA" ]]; then
    echo "ERROR: G0 source root is not the expected immutable snapshot: $source_root" >&2
    exit 2
fi
if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: configured Python interpreter is missing or not executable: $python_bin" >&2
    exit 2
fi
if [[ ! -f "$runner" ]]; then
    echo "ERROR: G0 runner is missing under the explicit source root: $runner" >&2
    exit 2
fi

git -C "$project_root" rev-parse --verify "${JER_REALNU_EXPECTED_SHA}^{commit}" >/dev/null
if [[ -n "$(git -C "$project_root" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "ERROR: canonical project checkout is dirty" >&2
    exit 2
fi

export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export JER_REALNU_EXPECTED_SHA
export JER_REALNU_G0_ROOT="$g0_root"
export JER_REALNU_PREFLIGHT_CONTRACT="$contract_path"
export JER_PYTHON="$python_bin"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - "$source_root" "$project_root" "$JER_REALNU_EXPECTED_SHA" "$g0_root" "$contract_path" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

source_root, project_root, expected_sha, g0_root, contract_path = map(Path, sys.argv[1:])
metadata_path = source_root / "SOURCE_SNAPSHOT.json"
metadata = json.loads(metadata_path.read_text())
tree_sha = subprocess.check_output(
    ["git", "-C", str(project_root), "rev-parse", f"{expected_sha}^{{tree}}"],
    text=True,
).strip()
contract = json.loads(contract_path.read_text())
if (
    metadata.get("commit_sha") != str(expected_sha)
    or metadata.get("tree_sha") != tree_sha
    or contract.get("source_sha") != str(expected_sha)
    or contract.get("source_branch") != "research/realnu-training-hotpath-v1"
    or Path(contract.get("source_root", "")).resolve() != source_root.resolve()
    or Path(contract.get("project_root", "")).resolve() != project_root.resolve()
    or Path(contract.get("g0_root", "")).resolve() != g0_root.resolve()
):
    raise SystemExit("Resolved G0 contract or immutable source metadata does not match this job")
PY

cd "$project_root"
"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The allocated realnu G0 job cannot access CUDA")
print(f"Realnu G0 CUDA device: {torch.cuda.get_device_name(0)}")
PY

exec "$python_bin" "$runner"
