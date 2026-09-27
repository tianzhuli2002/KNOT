#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Realnu production stages must run in their allocated Slurm jobs." >&2
    exit 2
fi

: "${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}"
: "${JER_REALNU_SOURCE_ROOT:?JER_REALNU_SOURCE_ROOT is required}"
: "${JER_REALNU_PRODUCTION_SOURCE_SHA:?JER_REALNU_PRODUCTION_SOURCE_SHA is required}"
: "${JER_REALNU_PRODUCTION_RUN_MANIFEST:?JER_REALNU_PRODUCTION_RUN_MANIFEST is required}"
stage=${JER_REALNU_PRODUCTION_STAGE:?JER_REALNU_PRODUCTION_STAGE is required}
project_root=$(realpath -e "$JER_REALNU_PROJECT_ROOT")
source_root=$(realpath -e "$JER_REALNU_SOURCE_ROOT")
source_sha=$JER_REALNU_PRODUCTION_SOURCE_SHA
run_manifest=$(realpath -e "$JER_REALNU_PRODUCTION_RUN_MANIFEST")
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}
runner="$source_root/scripts/production/run_realnu_production.py"

if [[ "$source_root" != "$project_root/runtime/source/$source_sha" ]]; then
    echo "ERROR: production source root is not the expected immutable snapshot: $source_root" >&2
    exit 2
fi
if [[ ! -x "$python_bin" || ! -f "$runner" ]]; then
    echo "ERROR: production Python or runner is missing: python=$python_bin runner=$runner" >&2
    exit 2
fi
if [[ -n "$(git -C "$project_root" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "ERROR: canonical project checkout is dirty" >&2
    exit 2
fi
git -C "$project_root" rev-parse --verify "${source_sha}^{commit}" >/dev/null

export JER_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA="$source_sha"
export JER_SOURCE_BRANCH="research/realnu-training-hotpath-v1"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export JER_REALNU_EXPECTED_SHA="$source_sha"
export JER_REALNU_PRODUCTION_RUN_MANIFEST="$run_manifest"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - "$source_root" "$project_root" "$source_sha" "$run_manifest" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

source_root, project_root, source_sha, manifest_path = map(Path, sys.argv[1:])
metadata = json.loads((source_root / "SOURCE_SNAPSHOT.json").read_text())
tree_sha = subprocess.check_output(
    ["git", "-C", str(project_root), "rev-parse", f"{source_sha}^{{tree}}"],
    text=True,
).strip()
manifest = json.loads(manifest_path.read_text())
if (
    metadata.get("commit_sha") != str(source_sha)
    or metadata.get("tree_sha") != tree_sha
    or (source_root / ".git").exists()
    or manifest.get("source_commit") != str(source_sha)
    or Path(manifest.get("source_root", "")).resolve() != source_root.resolve()
):
    raise SystemExit("Production manifest or immutable source snapshot identity is invalid")
PY

cd "$project_root"
if [[ "$stage" == "train" || "$stage" == "validate" ]]; then
    "$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Allocated realnu production GPU is unavailable")
print(f"Realnu production CUDA device: {torch.cuda.get_device_name(0)}")
PY
fi

args=(
    "$runner"
    --stage "$stage"
    --source-sha "$source_sha"
    --run-manifest "$run_manifest"
    --python "$python_bin"
)
if [[ "$stage" == "train" || "$stage" == "validate" ]]; then
    args+=(--mode "${JER_REALNU_PRODUCTION_MODE:?JER_REALNU_PRODUCTION_MODE is required}")
fi
exec "$python_bin" "${args[@]}"
