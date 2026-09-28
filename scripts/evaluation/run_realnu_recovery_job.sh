#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Realnu R2 stages must run in their allocated Slurm jobs." >&2
    exit 2
fi

: "${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}"
: "${JER_REALNU_TRAINING_SOURCE_ROOT:?JER_REALNU_TRAINING_SOURCE_ROOT is required}"
: "${JER_REALNU_EVALUATION_SOURCE_ROOT:?JER_REALNU_EVALUATION_SOURCE_ROOT is required}"
: "${JER_REALNU_SOURCE_SHA:?JER_REALNU_SOURCE_SHA is required}"
: "${JER_REALNU_EVALUATION_SHA:?JER_REALNU_EVALUATION_SHA is required}"
: "${JER_REALNU_R2_MANIFEST:?JER_REALNU_R2_MANIFEST is required}"
: "${JER_REALNU_R2_ROOT:?JER_REALNU_R2_ROOT is required}"

project_root=$(realpath -e "$JER_REALNU_PROJECT_ROOT")
training_source_root=$(realpath -e "$JER_REALNU_TRAINING_SOURCE_ROOT")
evaluation_source_root=$(realpath -e "$JER_REALNU_EVALUATION_SOURCE_ROOT")
run_manifest=$(realpath -e "$JER_REALNU_R2_MANIFEST")
r2_root=$(realpath -e "$JER_REALNU_R2_ROOT")
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}
stage=${JER_REALNU_RECOVERY_STAGE:?JER_REALNU_RECOVERY_STAGE is required}
mode=${JER_REALNU_RECOVERY_MODE:-}

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: production Python is missing or not executable: $python_bin" >&2
    exit 2
fi
if [[ -n "$(git -C "$project_root" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "ERROR: canonical project checkout is dirty" >&2
    exit 2
fi

export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_REALNU_SOURCE_ROOT="$training_source_root"
export JER_REALNU_EXPECTED_SHA="$JER_REALNU_SOURCE_SHA"
export JER_REALNU_PRODUCTION_SOURCE_SHA="$JER_REALNU_SOURCE_SHA"
export JER_REALNU_PRODUCTION_RUN_MANIFEST="$run_manifest"
export JER_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$training_source_root"
export JER_SOURCE_SHA="$JER_REALNU_SOURCE_SHA"
export JER_SOURCE_BRANCH="research/realnu-training-hotpath-v1"
export PYTHONPATH="$training_source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - "$project_root" "$training_source_root" "$evaluation_source_root" \
    "$JER_REALNU_SOURCE_SHA" "$JER_REALNU_EVALUATION_SHA" "$run_manifest" "$r2_root" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

project, training_source, evaluation_source = map(Path, sys.argv[1:4])
training_sha, evaluation_sha = sys.argv[4:6]
manifest_path, r2_root = Path(sys.argv[6]), Path(sys.argv[7])
manifest = json.loads(manifest_path.read_text())
training_meta = json.loads((training_source / "SOURCE_SNAPSHOT.json").read_text())
evaluation_meta = json.loads((evaluation_source / "SOURCE_SNAPSHOT.json").read_text())
training_tree = subprocess.check_output(
    ["git", "-C", str(project), "rev-parse", f"{training_sha}^{{tree}}"], text=True
).strip()
evaluation_tree = subprocess.check_output(
    ["git", "-C", str(project), "rev-parse", f"{evaluation_sha}^{{tree}}"], text=True
).strip()
if (
    training_source != project / "runtime/source" / training_sha
    or training_meta.get("commit_sha") != training_sha
    or training_meta.get("tree_sha") != training_tree
    or (training_source / ".git").exists()
    or evaluation_source != project / "runtime/source" / evaluation_sha
    or evaluation_meta.get("commit_sha") != evaluation_sha
    or evaluation_meta.get("tree_sha") != evaluation_tree
    or (evaluation_source / ".git").exists()
    or manifest.get("source_commit") != training_sha
    or manifest.get("evaluation_recovery", {}).get("source_sha") != evaluation_sha
    or Path(manifest.get("evaluation_recovery", {}).get("source_root", "")).resolve()
    != evaluation_source.resolve()
    or Path(manifest.get("evaluation_recovery", {}).get("r2_root", "")).resolve()
    != r2_root.resolve()
    or Path(manifest.get("inputs", {}).get("measured_input", "")).resolve().parent
    != (project / "results/experiments/partonic_ttbar2L_2M_realnu_v1/adapted/train_validation").resolve()
    or manifest.get("test_rows_read") != 0
    or manifest.get("inputs", {}).get("reserved_test_rows_materialised") != 0
):
    raise SystemExit("R2 manifest or immutable source snapshot identity is invalid")
PY

case "$stage" in
    validate-r2)
        if [[ -z "$mode" ]]; then
            echo "ERROR: validate-r2 requires JER_REALNU_RECOVERY_MODE" >&2
            exit 2
        fi
        exec "$python_bin" "$evaluation_source_root/scripts/production/run_realnu_production.py" \
            --stage validate-r2 --mode "$mode" --source-sha "$JER_REALNU_SOURCE_SHA" \
            --evaluation-recovery-sha "$JER_REALNU_EVALUATION_SHA" --run-manifest "$run_manifest"
        ;;
    aggregate-r2)
        exec "$python_bin" "$evaluation_source_root/scripts/production/run_realnu_production.py" \
            --stage aggregate-r2 --source-sha "$JER_REALNU_SOURCE_SHA" --run-manifest "$run_manifest"
        ;;
    core-evaluation)
        exec "$python_bin" "$evaluation_source_root/scripts/evaluation/run_realnu_core_evaluation.py" \
            --manifest "$run_manifest" --evaluation-root "$r2_root"
        ;;
    posterior64)
        exec "$python_bin" "$evaluation_source_root/scripts/evaluation/run_realnu_posterior_extension.py" \
            --manifest "$run_manifest" --output "$r2_root/posterior64" --samples 64
        ;;
    final-evaluation)
        exec "$python_bin" "$evaluation_source_root/scripts/evaluation/plot_realnu_campaign.py" \
            --manifest "$run_manifest" --output "$r2_root" \
            --posterior-manifest "$r2_root/posterior64/posterior64_manifest.json"
        ;;
    *)
        echo "ERROR: unknown R2 stage: $stage" >&2
        exit 2
        ;;
esac
