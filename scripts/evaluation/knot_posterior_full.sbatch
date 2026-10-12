#!/usr/bin/env bash
#SBATCH --job-name=knot-posterior-10k
#SBATCH --account=gpu-free
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/net/scratch/w00238tl/knot/logs/knot-evaluation-v1/full-%A_%a.log

set -euo pipefail

evaluation_root=/net/scratch/w00238tl/knot
source_root=/net/scratch/w00238tl/knot
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_commit="${KNOT_EVAL_EXPECTED_COMMIT:?KNOT_EVAL_EXPECTED_COMMIT is required}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) variant=M ;;
    1) variant=C ;;
    2) variant=J ;;
    *) echo "Unexpected KNOT evaluation array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "$evaluation_root"
[[ "$(git rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -d "$dependency_path/lgatr" ]]
export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$python_bin" -c 'import h5py, lgatr, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "lgatr", lgatr.__file__)'

config="$evaluation_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml"
research="$evaluation_root/configs/research/knot_${variant,,}_production.yaml"
manifest="$source_root/results/experiments/knot/matched-production-v1/${variant}/run_manifest.json"
checkpoint=$("$python_bin" - "$manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["best_checkpoint"])
PY
)
output="$evaluation_root/results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/${variant}"
[[ ! -e "$output" ]]

/usr/bin/time -v srun "$python_bin" scripts/evaluation/evaluate_knot_production_checkpoint.py \
    --variant "$variant" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --events 10000 \
    --samples 32 \
    --batch-size 64 \
    --seed 42011 \
    --device cuda

"$python_bin" - "$output/posterior.h5" "$output/metrics.json" <<'PY'
import json
import sys

import h5py
import numpy as np

posterior_path, metrics_path = sys.argv[1:]
metrics = json.load(open(metrics_path))
assert metrics["events"] == 10000 and metrics["samples_per_event"] == 32
assert metrics["sealed_test_used"] is False
with h5py.File(posterior_path, "r") as data:
    assert len(data["EVENT_ID"]) == 10000
    assert data["invisible"].shape == (10000, 32, 2, 4)
    assert data["parents"].shape == (10000, 32, 2, 4)
    assert data["diagnostics/assignment_truth_NLL"].shape == (10000,)
    assert np.isfinite(data["invisible"][:]).all()
    assert np.isfinite(data["parents"][:]).all()
print("KNOT 10k validation posterior output is finite and complete")
PY
