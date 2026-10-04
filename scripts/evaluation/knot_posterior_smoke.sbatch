#!/usr/bin/env bash
#SBATCH --job-name=knot-eval-smoke
#SBATCH --account=gpu-free
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator/logs/knot-evaluation-v1/smoke-%A_%a.log

set -euo pipefail

evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator
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

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
config="$evaluation_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml"
research="$evaluation_root/configs/research/knot_${variant,,}_production.yaml"
manifest="$source_root/results/experiments/knot/matched-production-v1/${variant}/run_manifest.json"
checkpoint=$("$python_bin" - "$manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["best_checkpoint"])
PY
)
output="$evaluation_root/results/experiments/knot/matched-production-v1/evaluation-v1/smoke/${variant}"

/usr/bin/time -v srun "$python_bin" scripts/evaluation/evaluate_knot_production_checkpoint.py \
    --variant "$variant" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --events 64 \
    --samples 4 \
    --batch-size 8 \
    --seed 42011 \
    --device cuda

"$python_bin" - "$output/posterior.h5" "$output/metrics.json" "$variant" <<'PY'
import json
import sys

import h5py
import numpy as np

h5_path, metric_path, variant = sys.argv[1:]
metrics = json.load(open(metric_path))
assert metrics["variant"] == variant
assert metrics["events"] == 64 and metrics["samples_per_event"] == 4
assert metrics["sealed_test_used"] is False
assert np.isfinite(metrics["joint"]["energy_score"]["coherent_GeV"])
with h5py.File(h5_path, "r") as data:
    assert len(data["EVENT_ID"]) == 64
    assert data["invisible"].shape == (64, 4, 2, 4)
    assert data["parents"].shape == (64, 4, 2, 4)
    assert data["measured_b/sample_four_vectors"].shape == (64, 4, 2, 4)
    assert np.isfinite(data["measured_b/sample_four_vectors"][:]).all()
    if variant == "C":
        assert "oracle/truth_assignment_truth_neutrino/parents" in data
        assert "oracle/predicted_assignment_truth_neutrino/parents" in data
print("KNOT production evaluator smoke passed for", variant)
PY
