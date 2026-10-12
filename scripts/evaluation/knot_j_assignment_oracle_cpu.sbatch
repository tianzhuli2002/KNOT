#!/usr/bin/env bash
#SBATCH --job-name=knot-j-asg-oracle
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/j-assignment-oracle-v1/oracle-%j.log

set -euo pipefail

source_root=/net/scratch/w00238tl/knot
evaluation_root=/net/scratch/w00238tl/knot
artifact_root=/net/scratch/w00238tl/knot
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
checkpoint=/net/scratch/w00238tl/knot/results/experiments/knot/matched-production-v1/J/checkpoints/best-16-1.213281.ckpt
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'

output_path="$artifact_root/results/experiments/knot/j-assignment-oracle-v1/full-2000"
[[ ! -e "$output_path" ]]
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/evaluate_knot_production_checkpoint.py" \
    --variant J \
    --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
    --research "$evaluation_root/configs/research/knot_j_production.yaml" \
    --checkpoint "$checkpoint" \
    --output "$output_path" \
    --events 2000 \
    --samples 32 \
    --inference-steps 8 \
    --inference-solver heun \
    --j-assignment-oracle \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

"$python_bin" - "$output_path" "$artifact_root/results/experiments/knot/posterior-convergence-v1/K32-Heun8" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

output, reference = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
metrics = json.loads((output / "metrics.json").read_text())
manifest = json.loads((output / "evaluation_manifest.json").read_text())
assert metrics["events"] == 2000 and metrics["samples_per_event"] == 32
assert metrics["inference"] == {"steps": 8, "solver": "heun"}
assert manifest["j_assignment_oracle"] is True and manifest["sealed_test_used"] is False
oracle = metrics["oracle_ladder"]
assert oracle["supported_events"] == 2000
assert oracle["posterior_draw_noise_matched_to_deployed"] is True
with h5py.File(output / "posterior.h5", "r") as data, h5py.File(reference / "posterior.h5", "r") as ref:
    assert np.array_equal(data["EVENT_ID"][:], ref["EVENT_ID"][:])
    assert data["oracle/truth_assignment/invisible"].shape == (2000, 32, 2, 4)
    assert data["oracle/truth_assignment/parents"].shape == (2000, 32, 2, 4)
    assert np.isfinite(data["oracle/truth_assignment/invisible"][:]).all()
    assert np.isfinite(data["oracle/truth_assignment/parents"][:]).all()
print(json.dumps({"status": "validated", "events": 2000, "samples": 32, "oracle": "truth_assignment", "same_event_ids_as": str(reference)}, indent=2))
PY
