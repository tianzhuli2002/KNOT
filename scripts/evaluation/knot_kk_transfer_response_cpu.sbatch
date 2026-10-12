#!/usr/bin/env bash
#SBATCH --job-name=knot-kk-response
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev/logs/kk-transfer-v1/response-%j.log

set -euo pipefail

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev
artifact_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator
input_root="$artifact_root/results/experiments/knot/kk-transfer-v1"
truth_root="$artifact_root/results/experiments/knot/kk-contract-v2/mc20d_DSID574801"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
checkpoint="$source_root/results/experiments/knot/matched-production-v1/J/checkpoints/best-16-1.213281.ckpt"
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

mkdir -p "$evaluation_root/logs/kk-transfer-v1"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" && -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, sklearn, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'

posterior="$input_root/transfer-mc20d_DSID574801/posterior.h5"
measured="$truth_root/measured.h5"
truth="$truth_root/truth.h5"
output="$input_root/response-diagnostics-v1/mc20d_DSID574801.json"
[[ -f "$posterior" && -f "$measured" && -f "$truth" && ! -e "$output" ]]

/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/analyze_knot_frozen_transfer.py" \
    --posterior "$posterior" \
    --measured "$measured" \
    --truth "$truth" \
    --output "$output" \
    --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
    --research "$evaluation_root/configs/research/knot_j_production.yaml" \
    --checkpoint "$checkpoint" \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

"$python_bin" - "$output" "$posterior" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

output, posterior_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
report = json.loads(output.read_text())
with h5py.File(posterior_path, "r") as posterior:
    ids = posterior["EVENT_ID"][:]
    assert report["events"] == len(ids)
    assert report["event_id_ordered_sha256"]
assignment = report["assignment"]
assert assignment["status"] == "scored_from_explicit_after_matching_labels"
assert assignment["supported_events"] == 4618
assert assignment["unsupported_events"] == 4103
assert np.isfinite(assignment["truth_assignment_NLL_nats"])
assert np.isfinite(report["mttbar_posterior_mean_response"]["OLS_estimate_vs_truth_slope"])
print(json.dumps({"status": "validated", "events": report["events"], "assignment": assignment["complete_MAP_accuracy"], "mtt_response_slope": report["mttbar_posterior_mean_response"]["OLS_estimate_vs_truth_slope"]}, indent=2))
PY
