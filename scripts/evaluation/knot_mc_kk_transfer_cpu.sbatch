#!/usr/bin/env bash
#SBATCH --job-name=knot-mc-transfer
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --array=0-3%2
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev/logs/kk-transfer-v2/mc-%A_%a.log

set -euo pipefail

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev
artifact_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator
input_root="$artifact_root/results/experiments/knot/kk-contract-v2"
output_root="$artifact_root/results/experiments/knot/kk-transfer-v1"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

case "$SLURM_ARRAY_TASK_ID" in
    0) variant=M; sample=mc20d_DSID574801 ;;
    1) variant=M; sample=mc23a_DSID601230 ;;
    2) variant=C; sample=mc20d_DSID574801 ;;
    3) variant=C; sample=mc23a_DSID601230 ;;
    *) exit 2 ;;
esac
config="$evaluation_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml"
research="$evaluation_root/configs/research/knot_${variant,,}_production.yaml"
if [[ "$variant" == M ]]; then
    checkpoint="$source_root/results/experiments/knot/matched-production-v1/M/checkpoints/best-19-1.330055.ckpt"
else
    checkpoint="$source_root/results/experiments/knot/matched-production-v1/C/checkpoints/best-16-1.224542.ckpt"
fi
sample_root="$input_root/$sample"
output_path="$output_root/transfer-$variant-$sample"
response_path="$output_root/response-diagnostics-v2/${variant}_${sample}.json"

[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" && -f "$sample_root/measured.h5" && -f "$sample_root/truth.h5" ]]
[[ ! -e "$output_path" && ! -e "$response_path" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, sklearn, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/evaluate_knot_frozen_transfer.py" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --input "$sample_root/measured.h5" \
    --truth "$sample_root/truth.h5" \
    --output "$output_path" \
    --samples 32 \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/analyze_knot_frozen_transfer.py" \
    --posterior "$output_path/posterior.h5" \
    --measured "$sample_root/measured.h5" \
    --truth "$sample_root/truth.h5" \
    --output "$response_path" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

"$python_bin" - "$output_path" "$response_path" "$sample_root" "$variant" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

output, response, sample_root = map(pathlib.Path, sys.argv[1:4])
variant = sys.argv[4]
metrics = json.loads((output / "metrics.json").read_text())
manifest = json.loads((output / "evaluation_manifest.json").read_text())
diagnostics = json.loads(response.read_text())
with h5py.File(sample_root / "measured.h5", "r") as measured, h5py.File(output / "posterior.h5", "r") as posterior:
    expected = measured["EVENT_ID"][:]
    assert np.array_equal(posterior["EVENT_ID"][:], expected)
    assert posterior["invisible"].shape == (len(expected), 32, 2, 4)
    assert posterior["parents"].shape == (len(expected), 32, 2, 4)
    assert np.isfinite(posterior["invisible"][:]).all()
    assert np.isfinite(posterior["parents"][:]).all()
assert metrics["variant"] == variant and metrics["events"] == len(expected)
assert manifest["variant"] == variant and manifest["truth_used_during_inference"] is False
assert diagnostics["events"] == len(expected)
if "574801" in str(sample_root):
    assert diagnostics["assignment"]["status"] == "scored_from_explicit_after_matching_labels"
    assert diagnostics["assignment"]["supported_events"] == 4618
else:
    assert diagnostics["assignment"]["status"] == "unavailable"
print(json.dumps({"status": "validated", "variant": variant, "events": len(expected), "assignment": diagnostics["assignment"]["status"], "mtt_bias_GeV": diagnostics["mttbar_posterior_mean_response"]["bias_GeV"]}, indent=2), flush=True)
PY
