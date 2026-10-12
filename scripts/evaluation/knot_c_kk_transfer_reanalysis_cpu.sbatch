#!/usr/bin/env bash
#SBATCH --job-name=knot-c-kk-diagnostic
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev/logs/kk-transfer-v2/c-response-%A_%a.log

set -euo pipefail

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev
artifact_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator/results/experiments/knot
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

case "$SLURM_ARRAY_TASK_ID" in
    0) sample=mc20d_DSID574801 ;;
    1) sample=mc23a_DSID601230 ;;
    *) exit 2 ;;
esac
config="$evaluation_root/configs/experiments/knot_scientific_closure/knot_c_production.public.yaml"
research="$evaluation_root/configs/research/knot_c_production.yaml"
checkpoint="$source_root/results/experiments/knot/matched-production-v1/C/checkpoints/best-16-1.224542.ckpt"
sample_root="$artifact_root/kk-contract-v2/$sample"
transfer_root="$artifact_root/kk-transfer-v1/transfer-C-$sample"
response_path="$artifact_root/kk-transfer-v1/response-diagnostics-v2/C_${sample}.json"

[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" && -f "$sample_root/measured.h5" && -f "$sample_root/truth.h5" ]]
[[ -f "$transfer_root/posterior.h5" && ! -e "$response_path" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/analyze_knot_frozen_transfer.py" \
    --posterior "$transfer_root/posterior.h5" \
    --measured "$sample_root/measured.h5" \
    --truth "$sample_root/truth.h5" \
    --output "$response_path" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

"$python_bin" - "$response_path" "$sample_root" <<'PY'
import json
import pathlib
import sys

response_path, sample_root = map(pathlib.Path, sys.argv[1:])
response = json.loads(response_path.read_text())
contract = json.loads((sample_root / "contract.json").read_text())
expected_events = contract["events"]["direct_charge_matched_e_mu_written"]
assert response["events"] == expected_events
if "574801" in str(sample_root):
    assert response["assignment"]["status"] == "scored_from_explicit_after_matching_labels"
    assert response["assignment"]["supported_events"] == contract["visible_assignment"]["supported_events"]
else:
    assert response["assignment"]["status"] == "unavailable"
print(json.dumps({"status": "validated", "events": response["events"], "assignment": response["assignment"]["status"], "mtt_bias_GeV": response["mttbar_posterior_mean_response"]["bias_GeV"]}, indent=2), flush=True)
PY
