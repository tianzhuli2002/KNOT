#!/usr/bin/env bash
#SBATCH --job-name=jer-afterfsr-smoke
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/jer-afterfsr-v1/smoke-%j.log

set -euo pipefail

project_root=/net/scratch/w00238tl/JointEventReconstruction
source_root=/net/scratch/w00238tl/JointEventReconstruction_afterfsr_p5
evaluation_root=/net/scratch/w00238tl/knot
artifact_root=/net/scratch/w00238tl/knot/results/experiments/knot/jer-afterfsr-v1
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_source_sha=040bf9c5f51a07157a768738288c94d43210df18

mkdir -p "$evaluation_root/logs/jer-afterfsr-v1"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_sha" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ -f "$artifact_root/data/jer_charge_side_after_fsr_v4.h5" ]]

export JER_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA="$expected_source_sha"
export PYTHONPATH="$dependency_path:$source_root/src:$source_root:$evaluation_root"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'

output="$artifact_root/training-smoke/P5-v4"
[[ ! -e "$output" ]]
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" -m joint_event_reconstruction.cli.run train \
    --config "$evaluation_root/configs/experiments/jer_afterfsr_p5.public.yaml" \
    --device cuda \
    --set "training.output=$output" \
    --set training.epochs=1 \
    --set training.workers=0 \
    --set data.events.train=512 \
    --set data.events.validation=128

"$python_bin" - "$output" "$expected_source_sha" <<'PY'
import json
import pathlib
import sys

import torch

output, expected_sha = pathlib.Path(sys.argv[1]), sys.argv[2]
for path in (output / "checkpoints/best-loss.ckpt", output / "config.resolved.yaml", output / "architecture.json", output / "inputs.json"):
    if not path.is_file():
        raise FileNotFoundError(f"P5 GPU smoke did not write required artifact: {path}")
checkpoint = torch.load(output / "checkpoints/best-loss.ckpt", map_location="cpu", weights_only=False)
source_sha = checkpoint.get("hyper_parameters", {}).get("git_sha")
if source_sha != expected_sha:
    raise ValueError(f"Smoke checkpoint source SHA differs: {source_sha} != {expected_sha}")
scores = [
    callback["best_model_score"]
    for callback in checkpoint.get("callbacks", {}).values()
    if isinstance(callback, dict) and "best_model_score" in callback
]
if not scores or not all(torch.isfinite(torch.as_tensor(score)).all() for score in scores):
    raise ValueError(f"P5 smoke did not save a finite validation loss: {scores}")
if not all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values() if torch.is_floating_point(value)):
    raise ValueError("P5 smoke checkpoint contains non-finite model tensors")
if not all(torch.isfinite(value).all() for state in checkpoint.get("optimizer_states", []) for entry in state.get("state", {}).values() for value in entry.values() if torch.is_tensor(value) and torch.is_floating_point(value)):
    raise ValueError("P5 smoke checkpoint contains non-finite optimizer state")
provenance = json.loads((output / "provenance.json").read_text())
test_hash = provenance["split_identity"]["index_hashes"]["test"]
if test_hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
    raise ValueError(f"P5 smoke input contains sealed test rows: {test_hash}")
print(json.dumps({"status": "gpu_smoke_training_and_checkpoint_passed", "output": str(output), "source_sha": source_sha, "checkpoint_step": checkpoint.get("global_step")}, indent=2))
PY

srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/production/check_jer_afterfsr_p5_gradients.py" \
    --config "$output/config.public.yaml" \
    --checkpoint "$output/checkpoints/best-loss.ckpt" \
    --set "training.output=$output" \
    --set training.epochs=1 \
    --set training.workers=0 \
    --set data.events.train=512 \
    --set data.events.validation=128
