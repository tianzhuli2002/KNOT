#!/usr/bin/env bash
#SBATCH --job-name=jer-afterfsr-val
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/jer-afterfsr-v1/validate-%j.log

set -euo pipefail

project_root=/net/scratch/w00238tl/JointEventReconstruction
source_root=/net/scratch/w00238tl/JointEventReconstruction_afterfsr_p5
evaluation_root=/net/scratch/w00238tl/knot
artifact_root=/net/scratch/w00238tl/knot/results/experiments/knot/jer-afterfsr-v1
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_source_sha=040bf9c5f51a07157a768738288c94d43210df18
expected_config_sha=797af8446abebe75f68f21ff14f68213fc58c5153788be9221d40214a5768670
expected_input_sha=e11473a3521dcb1b11cf1d6d1426be72aed6508f70cac96b07d800d9c0e9b2ec
config="$evaluation_root/configs/experiments/jer_afterfsr_p5.public.yaml"
output="$artifact_root/training/P5"
checkpoint="$output/checkpoints/best-loss.ckpt"
prediction="$output/predictions/validation/prediction.h5"
input="$artifact_root/data/jer_charge_side_after_fsr_v4.h5"

mkdir -p "$evaluation_root/logs/jer-afterfsr-v1"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_sha" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ -f "$checkpoint" ]]
[[ -f "$input" ]]
[[ ! -e "$prediction" ]]

export JER_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA="$expected_source_sha"
export JER_PROJECT_ROOT="$project_root"
export PYTHONPATH="$dependency_path:$source_root/src:$source_root:$evaluation_root"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$python_bin" - "$config" "$input" "$checkpoint" "$output/provenance.json" "$expected_source_sha" "$expected_config_sha" "$expected_input_sha" <<'PY'
import hashlib
import json
import pathlib
import sys

import torch
from joint_event_reconstruction.configuration import load_experiment

config_path, input_path, checkpoint_path, provenance_path = map(pathlib.Path, sys.argv[1:5])
expected_sha, expected_config_sha, expected_input_sha = sys.argv[5:8]

def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

config_sha = file_sha256(config_path)
input_sha = file_sha256(input_path)
if config_sha != expected_config_sha:
    raise ValueError(f"P5 validation config SHA differs: {config_sha} != {expected_config_sha}")
if input_sha != expected_input_sha:
    raise ValueError(f"P5 validation v4 input SHA differs: {input_sha} != {expected_input_sha}")
config = load_experiment(config_path)
if int(config.inference.posterior_samples) != 32 or int(config.inference.top_k_assignments) != 1:
    raise ValueError("P5 validation requires the configured top-1, 32-sample posterior")
if int(config.inference.invisible.flow_matching.steps) != 8 or str(config.inference.invisible.flow_matching.solver) != "heun":
    raise ValueError("P5 validation requires the configured Heun-8 flow sampler")
if int(config.inference.seed) != 42011 or config.data.max_events.val is not None:
    raise ValueError("P5 validation must use seed 42011 and the full validation split")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
source_sha = checkpoint.get("hyper_parameters", {}).get("git_sha")
if source_sha != expected_sha:
    raise ValueError(f"P5 validation checkpoint source SHA differs: {source_sha} != {expected_sha}")
provenance = json.loads(provenance_path.read_text())
test_hash = provenance["split_identity"]["index_hashes"]["test"]
if test_hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
    raise ValueError(f"P5 training used sealed test rows: {test_hash}")
print(json.dumps({"checkpoint": str(checkpoint_path), "source_sha": source_sha, "test_index_hash": test_hash, "config_sha256": config_sha, "input_sha256": input_sha, "posterior_samples": 32, "solver": "heun", "steps": 8}, indent=2))
PY

/usr/bin/time -v srun --cpu-bind=cores "$python_bin" -m joint_event_reconstruction.cli.run predict \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split validation \
    --device cuda

"$python_bin" - "$prediction" "$input" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np
from joint_event_reconstruction.prediction_schema import validate_prediction

prediction_path, input_path = map(pathlib.Path, sys.argv[1:3])
validate_prediction(prediction_path)
with h5py.File(input_path, "r") as source:
    event_ids = source["EVENT_ID"][:]
    validation_ids = event_ids[source["SPLIT"][:] == 1]
with h5py.File(prediction_path, "r") as source:
    predicted_ids = source["EVENT_ID"][:]
    roles = json.loads(source.attrs["invisible_roles"])
    samples = source["INVISIBLE/samples"]
    prediction_shape = tuple(samples.shape)
if len(np.unique(predicted_ids)) != len(predicted_ids):
    raise ValueError(f"Validation prediction has duplicate EVENT_ID values: {prediction_path}")
if not np.array_equal(predicted_ids, validation_ids):
    raise ValueError(
        "P5 prediction EVENT_ID sequence differs from the exact v4 validation split: "
        f"predicted={len(predicted_ids)}, expected={len(validation_ids)}"
    )
if roles != ["top.nu", "antitop.nu"]:
    raise ValueError(f"Unexpected charge-side neutrino role order: {roles}")
if prediction_shape != (len(validation_ids), 1, 32, 2, 4):
    raise ValueError(f"Unexpected P5 posterior shape: {prediction_shape}")
print(json.dumps({"status": "exact_validation_prediction_passed", "events": len(predicted_ids), "posterior_samples": 32, "roles": roles, "sample_shape": prediction_shape}, indent=2))
PY

srun --cpu-bind=cores "$python_bin" -m joint_event_reconstruction.cli.run evaluate \
    --config "$config" \
    --split validation \
    --plots all

"$python_bin" - "$output/evaluation/validation/metrics.json" "${prediction}" <<'PY'
import json
import pathlib
import sys

import h5py

metrics_path, prediction_path = map(pathlib.Path, sys.argv[1:3])
metrics = json.loads(metrics_path.read_text())
with h5py.File(prediction_path, "r") as source:
    events = len(source["EVENT_ID"])
if metrics.get("events") != events:
    raise ValueError(f"P5 evaluation event count mismatch: {metrics.get('events')} != {events}")
if metrics.get("posterior_samples") != 32 or metrics.get("top_k_assignments") != 1:
    raise ValueError("P5 evaluation did not use the configured 32-sample, top-1 posterior")
print(json.dumps({"status": "validation_metrics_passed", "events": events, "metrics": str(metrics_path)}, indent=2))
PY
