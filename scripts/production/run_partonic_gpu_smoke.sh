#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this smoke inside a deliberately allocated GPU job (SLURM_JOB_ID is unset)." >&2
  exit 2
fi

python_bin=${JER_PYTHON:-python}
experiment=${JER_EXPERIMENT:-configs/experiments/partonic_ttbar2L_2M_p0_gpu_smoke.public.yaml}
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${JER_GPU_SMOKE_OUTPUT:-"results/experiments/partonic_ttbar2L_2M_v1/gpu_smoke_${run_stamp}"}
run_root=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$run_root")
if [[ -e "$run_root" ]]; then
  echo "Refusing to reuse existing GPU smoke output directory: $run_root" >&2
  exit 2
fi
mkdir -p "$run_root"

export JER_PROJECT_ROOT="$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA was requested for the GPU smoke but torch.cuda.is_available() is false")
print(f"CUDA smoke device: {torch.cuda.get_device_name(0)}")
PY

"$python_bin" -m joint_event_reconstruction.cli.run show-config \
  --experiment "$experiment" \
  --set "training.output=$run_root/training" \
  > "$run_root/config.resolved.yaml"
"$python_bin" scripts/data/validate_partonic_contract.py \
  --experiment "$experiment" \
  > "$run_root/preflight.json"

"$python_bin" -m joint_event_reconstruction.cli.run train \
  --experiment "$experiment" \
  --device cuda \
  --set "training.output=$run_root/training" \
  2>&1 | tee "$run_root/train.log"

checkpoint="$run_root/training/checkpoints/best-loss.ckpt"
test -s "$checkpoint"
"$python_bin" scripts/data/validate_partonic_contract.py \
  --experiment "$experiment" \
  --checkpoint "$checkpoint" \
  > "$run_root/checkpoint_contract.json"

prediction="$run_root/validation_parent_samples.h5"
"$python_bin" -m joint_event_reconstruction.cli.run predict-partonic \
  --experiment "$experiment" \
  --checkpoint "$checkpoint" \
  --output "$prediction" \
  --split validation \
  --device cuda \
  --samples 2 \
  --steps 4 \
  --solver heun \
  --seed 42001 \
  --set model.parent_cfm.target_path=null \
  --set data.visible_truth=null \
  --set data.invisible_truth=null \
  --set 'training.trainable=[visible,assignment,queries]' \
  --set training.loss_weights.assignment=0 \
  --set training.loss_weights.partonic_cfm=0 \
  > "$run_root/predict.json"

evaluation="$run_root/validation_evaluation.json"
truth="$repo_root/results/experiments/partonic_ttbar2L_2M_v1/adapted/partonic_ttbar2l_development_maxjets8_v1_supervision.h5"
"$python_bin" -m joint_event_reconstruction.cli.run evaluate-partonic \
  --predictions "$prediction" \
  --truth "$truth" \
  --output "$evaluation" \
  > "$run_root/evaluate.log"
test -s "$prediction"
test -s "$evaluation"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$run_root/gpu_device.csv"
"$python_bin" - "$run_root" "$experiment" "$checkpoint" "$prediction" "$evaluation" "$truth" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
paths = {
    "experiment": Path(sys.argv[2]).resolve(),
    "checkpoint": Path(sys.argv[3]).resolve(),
    "prediction": Path(sys.argv[4]).resolve(),
    "evaluation": Path(sys.argv[5]).resolve(),
    "truth_sidecar": Path(sys.argv[6]).resolve(),
    "source_manifest": Path("results/experiments/partonic_ttbar2L_2M_v1/source_manifest_v1.json").resolve(),
    "split_membership": Path("results/experiments/partonic_ttbar2L_2M_v1/split_membership_v1.npz").resolve(),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
import torch

record = {
    "status": "passed",
    "source_commit": source_sha,
    "source_tree_dirty": dirty,
    "paths": {name: str(path) for name, path in paths.items()},
    "sha256": {name: sha256(path) for name, path in paths.items()},
    "device": torch.cuda.get_device_name(0),
    "conditioning": "P0",
    "prediction_contract": "truth-free measured-input reload",
    "prediction_budget": {"samples": 2, "solver": "heun", "steps": 4, "seed": 42001},
}
destination = run_root / "gpu_smoke_success.json"
temporary = destination.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
print(destination)
PY
