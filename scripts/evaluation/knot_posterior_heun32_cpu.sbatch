#!/usr/bin/env bash
#SBATCH --job-name=knot-j-heun32
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/posterior-convergence-v1/heun32-%j.log

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

"$python_bin" -c 'import h5py, lgatr, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'

output_path="$artifact_root/results/experiments/knot/posterior-convergence-v1/K128-Heun32"
[[ ! -e "$output_path" ]]
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/evaluate_knot_production_checkpoint.py" \
    --variant J \
    --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
    --research "$evaluation_root/configs/research/knot_j_production.yaml" \
    --checkpoint "$checkpoint" \
    --output "$output_path" \
    --events 2000 \
    --samples 128 \
    --inference-steps 32 \
    --inference-solver heun \
    --batch-size 32 \
    --seed 42011 \
    --device cpu

"$python_bin" - "$output_path" "$artifact_root/results/experiments/knot/posterior-convergence-v1" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

output, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
metrics = json.loads((output / "metrics.json").read_text())
manifest = json.loads((output / "evaluation_manifest.json").read_text())
assert metrics["events"] == 2000 and metrics["samples_per_event"] == 128
assert metrics["inference"] == {"steps": 32, "solver": "heun"}
assert manifest["inference_steps"] == 32 and manifest["inference_solver"] == "heun"
assert manifest["sealed_test_used"] is False
with h5py.File(output / "posterior.h5", "r") as data:
    ids = data["EVENT_ID"][:]
    assert data["invisible"].shape == (2000, 128, 2, 4)
    assert data["parents"].shape == (2000, 128, 2, 4)
    assert np.isfinite(data["invisible"][:]).all()
    assert np.isfinite(data["parents"][:]).all()
for name in ("K128-Heun8", "K128-Heun16"):
    with h5py.File(root / name / "posterior.h5", "r") as reference:
        if not np.array_equal(ids, reference["EVENT_ID"][:]):
            raise ValueError(f"Heun-32 event order differs from {name}")
    ref_manifest = json.loads((root / name / "evaluation_manifest.json").read_text())
    if manifest["validation_event_id_sha256"] != ref_manifest["validation_event_id_sha256"]:
        raise ValueError(f"Heun-32 event-ID hash differs from {name}")
print(json.dumps({"status": "validated", "events": len(ids), "samples": 128, "heun_steps": 32}, indent=2))
PY
