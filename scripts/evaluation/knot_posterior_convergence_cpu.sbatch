#!/usr/bin/env bash
#SBATCH --job-name=knot-j-convergence
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev/logs/posterior-convergence-v1/convergence-%j.log

set -euo pipefail

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev
artifact_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
checkpoint=/net/scratch/w00238tl/JointEventReconstruction_KNOT/results/experiments/knot/matched-production-v1/J/checkpoints/best-16-1.213281.ckpt
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

for run_spec in "32 8 K32-Heun8" "128 8 K128-Heun8" "128 16 K128-Heun16"; do
    read -r sample_count step_count run_name <<< "$run_spec"
    output_path="$artifact_root/results/experiments/knot/posterior-convergence-v1/$run_name"
    [[ ! -e "$output_path" ]]
    /usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
        "$evaluation_root/scripts/evaluation/evaluate_knot_production_checkpoint.py" \
        --variant J \
        --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
        --research "$evaluation_root/configs/research/knot_j_production.yaml" \
        --checkpoint "$checkpoint" \
        --output "$output_path" \
        --events 2000 \
        --samples "$sample_count" \
        --inference-steps "$step_count" \
        --inference-solver heun \
        --batch-size 32 \
        --seed 42011 \
        --device cpu
    "$python_bin" - "$output_path" "$sample_count" "$step_count" <<'PY'
import json
import sys

import h5py
import numpy as np

output, expected_samples, expected_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
metrics = json.load(open(f"{output}/metrics.json"))
manifest = json.load(open(f"{output}/evaluation_manifest.json"))
assert metrics["events"] == 2000 and metrics["samples_per_event"] == expected_samples
assert metrics["inference"] == {"steps": expected_steps, "solver": "heun"}
assert manifest["inference_steps"] == expected_steps and manifest["inference_solver"] == "heun"
assert manifest["sealed_test_used"] is False
with h5py.File(f"{output}/posterior.h5", "r") as data:
    assert data["EVENT_ID"].shape == (2000,)
    assert data["invisible"].shape == (2000, expected_samples, 2, 4)
    assert data["parents"].shape == (2000, expected_samples, 2, 4)
    assert np.isfinite(data["invisible"][:]).all()
    assert np.isfinite(data["parents"][:]).all()
print(f"Validated {output}: N=2000, K={expected_samples}, Heun={expected_steps}", flush=True)
PY
done

"$python_bin" - "$artifact_root/results/experiments/knot/posterior-convergence-v1" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

root = pathlib.Path(sys.argv[1])
names = ("K32-Heun8", "K128-Heun8", "K128-Heun16")
ids = []
for name in names:
    with h5py.File(root / name / "posterior.h5", "r") as data:
        ids.append(data["EVENT_ID"][:])
if not all(np.array_equal(ids[0], other) for other in ids[1:]):
    raise ValueError("Posterior convergence runs do not use identical ordered EVENT_IDs")
manifests = [json.loads((root / name / "evaluation_manifest.json").read_text()) for name in names]
if any(item["validation_event_id_sha256"] != manifests[0]["validation_event_id_sha256"] for item in manifests[1:]):
    raise ValueError("Posterior convergence event ID hashes differ")
print(json.dumps({"status": "validated", "events": len(ids[0]), "ordered_EVENT_ID_sha256": manifests[0]["validation_event_id_sha256"], "runs": names}, indent=2))
PY
