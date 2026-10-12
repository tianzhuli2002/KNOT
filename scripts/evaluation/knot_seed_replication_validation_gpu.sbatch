#!/usr/bin/env bash
#SBATCH --job-name=knot-seed-val
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/net/scratch/w00238tl/knot/logs/seed-pair-v2/eval-%A_%a.log

set -euo pipefail

frozen_root=/net/scratch/w00238tl/knot
project_root=/net/scratch/w00238tl/knot
artifact_root=/net/scratch/w00238tl/knot/results/experiments/knot/matched-seed-replication-v1
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_frozen_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_project_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

case "$SLURM_ARRAY_TASK_ID" in
    0) variant=C; seed=42012 ;;
    1) variant=J; seed=42012 ;;
    2) variant=C; seed=42013 ;;
    3) variant=J; seed=42013 ;;
    4) variant=C; seed=42011 ;;
    5) variant=J; seed=42011 ;;
    *) exit 2 ;;
esac
config="$project_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml"
research="$project_root/configs/research/knot_${variant,,}_production.yaml"
if [[ "$seed" == 42011 ]]; then
    run_root="$frozen_root/results/experiments/knot/matched-production-v1/$variant"
else
    run_root="$project_root/results/experiments/knot/matched-seed-replication-v1/${variant}_seed${seed}"
fi
output="$artifact_root/validation-2000/${variant}_seed${seed}"

[[ "$(git -C "$frozen_root" rev-parse HEAD)" == "$expected_frozen_commit" ]]
[[ -z "$(git -C "$frozen_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$project_root" rev-parse HEAD)" == "$expected_project_commit" ]]
[[ -f "$run_root/run_manifest.json" && ! -e "$output" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$project_root/src:$project_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

checkpoint=$("$python_bin" - "$run_root/run_manifest.json" "$run_root" "$variant" "$seed" <<'PY'
import json
import pathlib
import sys

manifest_path, run_root, variant, seed = map(str, sys.argv[1:])
manifest = json.loads(pathlib.Path(manifest_path).read_text())
assert manifest["training_status"] == "completed", manifest
assert manifest["variant"] == variant and manifest["seed"] == int(seed)
assert manifest["sealed_test_used"] is False
if int(seed) == 42011:
    assert manifest.get("additional_seed_replication") is not True
else:
    assert manifest["additional_seed_replication"] is True
    assert manifest["fixed_subset_seed"] == 42011
checkpoint = pathlib.Path(manifest["best_checkpoint"])
if not checkpoint.is_absolute():
    checkpoint = pathlib.Path(run_root) / checkpoint
if not checkpoint.is_file():
    raise FileNotFoundError(f"Completed seed run has no best checkpoint: {checkpoint}")
print(checkpoint)
PY
)

"$python_bin" -c 'import lightning, lgatr, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "lgatr", lgatr.__file__)'
/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$project_root/scripts/evaluation/evaluate_knot_production_checkpoint.py" \
    --variant "$variant" \
    --config "$config" \
    --research "$research" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --events 2000 \
    --samples 32 \
    --batch-size 32 \
    --seed 42011 \
    --device cuda

"$python_bin" - "$output" "$variant" "$seed" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
variant, seed = sys.argv[2], int(sys.argv[3])
metrics = json.loads((output / "metrics.json").read_text())
manifest = json.loads((output / "evaluation_manifest.json").read_text())
assert metrics["variant"] == variant and metrics["events"] == 2000
assert manifest["sealed_test_used"] is False
assert manifest["samples_per_event"] == 32
assert manifest["evaluation_seed"] == 42011
assert manifest["validation_event_id_sha256"]
print(json.dumps({"status": "seed_validation_complete", "variant": variant, "seed": seed, "events": 2000, "event_id_sha256": manifest["validation_event_id_sha256"], "mttbar": metrics["reconstruction"]["mttbar"]}, indent=2), flush=True)
PY
