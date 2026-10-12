#!/usr/bin/env bash
#SBATCH --job-name=knot-seed-pair
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/net/scratch/w00238tl/knot/logs/seed-pair-v2/full-%A_%a.log

set -euo pipefail

frozen_root=/net/scratch/w00238tl/knot
project_root=/net/scratch/w00238tl/knot
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_frozen_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_project_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44
output_root="$project_root/results/experiments/knot/matched-seed-replication-v1"

case "$SLURM_ARRAY_TASK_ID" in
    0) variant=C; seed=42012 ;;
    1) variant=J; seed=42012 ;;
    2) variant=C; seed=42013 ;;
    3) variant=J; seed=42013 ;;
    *) exit 2 ;;
esac
config="$project_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml"
research="$project_root/configs/research/knot_${variant,,}_production.yaml"
output="$output_root/${variant}_seed${seed}"

[[ "$(git -C "$frozen_root" rev-parse HEAD)" == "$expected_frozen_commit" ]]
[[ -z "$(git -C "$frozen_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$project_root" rev-parse HEAD)" == "$expected_project_commit" ]]
[[ ! -e "$output" && -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$project_root/src:$project_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$python_bin" -c 'import h5py, lightning, lgatr, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "lgatr", lgatr.__file__)'

/usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
    "$project_root/scripts/run_knot.py" train \
    --config "$config" \
    --research "$research" \
    --output "$output" \
    --accelerator gpu \
    --seed-override "$seed"

"$python_bin" - "$output" "$variant" "$seed" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
variant, seed = sys.argv[2], int(sys.argv[3])
manifest = json.loads((output / "run_manifest.json").read_text())
assert manifest["training_status"] == "completed", manifest
assert manifest["variant"] == variant and manifest["seed"] == seed
assert manifest["additional_seed_replication"] is True
assert manifest["fixed_subset_seed"] == 42011
assert manifest["sealed_test_used"] is False
assert pathlib.Path(manifest["best_checkpoint"]).is_file()
assert pathlib.Path(manifest["last_checkpoint"]).is_file()
print(f"KNOT-{variant} full seed-{seed} replication completed", flush=True)
PY
