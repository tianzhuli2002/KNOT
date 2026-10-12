#!/usr/bin/env bash
#SBATCH --job-name=knot-seed-smoke
#SBATCH --account=gpu-free
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/seed-pair-v1/smoke-%j.log

set -euo pipefail

frozen_root=/net/scratch/w00238tl/knot
project_root=/net/scratch/w00238tl/knot
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_frozen_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_project_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

[[ "$(git -C "$frozen_root" rev-parse HEAD)" == "$expected_frozen_commit" ]]
[[ -z "$(git -C "$frozen_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$project_root" rev-parse HEAD)" == "$expected_project_commit" ]]
[[ -d "$dependency_path/lgatr" ]]
export PYTHONPATH="$dependency_path:$project_root/src:$project_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$python_bin" -c 'import h5py, lightning, lgatr, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "lgatr", lgatr.__file__)'

output_root="$project_root/results/experiments/knot/seed-pair-smoke-v2"
[[ ! -e "$output_root" ]]

for variant in C J; do
    output="$output_root/${variant}_seed42012"
    /usr/bin/time -v srun "$python_bin" "$project_root/scripts/run_knot.py" train \
        --config "$project_root/configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml" \
        --research "$project_root/configs/research/knot_${variant,,}_production.yaml" \
        --output "$output" \
        --steps 20 \
        --accelerator gpu \
        --smoke \
        --seed-override 42012
    "$python_bin" - "$output/run_manifest.json" "$variant" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
variant = sys.argv[2]
assert manifest["training_status"] == "completed", manifest
assert manifest["variant"] == variant and manifest["seed"] == 42012
assert manifest["fixed_subset_seed"] == 42011
assert manifest["additional_seed_replication"] is True
assert manifest["smoke_validation"]["best_checkpoint_reload"] is True
assert manifest["sealed_test_used"] is False
print(f"KNOT {variant} seed-42012 GPU smoke passed", flush=True)
PY
done

"$python_bin" - "$output_root/C_seed42012/run_manifest.json" "$output_root/J_seed42012/run_manifest.json" <<'PY'
import json
import sys

c, j = (json.load(open(path)) for path in sys.argv[1:])
assert c["membership"] == j["membership"], "C/J smoke membership differs"
assert c["initialization"]["shared_initialization_sha256"] == j["initialization"]["shared_initialization_sha256"], "C/J shared initialization differs at the same seed"
print("Matched C/J smoke membership and shared initialization verified", flush=True)
PY
