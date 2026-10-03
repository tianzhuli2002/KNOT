#!/usr/bin/env bash
#SBATCH --job-name=knot-matched-v1
#SBATCH --account=gpu-free
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT/results/experiments/knot/matched-production-v1/slurm/train-%A_%a.log

set -euo pipefail

project_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
expected_commit="${KNOT_EXPECTED_COMMIT:?KNOT_EXPECTED_COMMIT is required}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) variant=M ;;
    1) variant=C ;;
    2) variant=J ;;
    *) echo "Unexpected KNOT array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "$project_root"
[[ "$(git rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -d "$dependency_path/lgatr" ]]
export PYTHONPATH="$dependency_path:$project_root/src:$project_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$python_bin" -c 'import h5py, lightning, lgatr, torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "lgatr", lgatr.__file__)'

output="$project_root/results/experiments/knot/matched-production-v1/${variant}"
/usr/bin/time -v srun "$python_bin" scripts/run_knot.py train \
    --config "configs/experiments/knot_scientific_closure/knot_${variant,,}_production.public.yaml" \
    --research "configs/research/knot_${variant,,}_production.yaml" \
    --output "$output" \
    --accelerator gpu

test -s "$output/run_manifest.json"
"$python_bin" - "$output/run_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
assert manifest["training_status"] == "completed", manifest
assert manifest["sealed_test_used"] is False
print("KNOT matched production completed for", manifest["variant"])
PY
