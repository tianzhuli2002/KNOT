#!/usr/bin/env bash
#SBATCH --job-name=knot-j-probe-mlp
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev/logs/encoder-probes-v3/probe-%j.log

set -euo pipefail

source_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT
evaluation_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaldev
artifact_root=/net/scratch/w00238tl/JointEventReconstruction_KNOT_evaluator
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
checkpoint=/net/scratch/w00238tl/JointEventReconstruction_KNOT/results/experiments/knot/matched-production-v1/J/checkpoints/best-16-1.213281.ckpt
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44
output_path="$artifact_root/results/experiments/knot/encoder-probes-v3/summary.json"
validation_posterior="$artifact_root/results/experiments/knot/posterior-convergence-v1/K32-Heun8/posterior.h5"

[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" && -f "$validation_posterior" ]]
[[ ! -e "$output_path" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'
srun --cpu-bind=cores "$python_bin" \
    "$evaluation_root/scripts/evaluation/probe_knot_frozen_encoder.py" \
    --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
    --research "$evaluation_root/configs/research/knot_j_production.yaml" \
    --checkpoint "$checkpoint" \
    --validation-posterior "$validation_posterior" \
    --output "$output_path" \
    --train-events 5000 \
    --validation-events 2000 \
    --batch-size 64 \
    --posterior-samples 8 \
    --posterior-steps 8 \
    --seed 42011 \
    --device cpu \
    --bootstrap-resamples 1000
