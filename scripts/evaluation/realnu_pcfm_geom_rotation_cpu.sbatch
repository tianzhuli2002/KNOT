#!/usr/bin/env bash
#SBATCH --job-name=realnu-geom-rotation
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/rotation/logs/PCFM_geom_untrained_%j.log

set -euo pipefail
source_root=/net/scratch/w00238tl/JointEventReconstruction_parent_geometry_runtime_snapshot_v1
project_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
production_root="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
config="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_geom.public.yaml"
initial_checkpoint="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/initialisations_objective_matched_v5/PCFM_Q_initial.ckpt"
scaler_checkpoint="$production_root/PCFM_Q/checkpoints/best-loss.ckpt"
output="$production_root/evaluation_v4_parent_comparison/rotation/PCFM_geom_untrained_rotation.json"

export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

exec "$python_bin" "$source_root/scripts/evaluation/diagnose_realnu_parent_rotation.py" \
  --config "$config" \
  --checkpoint "$initial_checkpoint" \
  --scaler-checkpoint "$scaler_checkpoint" \
  --initialize-from-shared-checkpoint \
  --output "$output" \
  --context-events 2048 \
  --sample-events 64 \
  --steps 8
