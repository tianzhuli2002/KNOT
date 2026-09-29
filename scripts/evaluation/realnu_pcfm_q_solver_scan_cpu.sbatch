#!/usr/bin/env bash
#SBATCH --job-name=realnu-q-scan-cpu
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=7-00:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/scan/logs/PCFM_Q_scan_cpu_%j.log

set -euo pipefail
source_root=/net/scratch/w00238tl/JointEventReconstruction_parent_geometry_snapshot_v1
project_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
production_root="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
config="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_q.public.yaml"
checkpoint="$production_root/PCFM_Q/checkpoints/best-loss.ckpt"
output="$production_root/evaluation_v4_parent_comparison/scan/PCFM_Q_solver_scan_10k_cpu.h5"

export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA=314bfc465180990a399e29cf63c48be4fb6ecd68
export JER_REALNU_EXPECTED_SHA="$JER_SOURCE_SHA"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

exec "$python_bin" "$source_root/scripts/evaluation/run_realnu_parent_sampling_scan.py" \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --events 10000 \
  --batch-size 64 \
  --samples 32 \
  --steps 4 8 16 32 \
  --device cpu
