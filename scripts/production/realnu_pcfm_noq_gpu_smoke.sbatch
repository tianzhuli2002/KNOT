#!/usr/bin/env bash
#SBATCH --job-name=realnu-noq-smoke
#SBATCH --account=gpu-free
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v3_diagnostics/parent_noquery/gpu_smoke_%j.log

set -euo pipefail
source_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries_diagnostics
export JER_REALNU_PROJECT_ROOT=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
export JER_PYTHON=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
cd "$source_root"
exec scripts/production/smoke_realnu_pcfm_noq_gpu.sh
