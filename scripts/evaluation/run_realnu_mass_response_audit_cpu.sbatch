#!/usr/bin/env bash
#SBATCH --job-name=realnu-q-massdiag
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/mass_response_audit_v1/jobs/logs/PCFM_Q_mass_response_%j.log

set -euo pipefail

analysis_root=/net/scratch/w00238tl/JointEventReconstruction_mass_response_audit
production_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2
comparison_root="$production_root/evaluation_v4_parent_comparison"
audit_root="$comparison_root/mass_response_audit_v1"
manifest="$comparison_root/manifest.json"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg

test -s "$manifest"
cd "$analysis_root"
"$python_bin" scripts/evaluation/analyze_realnu_mass_response.py \
  --comparison-manifest "$manifest" \
  --output-dir "$audit_root" \
  --bootstrap-replicates 500
