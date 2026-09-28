#!/usr/bin/env bash
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=fatpou01
#SBATCH --qos=fatpou01
#SBATCH --job-name=rn-v3-regression
#SBATCH --chdir=/net/scratch/w00238tl/JointEventReconstruction_topology_queries_diagnostics
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v3_diagnostics/regression/regression_gate_%j.log

set -euo pipefail

project_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
source_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries_diagnostics
production_root="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
r2_root="$production_root/evaluation_r2"
output_root="$production_root/evaluation_v3_diagnostics/regression/reproduced_2048"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python3.12

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "ERROR: diagnostics source tree must be clean before the regression gate" >&2
    exit 72
fi

source_sha="$(git rev-parse HEAD)"
export JER_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA="$source_sha"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export JER_REALNU_EXPECTED_SHA=cb2b60445499f78421503d4fccc4a43cae1444fd
export PYTHONPATH="$source_root/src:$source_root/scripts/production${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$project_root"
exec "$python_bin" "$source_root/scripts/evaluation/run_realnu_regression_gate.py" \
    --source-root "$source_root" \
    --project-root "$project_root" \
    --production-root "$production_root" \
    --r2-root "$r2_root" \
    --output-root "$output_root" \
    --events 2048 \
    --cpu-threads 6
