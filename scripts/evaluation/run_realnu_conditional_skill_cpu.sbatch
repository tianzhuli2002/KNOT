#!/usr/bin/env bash
#SBATCH --job-name=realnu-q-cond-skill
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/conditional_skill_audit_v1/jobs/logs/conditional_skill_%j.log

set -euo pipefail

analysis_root=/net/scratch/w00238tl/JointEventReconstruction_mass_response_audit
project_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
source_root=/net/scratch/w00238tl/JointEventReconstruction_parent_geometry_runtime_snapshot_v2
production_root="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
comparison_root="$production_root/evaluation_v4_parent_comparison"
mass_audit_root="$comparison_root/mass_response_audit_v1"
audit_root="$comparison_root/conditional_skill_audit_v1"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
rotation_json="$audit_root/rotation/PCFM_Q_source_only_rotation.json"
q_config="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_q.public.yaml"
q_checkpoint="$production_root/PCFM_Q/checkpoints/best-loss.ckpt"
q_record="$production_root/PCFM_Q/production_training.json"

if [[ -e "$audit_root/manifest.json" ]]; then
  echo "Refusing to replace an existing conditional-skill audit." >&2
  exit 2
fi
mkdir -p "$audit_root/jobs/logs" "$audit_root/rotation"

export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$analysis_root"
export JER_REALNU_SOURCE_ROOT="$analysis_root"
export PYTHONPATH="$analysis_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg

test -s "$comparison_root/manifest.json"
test -s "$mass_audit_root/cache/PCFM_Q_mass_samples_m32.npz"
test -s "$q_config"
test -s "$q_checkpoint"
test -s "$q_record"

cd "$analysis_root"
"$python_bin" scripts/evaluation/diagnose_realnu_parent_rotation.py \
  --config "$q_config" \
  --checkpoint "$q_checkpoint" \
  --record "$q_record" \
  --output "$rotation_json" \
  --context-events 2048 \
  --sample-events 64 \
  --steps 8

"$python_bin" scripts/evaluation/analyze_realnu_conditional_skill.py \
  --comparison-manifest "$comparison_root/manifest.json" \
  --mass-audit-dir "$mass_audit_root" \
  --rotation-json "$rotation_json" \
  --output-dir "$audit_root" \
  --permutations 32 \
  --bootstrap-replicates 500
