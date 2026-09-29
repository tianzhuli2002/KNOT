#!/usr/bin/env bash
#SBATCH --job-name=realnu-rotation-metrics
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/rotation/logs/rotation_metrics_%j.log

set -euo pipefail
source_root=/net/scratch/w00238tl/JointEventReconstruction_parent_geometry_runtime_snapshot_v2
project_root=/net/scratch/w00238tl/JointEventReconstruction_topology_queries
production_root="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
comparison_root="$production_root/evaluation_v4_parent_comparison"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
q_config="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_q.public.yaml"
geom_config="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_geom.public.yaml"
q_checkpoint="$production_root/PCFM_Q/checkpoints/best-loss.ckpt"
q_record="$production_root/PCFM_Q/production_training.json"
initial_checkpoint="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/initialisations_objective_matched_v5/PCFM_Q_initial.ckpt"

export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export JER_REALNU_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

q_output="$comparison_root/rotation/PCFM_Q_rotation_metrics_v2.json"
geom_output="$comparison_root/rotation/PCFM_geom_untrained_rotation_metrics_v2.json"
if [[ -e "$q_output" || -e "$geom_output" ]]; then
  echo "Refusing to replace rotation metrics output." >&2
  exit 2
fi

"$python_bin" "$source_root/scripts/evaluation/diagnose_realnu_parent_rotation.py" \
  --config "$q_config" \
  --checkpoint "$q_checkpoint" \
  --record "$q_record" \
  --output "$q_output" \
  --context-events 2048 \
  --sample-events 64 \
  --steps 8

"$python_bin" "$source_root/scripts/evaluation/diagnose_realnu_parent_rotation.py" \
  --config "$geom_config" \
  --checkpoint "$initial_checkpoint" \
  --scaler-checkpoint "$q_checkpoint" \
  --initialize-from-shared-checkpoint \
  --output "$geom_output" \
  --context-events 2048 \
  --sample-events 64 \
  --steps 8
