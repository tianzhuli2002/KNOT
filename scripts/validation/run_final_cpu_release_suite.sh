#!/usr/bin/env bash
set -euo pipefail

repo_root=/net/scratch/w00238tl/JointEventReconstruction
experiment_root="$repo_root/results/experiments/vyper_jer_30_4_matched_20260923"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
tiny_input="$experiment_root/validation/tiny_real/ttbar2L_tiny.h5"
parent_target="$experiment_root/data_contract/parent_truth/ttbar2L_parent_hybrid_truth.h5"
visible_checkpoint="$experiment_root/validation/jer_tiny_lgatr_visible_cpu_allocated/checkpoints/best.ckpt"
validation_root="$experiment_root/validation"

if [[ ! -s "$visible_checkpoint" ]]; then
    echo "Missing common visible/assignment smoke checkpoint: $visible_checkpoint" >&2
    exit 2
fi

common_overrides=(
    data.input="$tiny_input"
    data.cache=null
    data.split.indices_path=null
    data.split.require_existing=false
    data.events.train=32
    data.events.validation=16
    data.events.test=16
    training.batch_size=8
    training.workers=0
    training.epochs=1
    training.precision=32-true
)

run_cycle() {
    local config=$1
    local output=$2
    "$python_bin" "$repo_root/scripts/validation/run_tiny_jer_e2e.py" \
        --config "$experiment_root/configs/$config" \
        --output "$validation_root/$output" \
        --device cpu \
        --encoder configured \
        --head none \
        --initial-checkpoint "$visible_checkpoint" \
        --initial-components visible assignment \
        "${common_overrides[@]}"
}

run_cycle jer_reference_factorized.public.yaml jer_tiny_lgatr_factorized_final
run_cycle jer_joint.public.yaml jer_tiny_lgatr_joint_final
run_cycle jer_continued_control.public.yaml jer_tiny_lgatr_continued_final
run_cycle jer_joint_point_head.public.yaml jer_tiny_lgatr_point_final
run_cycle jer_joint_parent_head.public.yaml jer_tiny_lgatr_parent_final
run_cycle jer_joint_both_heads.public.yaml jer_tiny_lgatr_both_final

"$python_bin" "$repo_root/scripts/validation/validate_joint_gradients.py" \
    --config "$experiment_root/configs/jer_tiny_real.public.yaml" \
    --parent-target "$parent_target" \
    --output "$validation_root/gradient_and_optional_head_validation_final.json"

for output in \
    jer_tiny_lgatr_factorized_final \
    jer_tiny_lgatr_joint_final \
    jer_tiny_lgatr_continued_final \
    jer_tiny_lgatr_point_final \
    jer_tiny_lgatr_parent_final \
    jer_tiny_lgatr_both_final; do
    test -s "$validation_root/$output/checkpoints/best.ckpt"
    test -s "$validation_root/$output/predictions.h5"
    test -s "$validation_root/$output/evaluation.json"
    test -s "$validation_root/$output/e2e.json"
done
test -s "$validation_root/gradient_and_optional_head_validation_final.json"
printf '%s\n' "Final CPU release suite passed: $experiment_root"
