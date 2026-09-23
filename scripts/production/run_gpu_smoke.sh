#!/usr/bin/env bash
set -euo pipefail

repo_root=/net/scratch/w00238tl/JointEventReconstruction
vyper_root=/net/scratch/w00238tl/VyPER_matched_upstream
experiment_root="$repo_root/results/experiments/vyper_jer_30_4_matched_20260923"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python
vyper_python=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER/bin/python
manifest="$experiment_root/data_contract/matched_event_manifest.h5"
jer_config="$experiment_root/configs/jer_tiny_real.public.yaml"
vyper_config="$vyper_root/configs/matched_30_4_tiny_actual.yaml"
gpu_root="$experiment_root/validation/gpu_smoke"
jer_output="$gpu_root/jer_joint"
vyper_output="$gpu_root/vyper"
vyper_prediction="$vyper_output/predictions.h5"

expected_source_sha=${EXPECTED_JER_SOURCE_SHA:?EXPECTED_JER_SOURCE_SHA is required}
expected_manifest_sha=${EXPECTED_MANIFEST_SHA256:?EXPECTED_MANIFEST_SHA256 is required}
expected_vyper_sha=${EXPECTED_VYPER_SOURCE_SHA:?EXPECTED_VYPER_SOURCE_SHA is required}
expected_jer_config_sha=${EXPECTED_JER_CONFIG_SHA256:?EXPECTED_JER_CONFIG_SHA256 is required}
expected_vyper_config_sha=${EXPECTED_VYPER_CONFIG_SHA256:?EXPECTED_VYPER_CONFIG_SHA256 is required}

[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$expected_source_sha" ]]
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$expected_manifest_sha" ]]
[[ "$(sha256sum "$jer_config" | cut -d' ' -f1)" == "$expected_jer_config_sha" ]]
[[ "$(git -C "$vyper_root" rev-parse HEAD)" == "$expected_vyper_sha" ]]
[[ "$(sha256sum "$vyper_config" | cut -d' ' -f1)" == "$expected_vyper_config_sha" ]]

if [[ -e "$gpu_root" ]]; then
    if find "$gpu_root" -type f -print -quit | grep -q .; then
        echo "Refusing to reuse non-empty GPU smoke root: $gpu_root" >&2
        exit 3
    fi
else
    mkdir -p "$gpu_root"
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$gpu_root/start_utc.txt"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv > "$gpu_root/gpu_before.csv"

set +e
/usr/bin/time -v "$python_bin" "$repo_root/scripts/validation/run_tiny_jer_e2e.py" \
    --config "$jer_config" \
    --output "$jer_output" \
    --device gpu \
    --encoder configured \
    --head none \
    --set data.input="$experiment_root/validation/tiny_real/ttbar2L_tiny.h5" \
    --set data.cache=null \
    --set data.split.indices_path=null \
    --set data.split.require_existing=false \
    --set data.events.train=32 \
    --set data.events.validation=16 \
    --set data.events.test=16 \
    --set training.batch_size=8 \
    --set training.workers=0 \
    --set training.epochs=1 \
    --set training.precision=32-true \
    > "$gpu_root/jer_train_predict.log" 2>&1
jer_status=$?
set -e
if [[ $jer_status -ne 0 ]]; then
    cat "$gpu_root/jer_train_predict.log"
    exit "$jer_status"
fi

set +e
/usr/bin/time -v "$vyper_python" -m VyPER.train \
    --config-name=matched_30_4_tiny_actual \
    device.accelerator=gpu \
    device.num_workers=1 \
    training.save_directory="$vyper_output" \
    predicting.model_directory="$vyper_output/version_0" \
    predicting.save_as="$vyper_prediction" \
    > "$gpu_root/vyper_train.log" 2>&1
vyper_train_status=$?
set -e
if [[ $vyper_train_status -ne 0 ]]; then
    cat "$gpu_root/vyper_train.log"
    exit "$vyper_train_status"
fi

# PyTorch 2.11 rejects the trusted Lightning checkpoint's defaultdict when
# the pinned upstream prediction entry point passes weights_only=True.  Keep
# the pinned upstream architecture and sampler unchanged; apply a scoped
# loader-only compatibility patch for this smoke and restore the source on
# every exit path.
vyper_predict_source="$vyper_root/VyPER/predict.py"
vyper_predict_backup="$gpu_root/VyPER_predict.py.before_compatibility_patch"
cp "$vyper_predict_source" "$vyper_predict_backup"
restore_vyper_predict() {
    cp "$vyper_predict_backup" "$vyper_predict_source"
}
trap restore_vyper_predict EXIT
"$python_bin" - "$vyper_predict_source" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "weights_only = True # Required since PyTorch 2.9."
new = "weights_only = False # Trusted local checkpoint; PyTorch 2.11 compatibility."
if text.count(old) != 1:
    raise SystemExit(f"expected one pinned weights_only loader line in {path}")
path.write_text(text.replace(old, new))
PY
printf '%s\n' "temporary VyPER predict.py weights_only=False compatibility patch; restored by EXIT trap" > "$gpu_root/vyper_loader_compatibility.txt"

set +e
/usr/bin/time -v "$vyper_python" -m VyPER.predict \
    --config-name=matched_30_4_tiny_actual \
    device.accelerator=gpu \
    device.num_workers=1 \
    predicting.model_directory="$vyper_output/version_0" \
    predicting.save_as="$vyper_prediction" \
    > "$gpu_root/vyper_predict.log" 2>&1
vyper_predict_status=$?
set -e
if [[ $vyper_predict_status -ne 0 ]]; then
    cat "$gpu_root/vyper_predict.log"
    exit "$vyper_predict_status"
fi

"$vyper_python" "$repo_root/scripts/evaluation/adapt_vyper_complete_assignment.py" \
    --prediction "$vyper_prediction" \
    --view "$experiment_root/validation/vyper_tiny_views_jet8/test.h5" \
    --output "$gpu_root/vyper_assignment.json"

test -s "$jer_output/checkpoints/best.ckpt"
test -s "$jer_output/predictions.h5"
test -s "$jer_output/evaluation.json"
test -s "$vyper_prediction"
test -s "$gpu_root/vyper_assignment.json"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv > "$gpu_root/gpu_after.csv"
date -u +%Y-%m-%dT%H:%M:%SZ > "$gpu_root/end_utc.txt"
printf '%s\n' "GPU smoke passed: $gpu_root"
