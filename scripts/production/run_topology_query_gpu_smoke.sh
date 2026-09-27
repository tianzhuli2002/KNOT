#!/usr/bin/env bash
set -euo pipefail

# Isolated GPU execution gate for the topology-query research branch.  This
# script deliberately uses only the synthetic truth-labelled fixture and never
# touches the matched VyPER--JER production root.
source_root="$(realpath -e "${JER_TOPOLOGY_SOURCE_ROOT:?JER_TOPOLOGY_SOURCE_ROOT is required}")"
project_root="$(realpath -e "${JER_TOPOLOGY_PROJECT_ROOT:?JER_TOPOLOGY_PROJECT_ROOT is required}")"
experiment_root="$project_root/results/experiments/topology_query_joint_20260923"
smoke_root="$experiment_root/validation/gpu_smoke_topology_query"
config="$source_root/configs/experiments/topology_query_synthetic_full.public.yaml"
measured="$experiment_root/synthetic_data/measured_v2.h5"
partonic="$experiment_root/synthetic_data/partonic_truth_v2.h5"
python_bin="/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python"
checkpoint="$smoke_root/training/checkpoints/best-loss.ckpt"
prediction="$smoke_root/predictions/test.h5"
evaluation="$smoke_root/evaluation/test.json"

expected_source_sha="${EXPECTED_SOURCE_SHA:?EXPECTED_SOURCE_SHA is required}"
expected_config_sha256="${EXPECTED_CONFIG_SHA256:?EXPECTED_CONFIG_SHA256 is required}"
expected_measured_sha256="${EXPECTED_MEASURED_SHA256:?EXPECTED_MEASURED_SHA256 is required}"
expected_partonic_sha256="${EXPECTED_PARTONIC_SHA256:?EXPECTED_PARTONIC_SHA256 is required}"

[[ "$(git -C "$project_root" rev-parse HEAD)" == "$expected_source_sha" ]]
[[ -z "$(git -C "$project_root" status --porcelain --untracked-files=no)" ]]
[[ "$(sha256sum "$config" | cut -d' ' -f1)" == "$expected_config_sha256" ]]
[[ "$(sha256sum "$measured" | cut -d' ' -f1)" == "$expected_measured_sha256" ]]
[[ "$(sha256sum "$partonic" | cut -d' ' -f1)" == "$expected_partonic_sha256" ]]

if [[ -e "$smoke_root" ]]; then
    if find "$smoke_root" -type f -print -quit | grep -q .; then
        echo "Refusing to reuse non-empty topology-query GPU smoke root: $smoke_root" >&2
        exit 3
    fi
else
    mkdir -p "$smoke_root"
fi
mkdir -p "$smoke_root/training" "$smoke_root/predictions" "$smoke_root/evaluation"

export JER_PROJECT_ROOT="$project_root"
export JER_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/src"
export JER_SOURCE_SHA="$expected_source_sha"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

date -u +%Y-%m-%dT%H:%M:%SZ > "$smoke_root/start_utc.txt"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv > "$smoke_root/gpu_before.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits --loop-ms=1000 > "$smoke_root/gpu_samples.csv" 2> "$smoke_root/gpu_samples.err" &
sampler_pid=$!
stop_sampler() {
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
}
trap stop_sampler EXIT

/usr/bin/time -v "$python_bin" -m joint_event_reconstruction.cli.run train \
    --experiment "$config" \
    --device cuda \
    --set data.events.train=8 \
    --set data.events.validation=4 \
    --set data.events.test=4 \
    --set training.batch_size=4 \
    --set training.epochs=1 \
    --set training.workers=0 \
    --set training.output="$smoke_root/training" \
    > "$smoke_root/train.log" 2>&1

test -s "$checkpoint"

# Prediction is intentionally a separate process: this checks the maintained
# checkpoint loader, target-scaler restoration and truth-free deployment path.
/usr/bin/time -v "$python_bin" -m joint_event_reconstruction.cli.run predict \
    --experiment "$config" \
    --checkpoint "$checkpoint" \
    --output "$prediction" \
    --split test \
    --device cuda \
    --set data.events.train=8 \
    --set data.events.validation=4 \
    --set data.events.test=4 \
    --set inference.invisible.samples_per_assignment=1 \
    --set inference.invisible.point_estimator=cfm_single_draw \
    --set inference.invisible.flow.steps=1 \
    > "$smoke_root/predict.log" 2>&1

test -s "$prediction"
"$python_bin" - "$prediction" <<'PY'
import sys
from joint_event_reconstruction.prediction_schema import validate_prediction

validate_prediction(sys.argv[1])
print("prediction_schema=PASS")
PY

"$python_bin" -m joint_event_reconstruction.cli.run evaluate \
    --predictions "$prediction" > "$evaluation"
test -s "$evaluation"

stop_sampler
trap - EXIT
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv > "$smoke_root/gpu_after.csv"
"$python_bin" - "$smoke_root/gpu_samples.csv" "$smoke_root/gpu_resource_summary.txt" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = []
for line in source.read_text().splitlines():
    fields = [item.strip() for item in line.split(",")]
    if len(fields) >= 6:
        try:
            rows.append((float(fields[3]), float(fields[5])))
        except ValueError:
            pass
if not rows:
    raise SystemExit("GPU sampler recorded no numeric samples")
peak_memory = max(row[0] for row in rows)
peak_utilisation = max(row[1] for row in rows)
target.write_text(
    "nvidia-smi sampled peak memory used MiB: %.1f\n"
    "nvidia-smi sampled peak GPU utilisation percent: %.1f\n"
    "sample_count: %d\n" % (peak_memory, peak_utilisation, len(rows))
)
PY

date -u +%Y-%m-%dT%H:%M:%SZ > "$smoke_root/end_utc.txt"
printf 'GPU topology-query smoke passed: %s\n' "$smoke_root"
