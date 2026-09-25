#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  evaluate_partonic_test.sh --experiment CONFIG --checkpoint CKPT \
    --measured TEST_MEASURED.h5 --truth TEST_SUPERVISION.h5 --output-dir DIR \
    [--samples N] [--steps N] [--solver euler|heun] [--seed N] [--device cuda|cpu]

This command is for a later, explicitly approved final evaluation. It requires
a test-only adapter and uses the checkpoint's saved training scaler. It does
not fit scalers or train on test events.
EOF
}

experiment=
checkpoint=
measured=
truth=
output_dir=
samples=100
steps=32
solver=heun
seed=42001
device=cuda
while (($#)); do
  case "$1" in
    --experiment|--checkpoint|--measured|--truth|--output-dir|--samples|--steps|--solver|--seed|--device)
      (($# >= 2)) || { usage; exit 2; }
      key=${1#--}
      value=$2
      case "$key" in
        experiment) experiment=$value ;;
        checkpoint) checkpoint=$value ;;
        measured) measured=$value ;;
        truth) truth=$value ;;
        output-dir) output_dir=$value ;;
        samples) samples=$value ;;
        steps) steps=$value ;;
        solver) solver=$value ;;
        seed) seed=$value ;;
        device) device=$value ;;
      esac
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for name in experiment checkpoint measured truth output_dir; do
  if [[ -z "${!name}" ]]; then
    echo "Missing required --${name//_/-}" >&2
    usage
    exit 2
  fi
done
if [[ "$solver" != euler && "$solver" != heun ]]; then
  echo "--solver must be euler or heun" >&2
  exit 2
fi
if ((samples <= 0 || steps <= 0)); then
  echo "--samples and --steps must be positive" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
python_bin=${JER_PYTHON:-python}
export JER_PROJECT_ROOT="$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output_dir"
output_dir=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_dir")

"$python_bin" scripts/data/validate_partonic_contract.py \
  --experiment "$experiment" \
  --checkpoint "$checkpoint" \
  --allow-reserved-test \
  --set "data.input=$measured" \
  --set data.visible_truth=null \
  --set "model.parent_cfm.target_path=$truth" \
  > "$output_dir/test_contract.json"

prediction="$output_dir/partonic_test_samples.h5"
"$python_bin" -m joint_event_reconstruction.cli.run predict-partonic \
  --experiment "$experiment" \
  --checkpoint "$checkpoint" \
  --output "$prediction" \
  --split test \
  --device "$device" \
  --samples "$samples" \
  --steps "$steps" \
  --solver "$solver" \
  --seed "$seed" \
  --set "data.input=$measured" \
  --set "data.inference_input=$measured" \
  --set data.visible_truth=null \
  --set data.invisible_truth=null \
  --set data.parent.path=null \
  --set data.cache=null \
  --set model.parent_cfm.target_path=null \
  --set 'training.trainable=[visible,assignment,queries]' \
  --set training.loss_weights.assignment=0 \
  --set training.loss_weights.invisible=0 \
  --set training.loss_weights.parent=0 \
  --set training.loss_weights.partonic_cfm=0 \
  > "$output_dir/predict.json"
"$python_bin" -m joint_event_reconstruction.cli.run evaluate-partonic \
  --predictions "$prediction" \
  --truth "$truth" \
  --output "$output_dir/evaluation.json" \
  > "$output_dir/evaluate.log"
printf 'Wrote final-test prediction and evaluation to %s\n' "$output_dir"
