#!/usr/bin/env bash
#SBATCH --job-name=knot-j-transfer
#SBATCH --partition=multicore_small
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/w00238tl/knot/logs/kk-transfer-v1/transfer-%j.log

set -euo pipefail

source_root=/net/scratch/w00238tl/knot
evaluation_root=/net/scratch/w00238tl/knot
artifact_root=/net/scratch/w00238tl/knot
input_root="$artifact_root/results/experiments/knot/kk-transfer-v1"
python_bin=/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER_main_20261002/bin/python
dependency_path=/net/scratch/w00238tl/.knot-pythonpath-20261003
checkpoint="$source_root/results/experiments/knot/matched-production-v1/J/checkpoints/best-16-1.213281.ckpt"
expected_source_commit=eebeb524f877388a10432fc72c7831425f6174e9
expected_evaluation_commit=ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44

mkdir -p "$evaluation_root/logs/kk-transfer-v1"
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_source_commit" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]
[[ "$(git -C "$evaluation_root" rev-parse HEAD)" == "$expected_evaluation_commit" ]]
[[ -f "$checkpoint" ]]
[[ -d "$dependency_path/lgatr" ]]

export PYTHONPATH="$dependency_path:$evaluation_root/src:$evaluation_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" -c 'import h5py, lgatr, sklearn, torch; print("torch", torch.__version__, "lgatr", lgatr.__file__)'

for sample in "mc20d_DSID574801 574801" "mc23a_DSID601230 601230"; do
    read -r sample_name dsid <<< "$sample"
    sample_root="$input_root/$sample_name"
    output_path="$input_root/transfer-$sample_name"
    [[ -f "$sample_root/measured.h5" && -f "$sample_root/truth.h5" ]]
    [[ ! -e "$output_path" ]]
    /usr/bin/time -v srun --cpu-bind=cores "$python_bin" \
        "$evaluation_root/scripts/evaluation/evaluate_knot_frozen_transfer.py" \
        --config "$evaluation_root/configs/experiments/knot_scientific_closure/knot_j_production.public.yaml" \
        --research "$evaluation_root/configs/research/knot_j_production.yaml" \
        --checkpoint "$checkpoint" \
        --input "$sample_root/measured.h5" \
        --truth "$sample_root/truth.h5" \
        --output "$output_path" \
        --samples 32 \
        --batch-size 32 \
        --seed 42011 \
        --device cpu
    "$python_bin" - "$output_path" "$sample_root" "$dsid" <<'PY'
import json
import pathlib
import sys

import h5py
import numpy as np

output, sample_root, dsid = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
metrics = json.loads((output / "metrics.json").read_text())
manifest = json.loads((output / "evaluation_manifest.json").read_text())
with h5py.File(sample_root / "measured.h5", "r") as measured, h5py.File(output / "posterior.h5", "r") as posterior:
    expected = measured["EVENT_ID"][:]
    assert np.array_equal(posterior["EVENT_ID"][:], expected)
    assert posterior["invisible"].shape == (len(expected), 32, 2, 4)
    assert posterior["parents"].shape == (len(expected), 32, 2, 4)
    assert np.isfinite(posterior["invisible"][:]).all()
    assert np.isfinite(posterior["parents"][:]).all()
assert metrics["events"] == len(expected)
assert metrics["sealed_test_used"] is False
assert manifest["truth_used_during_inference"] is False
with h5py.File(sample_root / "measured.h5", "r") as measured:
    assert manifest["source_root_sha256"] == str(measured.attrs["source_root_sha256"])
print(f"Validated DSID {dsid}: N={len(expected)}, K=32, frozen Heun-8", flush=True)
PY
done
