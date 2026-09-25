#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  submit_partonic_phase.sh --phase p0|p1 --smoke-artifact PATH [--sbatch-arg ARG ...] [--submit]

Without --submit this validates the smoke artifact and prints the command only.
When --submit is present, provide the intended Slurm resource options through
one or more --sbatch-arg values. This script supplies no resource defaults.
EOF
}

phase=
smoke_artifact=
submit=0
sbatch_args=()
while (($#)); do
  case "$1" in
    --phase)
      (($# >= 2)) || { usage; exit 2; }
      phase=$2
      shift 2
      ;;
    --smoke-artifact)
      (($# >= 2)) || { usage; exit 2; }
      smoke_artifact=$2
      shift 2
      ;;
    --sbatch-arg)
      (($# >= 2)) || { usage; exit 2; }
      sbatch_args+=("$2")
      shift 2
      ;;
    --submit)
      submit=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$phase" in
  p0) experiment="configs/experiments/partonic_ttbar2L_2M_p0_phase3.public.yaml" ;;
  p1) experiment="configs/experiments/partonic_ttbar2L_2M_p1_phase4.public.yaml" ;;
  *) echo "--phase must be p0 or p1" >&2; usage; exit 2 ;;
esac
if [[ -z "$smoke_artifact" ]]; then
  echo "An explicit successful GPU-smoke artifact is required." >&2
  usage
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
python_bin=${JER_PYTHON:-python}
export JER_PROJECT_ROOT="$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
smoke_artifact=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$smoke_artifact")
test -s "$smoke_artifact" || { echo "GPU-smoke artifact is missing: $smoke_artifact" >&2; exit 2; }

"$python_bin" - "$smoke_artifact" "$phase" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

artifact_path = Path(sys.argv[1])
phase = sys.argv[2]
artifact = json.loads(artifact_path.read_text())
if artifact.get("status") != "passed":
    raise SystemExit(f"GPU smoke did not pass: {artifact.get('status')!r}")
if artifact.get("source_tree_dirty") is not False:
    raise SystemExit("GPU-smoke source tree was dirty; rerun the smoke from the committed source")
current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if artifact.get("source_commit") != current_commit:
    raise SystemExit(
        f"GPU smoke commit {artifact.get('source_commit')!r} differs from current HEAD {current_commit!r}"
    )
paths = artifact.get("paths", {})
checksums = artifact.get("sha256", {})
for name in ("source_manifest", "split_membership"):
    path = Path(paths.get(name, ""))
    if not path.is_file():
        raise SystemExit(f"Smoke artifact source identity file is missing: {name}={path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if checksums.get(name) != digest:
        raise SystemExit(f"Smoke artifact {name} checksum no longer matches {path}")
if artifact.get("conditioning") != "P0":
    raise SystemExit("The required smoke gate must exercise real-data parent-CFM P0")
print(f"Validated GPU smoke for phase {phase}: {artifact_path}")
PY

"$python_bin" -m joint_event_reconstruction.cli.run show-config \
  --experiment "$experiment" > /dev/null
"$python_bin" scripts/data/validate_partonic_contract.py \
  --experiment "$experiment" > /dev/null

runner="$repo_root/scripts/production/run_partonic_phase.sh"
command=(sbatch "${sbatch_args[@]}" "$runner" "$experiment")
if ((submit == 0)); then
  printf 'Validated; not submitted. Command:'
  printf ' %q' "${command[@]}"
  printf '\nRe-run with --submit to initiate this Slurm submission.\n'
  exit 0
fi
if ((${#sbatch_args[@]} == 0)); then
  echo "--submit requires explicit --sbatch-arg resource options; no site defaults are inferred." >&2
  exit 2
fi
exec "${command[@]}"
