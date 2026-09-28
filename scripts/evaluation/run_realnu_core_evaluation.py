#!/usr/bin/env python3
"""Run the independent core validation products before posterior64 finishes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from plot_realnu_campaign import run


def run_core(manifest: Path, evaluation_root: Path) -> dict:
    manifest = manifest.expanduser().resolve()
    evaluation_root = evaluation_root.expanduser().resolve()
    output = evaluation_root / "core"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite core evaluation output: {output}")
    output.mkdir(parents=True)
    for name in (
        "evaluation_contract.json",
        "EVALUATION_CONTRACT.md",
        "classical_baseline_status.md",
    ):
        shutil.copy2(evaluation_root / name, output / name)
    shutil.copytree(evaluation_root / "manifests", output / "manifests")
    result = run(manifest, output)
    (output / "CORE_EVALUATION_STATUS.md").write_text(
        "# Core validation evaluation\n\n"
        f"- Status: {result['status']}\n"
        f"- Validation events: {result['validation_events']:,}\n"
        "- Sealed test used: false\n"
        "- Posterior-extension dependency: none\n\n"
        "These independent core products use the standard production prediction sample counts. "
        "The final 64-sample posterior evaluation and decision freeze are added after POSTERIOR64_R2 completes.\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_core(args.manifest, args.evaluation_root)
    print(
        f"core evaluation {result['status']}: "
        f"{result['validation_events']} events; sealed test used=false"
    )


if __name__ == "__main__":
    main()
