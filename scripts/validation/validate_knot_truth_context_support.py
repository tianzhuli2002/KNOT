"""Verify truth-infeasible rows are excluded from C/J conditional flows."""

import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.knot import KnotModule

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/experiments/knot/production-launch-v1"
CONFIG = ROOT / "configs/experiments/knot_scientific_closure/knot_m_real_cpu.public.yaml"


def main():
    torch.set_num_threads(1)
    config = load_experiment(CONFIG)
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu", complete_assignments=True)
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    if not batch["visible_assignment_support"].any():
        raise ValueError("Small real-data batch unexpectedly has no supported assignments")
    # Simulate rows whose truth state was truncated from the model support. The
    # production sidecar sets these same two support masks false for its 256 rows.
    batch = copy.deepcopy(batch)
    batch["visible_assignment_support"].zero_()
    batch["structured"]["objective_support"].zero_()
    results = {}
    for variant in ("M", "C", "J"):
        local = copy.deepcopy(config)
        local.model.parent_cfm.enabled = False
        research = OmegaConf.to_container(
            OmegaConf.load(ROOT / f"configs/research/knot_{variant.lower()}_production.yaml"),
            resolve=True,
        )
        torch.manual_seed(int(local.experiment.seed))
        module = KnotModule(
            local,
            visible,
            research=research,
            invisible_scaler=data.invisible_scaler,
            partonic_scaler=data.partonic_scaler,
            synthetic_data=False,
        )
        losses = module(batch)
        component_names = (
            ("invisible_loss", "partonic_cfm_loss")
            if variant in {"M", "C"}
            else ("joint_loss",)
        )
        values = {name: float(losses[name].detach()) for name in component_names}
        if not all(torch.isfinite(losses[name]) for name in component_names):
            raise FloatingPointError(f"{variant}: nonfinite conditional loss {values}")
        if variant in {"C", "J"} and any(value != 0.0 for value in values.values()):
            raise AssertionError(f"{variant}: unsupported truth context entered the objective: {values}")
        if variant == "M" and any(value <= 0.0 for value in values.values()):
            raise AssertionError(f"M marginal losses were unexpectedly masked: {values}")
        results[variant] = {
            "component_losses": values,
            "truth_assignment_support_for_tested_rows": False,
            "conditional_objective_excluded": variant in {"C", "J"},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "production_unsupported_events": 256,
        "production_unsupported_cause": "candidate_truncation at max_jets=8",
        "masked_conditional_variants": ["C", "J"],
        "marginal_control_remains_usable": "M",
        "results": results,
        "reserved_test_used": False,
    }
    (OUT / "truth_context_support_check.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
