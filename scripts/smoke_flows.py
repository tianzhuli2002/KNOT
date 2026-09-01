#!/usr/bin/env python3
"""Focused smoke coverage for continuous, discrete and joint Flow Matching."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
    validate_experiment,
)
from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.models.joint_flow import JointFlowState
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.smoke_rng import seed_smoke


FLOW_MODES = (
    "cfm_no_assignment",
    "ot_cfm",
    "dfm_assignment",
    "joint_flow",
    "joint_flow_recurrent",
)

_MODE_SEED_OFFSETS = {
    "cfm_no_assignment": 201,
    "ot_cfm": 202,
    "dfm_assignment": 203,
    "joint_flow": 204,
    "joint_flow_recurrent": 205,
}


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def configure(base, mode):
    """Make a tiny explicit internal smoke config from one resolved experiment."""
    config = copy.deepcopy(base)
    assignment = config.model.assignment
    config.model.mode = "factorized"
    config.model.visible.layers = 1
    assignment.reasoning.enabled = False
    assignment.reasoning.block_layers = 1
    assignment.formulation = "discriminative"
    assignment.feedback.enabled = False
    config.model.invisible.enabled = True
    config.model.invisible.formulation = "flow_matching"
    config.model.invisible.conditioning.assignment = "none"
    config.model.invisible.conditioning.allow_truth_oracle = False
    config.model.invisible.denoiser.d_model = 32
    config.model.invisible.denoiser.n_heads = 4
    config.model.invisible.denoiser.layers = 1
    config.model.invisible.denoiser.d_feedforward = 64
    config.model.invisible.denoiser.zero_init_output = False
    config.model.invisible.flow_matching.coupling = "independent"
    config.inference.invisible.flow_matching.solver = "euler"
    config.inference.invisible.flow_matching.steps = 2
    config.model.coupling.invisible_to_assignment.enabled = False
    config.model.compatibility.enabled = False
    config.model.joint_flow.recurrent_state = False
    config.model.joint_flow.block_layers = 1
    config.model.joint_flow.train_steps = [1, 2]
    config.inference.joint_flow.steps = 2
    config.inference.joint_flow.sample_chunk_size = 2
    config.inference.posterior_samples = 2

    if mode == "ot_cfm":
        config.model.invisible.flow_matching.coupling = "minibatch_ot"
    elif mode == "dfm_assignment":
        assignment.formulation = "flow_matching"
        config.model.invisible.enabled = False
    elif mode in {"joint_flow", "joint_flow_recurrent"}:
        config.model.mode = "joint_flow"
        assignment.formulation = "flow_matching"
        config.model.invisible.enabled = True
        config.model.joint_flow.recurrent_state = mode == "joint_flow_recurrent"

    if mode == "dfm_assignment":
        config.training.trainable = ["visible", "assignment"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 0.0
    else:
        config.training.trainable = ["visible", "assignment", "invisible"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 1.0
    config.training.loss_weights.compatibility = 0.0
    validate_experiment(config)
    return config


def _finite_gradients(module):
    values = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return bool(values) and all(torch.isfinite(value).all() for value in values)


def _gradient_l1(values) -> float:
    return sum(float(value.abs().sum().detach()) for value in values if value is not None)


def _joint_coupling_gradients(module, batch) -> tuple[float, float]:
    """Deterministically verify both cross-modal paths in one joint-flow step."""
    model = module.event_model
    joint = model.joint_flow
    visible = model.encode_visible(batch)
    space = AssignmentSpace.from_structured(batch["structured"])
    candidates = model.reconstruction.encode_candidates(visible["tokens"], batch["structured"])
    batch_size = len(batch["tokens"])
    time = torch.full(
        (batch_size,), 0.5, device=batch["tokens"].device, dtype=batch["tokens"].dtype
    )
    state = JointFlowState(
        assignment=torch.full(
            (batch_size,), space.mask_state, dtype=torch.long, device=batch["tokens"].device
        ),
        invisible=batch["invisible_truth_scaled"],
        topology=None,
    )
    step = joint.step(
        visible,
        batch["structured"],
        model.reconstruction,
        model.invisible,
        space,
        candidates,
        state,
        time,
    )
    assignment_probe = F.cross_entropy(step.complete_assignment_logits, space.truth_state)
    invisible_to_assignment = torch.autograd.grad(
        assignment_probe,
        tuple(joint.reasoner.invisible_projection.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    velocity_probe = step.invisible_velocity.square().mean()
    assignment_to_invisible = torch.autograd.grad(
        velocity_probe,
        tuple(joint.state_encoder.parameters()),
        retain_graph=False,
        allow_unused=True,
    )
    return _gradient_l1(assignment_to_invisible), _gradient_l1(invisible_to_assignment)


def run_mode(base, visible, batch, scaler, mode, device):
    seed_smoke(int(base.experiment.seed) + _MODE_SEED_OFFSETS[mode])
    config = configure(base, mode)
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    ).to(device)
    module.train()
    module.zero_grad(set_to_none=True)
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise RuntimeError("non-finite total loss")

    summary = {
        "loss": float(output["loss"].detach()),
        "parameters": sum(parameter.numel() for parameter in module.parameters()),
    }
    if mode.startswith("joint_flow"):
        a_to_nu, nu_to_a = _joint_coupling_gradients(module, batch)
        if a_to_nu <= 0 or nu_to_a <= 0:
            raise RuntimeError(
                "joint cross-modal gradients missing in deterministic coupling probe: "
                f"A->nu={a_to_nu}, nu->A={nu_to_a}"
            )
        summary["assignment_to_invisible_gradient_l1"] = a_to_nu
        summary["invisible_to_assignment_gradient_l1"] = nu_to_a

    output["loss"].backward()
    if not _finite_gradients(module):
        raise RuntimeError("missing or non-finite gradients")

    module.eval()
    with torch.inference_mode():
        if mode in {"cfm_no_assignment", "ot_cfm"}:
            generated, nfe = module.event_model.invisible.sample(
                output["visible"],
                output["invisible_parent_context"],
                samples=2,
                budget=2,
                generator=torch.Generator(device=device).manual_seed(9),
            )
            expected = (len(batch["tokens"]), 2, len(module.spec.invisible_roles), 3)
            if generated.shape != expected:
                raise RuntimeError(f"CFM sample shape is wrong: {tuple(generated.shape)}")
            if nfe != 2 or not torch.isfinite(generated).all():
                raise RuntimeError("CFM Euler sampling failed")
            summary.update({"nfe": nfe, "sample_shape": list(generated.shape)})
        elif mode == "dfm_assignment":
            sampled = module.event_model.assignment_flow.sample(
                output["visible"],
                batch["structured"],
                module.event_model.reconstruction,
                steps=2,
                generator=torch.Generator(device=device).manual_seed(10),
            )
            states = sampled["sampled_assignment_state"]
            valid = batch["structured"]["assignments"]["valid_mask"]
            rows = torch.arange(len(states), device=states.device)
            if not valid[rows, states].all():
                raise RuntimeError("DFM generated an event-invalid assignment state")
            summary.update(
                {
                    "nfe": sampled["nfe"],
                    "assignment_state_count": int(
                        batch["structured"]["assignments"]["root_candidate_indices"].shape[0]
                    ),
                }
            )
        else:
            sampled = module.event_model.joint_flow.sample(
                output["visible"],
                batch["structured"],
                module.event_model.reconstruction,
                module.event_model.invisible,
                samples=2,
                steps=2,
                sample_chunk_size=2,
                generator=torch.Generator(device=device).manual_seed(11),
            )
            states = sampled["assignment_state"]
            valid = batch["structured"]["assignments"]["valid_mask"]
            rows = torch.arange(len(states), device=states.device).unsqueeze(1).expand_as(states)
            if not valid[rows, states].all():
                raise RuntimeError("Joint Flow generated an event-invalid assignment state")
            if not torch.isfinite(sampled["invisible"]).all():
                raise RuntimeError("Joint Flow generated non-finite invisible momenta")
            summary.update(
                {
                    "nfe": sampled["nfe"],
                    "joint_assignment_shape": list(states.shape),
                    "joint_invisible_shape": list(sampled["invisible"].shape),
                    "sample_chunk_size": int(sampled["sample_chunk_size"]),
                    "recurrent_state": bool(sampled["recurrent_state"]),
                    "training_recurrent_steps": int(output["recurrent_steps"]),
                }
            )
            if mode == "joint_flow_recurrent" and output["recurrent_steps"] not in {1, 2}:
                raise RuntimeError("Recurrent joint-flow training used an invalid unroll depth")
    return summary


def run_flow_smoke(
    experiment_path: Path,
    *,
    device: str = "cpu",
    events: int = 8,
    modes=None,
    real_data: bool = False,
):
    torch.set_num_threads(1)
    base = load_experiment(experiment_path)
    seed_smoke(int(base.experiment.seed) + 20_000)
    if not real_data and base.data.tensor_cache is None:
        if base.data.invisible.path is None:
            raise ValueError(
                "File-backed flow smoke requires a tensor cache or configured "
                "invisible_truth; use scripts/smoke_synthetic.py for an in-memory smoke"
            )
        # A compact real experiment deliberately has no obsolete visible tensor
        # cache.  Use its canonical HDF5 pair instead of attempting to create a
        # synthetic sidecar from a cache that does not exist.
        real_data = True
    if not real_data and base.experiment.name == "ttbar1L" and base.data.tensor_cache is None:
        base.data.tensor_cache = (
            "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/SB_paper/results/"
            "transformer_full_campaign_joint_first_v2/cache/"
            "transformer_ttbar1L_semantic_reconstruction_full__jets8_lep1_met1__"
            "tok17__all__filelabels__458b8dcf1959.pt"
        )
    for split in ("train", "val", "test"):
        base.data.max_events[split] = events

    if not real_data:
        sidecar = Path("data/synthetic") / f"{base.experiment.name}_flow_{events}.h5"
        command = [
            sys.executable,
            "scripts/data/make_synthetic_invisible.py",
            "--experiment",
            str(experiment_path),
            "--output",
            str(sidecar),
            "--max-events",
            str(events),
            "--seed",
            str(base.experiment.seed),
        ]
        if base.data.tensor_cache is not None:
            command += ["--tensor-cache", str(base.data.tensor_cache)]
        subprocess.run(command, check=True)
        base.data.invisible.path = str(sidecar.resolve())
        base.data.invisible.allow_synthetic = True
    elif base.data.invisible.path is None:
        raise ValueError("Real-data flow smoke requires invisible_truth in the experiment")

    visible = load_visible_config(base)
    visible.train.n_epochs = 1
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = events
        visible.dataset[f"{split}_num_workers"] = 0
    visible.dataset.pin_memory = False
    visible.dataset.persistent_workers = False

    data_config = configure(base, "joint_flow")
    data = StructuredDataModule(
        visible,
        data_config.topology_config,
        tensor_cache=data_config.data.tensor_cache,
        invisible_path=data_config.data.invisible.path,
        allow_synthetic=bool(base.data.invisible.allow_synthetic),
        max_events={"train": events, "val": events, "test": events},
        subset_seed=int(base.data.subset_seed),
        build_assignments=True,
        max_assignment_states=int(data_config.data.max_assignment_states),
    )
    data.setup("fit")
    batch = move(next(iter(data.train_dataloader())), torch.device(device))

    results = {}
    for mode in FLOW_MODES if modes is None else modes:
        results[mode] = {
            "status": "PASS",
            **run_mode(base, visible, batch, data.invisible_scaler, mode, device),
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", type=int, default=8)
    args = parser.parse_args()
    run_flow_smoke(Path(args.experiment), device=args.device, events=args.events)


if __name__ == "__main__":
    main()
