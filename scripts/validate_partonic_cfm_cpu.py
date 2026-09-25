"""Bounded CPU checks for the joint after-FSR parent conditional flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.models.partonic import (
    PartonicCFM,
    parent_pair_observables,
)


def _fixture():
    generator = torch.Generator(device="cpu").manual_seed(80421)
    momentum = torch.randn((16, 2, 3), generator=generator) * 80.0
    mass = 170.0 + torch.randn((16, 2), generator=generator) * 8.0
    energy = (momentum.square().sum(-1) + mass.square()).sqrt()
    truth = torch.cat((energy.unsqueeze(-1), momentum), dim=-1)
    support = torch.ones((16, 2), dtype=torch.bool)
    scaler = PartonicScaler.fit(truth, support, PartonicCFM.ROLES)
    return truth, support, scaler


def run() -> dict:
    torch.manual_seed(80421)
    truth, support, scaler = _fixture()
    batch, width = 4, 8
    visible = torch.randn(batch, 5, width)
    visible_mask = torch.tensor(
        [[False, False, False, True, True]] * batch, dtype=torch.bool
    )
    parents = torch.randn(batch, 2, width, requires_grad=True)
    assignment = torch.randn(batch, 2, width, requires_grad=True)
    neutrinos = torch.randn(batch, 2, 3)
    values = truth[:batch].clone()
    row_support = torch.tensor([[True, True], [True, True], [False, False], [True, True]])
    result = {"modes": {}, "round_trip_max_abs": None}

    for mode in ("P0", "P1", "P2"):
        module = PartonicCFM(
            width, 2, 1, 16, 0.0, scaler=scaler, conditioning=mode
        )
        kwargs = {}
        if mode in {"P1", "P2"}:
            kwargs["assignment_context"] = assignment
        if mode == "P2":
            kwargs["predicted_neutrino_momentum"] = neutrinos
        rng = torch.Generator(device="cpu").manual_seed(127 + len(mode))
        losses = module.loss(
            values,
            row_support,
            visible,
            visible_mask,
            parents,
            generator=rng,
            **kwargs,
        )
        if not torch.isfinite(losses["loss"]):
            raise AssertionError(f"{mode} loss is not finite")
        losses["loss"].backward()
        if parents.grad is None or not torch.isfinite(parents.grad).all():
            raise AssertionError(f"{mode} did not propagate finite gradients to parent contexts")

        sample_generator = torch.Generator(device="cpu").manual_seed(90210)
        sampled = module.sample(
            visible,
            visible_mask,
            parents.detach(),
            samples=3,
            steps=2,
            solver="heun",
            generator=sample_generator,
            **{key: value.detach() if torch.is_tensor(value) else value for key, value in kwargs.items()},
        )
        replay = module.sample(
            visible,
            visible_mask,
            parents.detach(),
            samples=3,
            steps=2,
            solver="heun",
            generator=torch.Generator(device="cpu").manual_seed(90210),
            **{key: value.detach() if torch.is_tensor(value) else value for key, value in kwargs.items()},
        )
        if not torch.equal(sampled["parents"], replay["parents"]):
            raise AssertionError(f"{mode} sampling is not seed-reproducible")
        if sampled["parents"].shape != (batch, 3, 2, 4):
            raise AssertionError(f"{mode} sample shape is wrong")
        if not torch.equal(sampled["pair"]["ttbar"], sampled["parents"].sum(dim=-2)):
            raise AssertionError(f"{mode} pair four-vector is not the exact parent sum")
        if not all(torch.isfinite(value).all() for key, value in sampled["pair"].items() if key != "ttbar"):
            raise AssertionError(f"{mode} derived pair observables are not finite")

        result["modes"][mode] = {
            "loss": float(losses["loss"].detach()),
            "support_fraction": float(losses["support_fraction"]),
            "gradient_norm": float(parents.grad.norm()),
            "sample_shape": list(sampled["parents"].shape),
            "solver_nfe": sampled["nfe"],
        }
        parents.grad = None

    encoded = scaler.transform(truth)
    decoded = scaler.inverse(encoded)
    result["round_trip_max_abs"] = float((decoded - truth).abs().max())
    if not torch.allclose(decoded, truth, atol=1.0e-4, rtol=1.0e-5):
        raise AssertionError("Partonic scaler does not round-trip physical parents")

    # Unsupported rows remain in the common batch but have exactly zero loss weight.
    module = PartonicCFM(width, 2, 1, 16, 0.0, scaler=scaler, conditioning="P0")
    support_events = torch.tensor([True, True, False, False])
    rng_a = torch.Generator(device="cpu").manual_seed(3501)
    rng_b = torch.Generator(device="cpu").manual_seed(3501)
    normal = values.clone()
    changed = values.clone()
    changed[~support_events, :, 1] += 100000.0
    changed[~support_events, :, 0] = torch.sqrt(
        changed[~support_events, :, 1:].square().sum(dim=-1)
        + 170.0**2
    )
    loss_a = module.loss(normal, support_events, visible, visible_mask, parents.detach(), generator=rng_a)["loss"]
    loss_b = module.loss(changed, support_events, visible, visible_mask, parents.detach(), generator=rng_b)["loss"]
    if not torch.equal(loss_a, loss_b):
        raise AssertionError("Unsupported placeholders affected support-masked CFM loss")
    zero = module.loss(values, torch.zeros(batch, dtype=torch.bool), visible, visible_mask, parents.detach())["loss"]
    if not torch.isfinite(zero) or zero.requires_grad is False:
        raise AssertionError("All-unsupported CFM loss must be finite and gradient-safe")
    zero.backward()
    result["support_mask"] = {
        "unsupported_placeholder_invariant": True,
        "all_unsupported_loss": float(zero.detach()),
        "all_unsupported_has_grad_fn": bool(zero.grad_fn),
    }

    # Check the persisted scaler/role contract and strict model reload.
    p0 = PartonicCFM(width, 2, 1, 16, 0.0, scaler=scaler, conditioning="P0")
    saved = {"state_dict": p0.state_dict(), "scaler": scaler.as_dict(), "roles": list(p0.ROLES)}
    restored_scaler = PartonicScaler.from_dict(saved["scaler"])
    restored = PartonicCFM(width, 2, 1, 16, 0.0, scaler=restored_scaler, conditioning="P0")
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.eval()
    result["checkpoint_contract"] = {
        "roles": saved["roles"],
        "scaler_representation": restored_scaler.representation,
        "strict_state_reload": True,
    }
    result["truth_definition"] = "synthetic fixture for implementation validation only"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
