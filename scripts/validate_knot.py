"""CPU-only implementation validation. Creates and reads synthetic train/val only."""

import copy
import json
import subprocess
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.knot import KnotModule
from omegaconf import OmegaConf

from scripts.data.build_topology_query_synthetic import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/knot/development"


def fixture():
    OUT.mkdir(parents=True, exist_ok=True)
    measured = OUT / "measured.h5"
    parents = OUT / "parents.h5"
    if not measured.exists():
        build(measured, parents, 32, 451)
        with h5py.File(measured, "r+") as f:
            f["SPLIT"][:] = np.where(np.arange(32) < 24, 0, 1)
        with h5py.File(parents, "r+") as f:
            support = f["PARTONIC/SUPPORT"][:, :2]
            del f["PARTONIC/SUPPORT"]
            f.create_dataset("PARTONIC/SUPPORT", data=support)
            del f["PARTONIC/ttbar"]
            f.attrs["partonic_roles"] = json.dumps(["top_e", "top_mu"])
            f.attrs["partonic_truth_bindings"] = json.dumps(
                {
                    "top_e": "synthetic_hard_process_top_e",
                    "top_mu": "synthetic_hard_process_top_mu",
                }
            )


def norm(module, prefix):
    return (
        sum(
            float(p.grad.square().sum())
            for n, p in module.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )
        ** 0.5
    )


def prediction_batch(batch):
    clean = {k: v for k, v in batch.items() if "truth" not in k and "support" not in k}
    structured = copy.deepcopy(clean["structured"])
    structured["assignments"]["truth_state"] = torch.full_like(
        structured["assignments"]["truth_state"], -777
    )
    for pool in structured["pools"].values():
        if "truth_index" in pool:
            pool["truth_index"].fill_(-777)
        if "reconstructable" in pool:
            pool["reconstructable"].fill_(False)
    clean["structured"] = structured
    return clean


def main():
    torch.set_num_threads(1)
    torch.manual_seed(451)
    fixture()
    report = {}
    for variant in ("M", "C", "J"):
        config = load_experiment(
            ROOT / f"configs/research/knot_{variant.lower()}_jer.yaml"
        )
        research = OmegaConf.to_container(
            OmegaConf.load(ROOT / f"configs/research/knot_{variant.lower()}.yaml"),
            resolve=True,
        )
        visible = load_visible_config(config)
        data = build_data(config, visible, device="cpu", complete_assignments=True)
        data.setup("fit")
        batch = next(iter(data.train_dataloader()))
        kwargs = {
            "invisible_scaler": data.invisible_scaler,
            "partonic_scaler": data.partonic_scaler,
            "synthetic_data": True,
        }
        module = KnotModule(config, visible, research=research, **kwargs)
        output = module(batch)
        assert torch.isfinite(output["loss"])
        output["loss"].backward()
        gradients = {
            prefix: norm(module, prefix)
            for prefix in [
                "event_model.visible",
                "event_model.topology_queries",
                "event_model.reconstruction",
                "parent_generator" if variant != "J" else "joint_continuous",
            ]
        }
        assert all(v > 0 for v in gradients.values()), gradients

        def forbidden(*args, **kwargs):
            raise AssertionError("Inactive objective executed")

        if variant == "J":
            module.event_model.invisible.sample = forbidden
            module.event_model.invisible.loss = forbidden
            # A parent velocity must respond to an invisible state coordinate.
            context_visible = module.event_model.encode_visible(batch)
            result, space = module.assignment(batch, context_visible)
            states = space.truth_state[:, None]
            context = torch.cat(
                (
                    module.invisible_context(result, space, states, context_visible)[
                        :, 0
                    ],
                    module.parent_context(result, space, states, context_visible)[:, 0],
                ),
                dim=1,
            )
            memory, mask = module.event_model.invisible.visible_context(context_visible)
            state = torch.randn(
                len(memory),
                int(module.continuous.active_mask.sum()),
                requires_grad=True,
            )
            velocity = module.continuous.velocity(
                state, torch.full((len(memory),), 0.5), memory, mask, context
            )
            derivative = torch.autograd.grad(
                velocity[:, module.n_invisible :].sum(), state
            )[0][:, : module.n_invisible]
            assert derivative.abs().max() > 0
        saved_weights = dict(module.weights)
        module.weights = {key: 0.0 for key in saved_weights}
        previous = module.event_model.reconstruction.forward
        module.event_model.reconstruction.forward = forbidden
        assert module(batch)["loss"].item() == 0
        module.event_model.reconstruction.forward = previous
        module.weights = saved_weights
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            max_epochs=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=OUT / variant,
        )
        trainer.fit(module, train_dataloaders=data.train_dataloader())
        assert norm(module, "event_model.topology_queries.invisible_projection") > 0
        if variant != "J":
            assert norm(module, "event_model.invisible") > 0
        else:
            packed = batch["invisible_truth_scaled"][:, module.invisible_active]
            assert torch.equal(
                module.unpack_invisible(packed), batch["invisible_truth_scaled"]
            )
        checkpoint = OUT / f"{variant}.ckpt"
        trainer.save_checkpoint(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored = KnotModule(config, visible, research=research, **kwargs)
        restored.validate_checkpoint_metadata(saved)
        restored.load_state_dict(saved["state_dict"])
        restored.eval()
        calls = []
        ancestry = {}
        handle = restored.event_model.visible.register_forward_hook(
            lambda *args, calls=calls: calls.append(1)
        )
        original = restored.parent_context

        def capture(
            result,
            space,
            states,
            visible,
            neutrino=None,
            ancestry=ancestry,
            original=original,
        ):
            ancestry["states"] = states.clone()
            ancestry["neutrino"] = None if neutrino is None else neutrino.clone()
            return original(result, space, states, visible, neutrino)

        restored.parent_context = capture
        clean = prediction_batch(batch)
        posterior = restored.posterior(
            clean, samples=3, generator=torch.Generator().manual_seed(12)
        )
        handle.remove()
        assert len(calls) == 1
        assert torch.equal(posterior.event_id, batch["event_index"])
        assert torch.equal(ancestry["states"], posterior.assignment_state)
        if variant == "C":
            assert torch.equal(ancestry["neutrino"], posterior.invisible[..., 1:])
        assert not torch.equal(posterior.invisible[:, 0], posterior.invisible[:, 1])
        assert not torch.equal(posterior.parents[:, 0], posterior.parents[:, 1])
        # Poisoned truth vs original truth cannot alter normal inference.
        other = restored.posterior(
            batch, samples=3, generator=torch.Generator().manual_seed(12)
        )
        assert torch.equal(other.invisible, posterior.invisible) and torch.equal(
            other.parents, posterior.parents
        )
        path = OUT / f"{variant}_posterior.h5"
        if path.exists():
            path.unlink()
        posterior.write(path)
        with h5py.File(OUT / f"{variant}_posterior.h5") as f:
            assert np.array_equal(f["EVENT_ID"][:], batch["event_index"])
        if variant in {"C", "J"}:
            predicted = KnotModule(
                config,
                visible,
                research={**research, "training_conditioning": "predicted"},
                **kwargs,
            )
            predicted(batch)["loss"].backward()
        report[variant] = {
            "loss": float(output["loss"].detach()),
            "gradients": gradients,
            "encoder_calls_per_posterior": len(calls),
            "checkpoint_reload": True,
            "inactive_objectives_skipped": True,
            "invisible_query_projection_gradient": norm(
                module, "event_model.topology_queries.invisible_projection"
            ),
            "joint_state_interaction": True if variant == "J" else None,
            "posterior_shape": list(posterior.parents.shape),
        }
    report.update(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        worktree=str(ROOT),
        sealed_test_used=False,
        population="32 synthetic events, 24 train / 8 validation; no test rows",
        knot_c_coherent_ancestral_sampling="PASS",
        knot_j_joint_continuous_sampling="PASS",
    )
    (OUT / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
