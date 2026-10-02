"""Run bounded real train/validation M/C/J KNOT CPU closure checks."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.knot import KnotModule

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/experiments/knot_scientific_closure"
RESEARCH_DIR = ROOT / "configs/research"
OUTPUT = ROOT / "results/experiments/knot/scientific-closure-run2/real_sample/cpu_smoke"
SAMPLES = 4
SEED = 20261002
SHARED_PREFIXES = (
    "event_model.visible.",
    "event_model.topology_queries.",
    "event_model.reconstruction.",
)


def gradient_norm(module, prefix):
    squared = [
        parameter.grad.detach().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def initialization_summary(module):
    parameters = dict(module.named_parameters())
    shared_names = sorted(
        name for name in parameters if name.startswith(SHARED_PREFIXES)
    )
    hashes = {}
    for prefix in SHARED_PREFIXES:
        digest = hashlib.sha256()
        names = [name for name in shared_names if name.startswith(prefix)]
        for name in names:
            tensor = parameters[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
            digest.update(tensor.numpy().tobytes())
        hashes[prefix.removeprefix("event_model.").rstrip(".")] = digest.hexdigest()
    digest = hashlib.sha256()
    for name in shared_names:
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    total = sum(parameter.numel() for parameter in parameters.values())
    shared = sum(parameters[name].numel() for name in shared_names)
    return {
        "shared_initialization_hash": digest.hexdigest(),
        "shared_component_hashes": hashes,
        "total_parameters": int(total),
        "shared_parameters": int(shared),
        "variant_specific_parameters": int(total - shared),
        "variant_initialization_seed": SEED,
    }


def prediction_batch(batch):
    clean = {
        key: value
        for key, value in batch.items()
        if "truth" not in key.lower() and "support" not in key.lower()
    }
    structured = copy.deepcopy(clean["structured"])
    assignments = structured.get("assignments")
    if assignments is not None:
        assignments["truth_state"] = torch.full_like(assignments["truth_state"], -777)
    for pool in structured["pools"].values():
        pool["truth_index"].fill_(-777)
        pool["reconstructable"].fill_(False)
    clean["structured"] = structured
    return clean


def joint_evaluator(posterior, seed):
    n = posterior.invisible[:, :, 0, 3].detach().cpu().numpy()
    p = posterior.parents[:, :, 0, 3].detach().cpu().numpy()
    permutation = np.argsort(
        np.random.default_rng(seed).random(p.shape), axis=1, kind="stable"
    )
    p_shuffled = np.take_along_axis(p, permutation, axis=1)

    def metrics(left, right):
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        covariance = np.mean(left_centered * right_centered, axis=1)
        left_var = np.mean(left_centered**2, axis=1)
        right_var = np.mean(right_centered**2, axis=1)
        correlation = covariance / np.sqrt(np.maximum(left_var * right_var, 1.0e-12))
        return {
            "mean_conditional_covariance": float(covariance.mean()),
            "mean_conditional_correlation": float(correlation.mean()),
        }

    def marginals():
        return {
            "N_pz_mean": float(n.mean()),
            "N_pz_variance": float(n.var()),
            "P_top_pz_mean": float(p.mean()),
            "P_top_pz_variance": float(p.var()),
        }

    coherent_marginals = marginals()
    shuffled_marginals = {
        "N_pz_mean": float(n.mean()),
        "N_pz_variance": float(n.var()),
        "P_top_pz_mean": float(p_shuffled.mean()),
        "P_top_pz_variance": float(p_shuffled.var()),
    }
    return {
        "coherent_marginals": coherent_marginals,
        "shuffled_marginals": shuffled_marginals,
        "marginal_max_absolute_change": max(
            abs(coherent_marginals[key] - shuffled_marginals[key])
            for key in coherent_marginals
        ),
        "coherent_joint": metrics(n, p),
        "shuffled_joint": metrics(n, p_shuffled),
    }


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=("M", "C", "J"), default=("M", "C", "J"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    torch.set_num_threads(1)
    report = {}

    for variant in args.variants:
        config_path = CONFIG_DIR / f"knot_{variant.lower()}_real_cpu.public.yaml"
        research_path = RESEARCH_DIR / f"knot_{variant.lower()}.yaml"
        config = load_experiment(config_path)
        config.training.output = str(OUTPUT / variant)
        research = OmegaConf.to_container(OmegaConf.load(research_path), resolve=True)
        if research["variant"] != variant or research["training_conditioning"] != "truth":
            raise ValueError(f"Unexpected KNOT research config: {research_path}")
        visible = load_visible_config(config)
        data = build_data(config, visible, device="cpu", complete_assignments=True)
        data.setup("fit")
        train_batch = next(iter(data.train_dataloader()))
        validation_batch = next(iter(data.val_dataloader()))
        if len(train_batch["event_index"]) == 0 or len(validation_batch["event_index"]) == 0:
            raise ValueError("Real KNOT smoke requires both train and validation events")
        if set(train_batch["event_index"].tolist()) & set(validation_batch["event_index"].tolist()):
            raise ValueError("KNOT train and validation EVENT_IDs overlap")

        knot_config = copy.deepcopy(config)
        if knot_config.model.get("parent_cfm") is not None:
            knot_config.model.parent_cfm.enabled = False
        torch.manual_seed(SEED)
        module = KnotModule(
            knot_config,
            visible,
            research=research,
            invisible_scaler=data.invisible_scaler,
            partonic_scaler=data.partonic_scaler,
            synthetic_data=False,
        )
        init = initialization_summary(module)
        module.train()
        output = module(train_batch)
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError(f"{variant}: nonfinite initial total loss")
        component_losses = {
            key: float(value.detach())
            for key, value in output.items()
            if key.endswith("_loss")
        }
        if not component_losses or not all(np.isfinite(value) for value in component_losses.values()):
            raise FloatingPointError(f"{variant}: missing or nonfinite component losses")
        output["loss"].backward()
        groups = {
            "visible": gradient_norm(module, "event_model.visible"),
            "queries": gradient_norm(module, "event_model.topology_queries"),
            "assignment": gradient_norm(module, "event_model.reconstruction"),
        }
        continuous_prefix = "joint_continuous" if variant == "J" else "parent_generator"
        groups["continuous"] = gradient_norm(module, continuous_prefix)
        if variant != "J":
            groups["invisible"] = gradient_norm(module, "event_model.invisible")
        if any(not np.isfinite(value) or value <= 0 for value in groups.values()):
            raise FloatingPointError(f"{variant}: intended gradient missing/nonfinite: {groups}")
        module.zero_grad(set_to_none=True)

        train_dir = OUTPUT / variant
        train_dir.mkdir()
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            max_epochs=2,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=train_dir,
        )
        trainer.fit(module, train_dataloaders=data.train_dataloader())
        checkpoint_path = train_dir / "model.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored = KnotModule(
            knot_config,
            visible,
            research=research,
            invisible_scaler=InvisibleScaler.from_dict(checkpoint["invisible_scaler"]),
            partonic_scaler=PartonicScaler.from_dict(checkpoint["partonic_scaler"]),
            synthetic_data=False,
        )
        restored.validate_checkpoint_metadata(checkpoint)
        restored.load_state_dict(checkpoint["state_dict"])
        restored.eval()

        with torch.no_grad():
            validation_output = restored(validation_batch)
        val_components = {
            key: float(value.detach())
            for key, value in validation_output.items()
            if key.endswith("_loss")
        }
        if not all(np.isfinite(value) for value in val_components.values()):
            raise FloatingPointError(f"{variant}: nonfinite validation objective")

        clean_validation = prediction_batch(validation_batch)
        posterior = restored.posterior(
            clean_validation,
            samples=SAMPLES,
            generator=torch.Generator(device="cpu").manual_seed(SEED + 1),
        )
        if posterior.parents.shape[1] != SAMPLES or not torch.isfinite(posterior.parents).all() or not torch.isfinite(posterior.invisible).all():
            raise FloatingPointError(f"{variant}: invalid posterior prediction")
        posterior_path = train_dir / "posterior_validation.h5"
        posterior.write(posterior_path)
        with h5py.File(posterior_path, "r") as saved:
            if not np.array_equal(saved["EVENT_ID"][:], validation_batch["event_index"].numpy()):
                raise ValueError(f"{variant}: posterior EVENT_IDs changed during save/reload")

        oracle = None
        if variant == "C":
            truth_conditioned = val_components
            predicted_research = {**research, "training_conditioning": "predicted"}
            predicted = KnotModule(
                knot_config,
                visible,
                research=predicted_research,
                invisible_scaler=InvisibleScaler.from_dict(checkpoint["invisible_scaler"]),
                partonic_scaler=PartonicScaler.from_dict(checkpoint["partonic_scaler"]),
                synthetic_data=False,
            )
            predicted.load_state_dict(checkpoint["state_dict"])
            predicted.eval()
            torch.manual_seed(SEED + 2)
            with torch.no_grad():
                predicted_output = predicted(validation_batch)
            pred_components = {
                key: float(value.detach())
                for key, value in predicted_output.items()
                if key.endswith("_loss")
            }
            oracle = {
                "N_given_A_truth_loss": truth_conditioned.get("invisible_loss"),
                "N_given_A_pred_loss": pred_components.get("invisible_loss"),
                "P_given_A_truth_N_truth_loss": truth_conditioned.get("partonic_cfm_loss"),
                "P_given_A_pred_N_pred_loss": pred_components.get("partonic_cfm_loss"),
                "relative_N_context_loss_change": (
                    None
                    if not truth_conditioned.get("invisible_loss")
                    else pred_components["invisible_loss"] / truth_conditioned["invisible_loss"] - 1.0
                ),
                "relative_P_context_loss_change": (
                    None
                    if not truth_conditioned.get("partonic_cfm_loss")
                    else pred_components["partonic_cfm_loss"] / truth_conditioned["partonic_cfm_loss"] - 1.0
                ),
            }

        report[variant] = {
            "train_events_seen": len(train_batch["event_index"]),
            "train_population_events": len(data.train_dataset),
            "validation_events_seen": len(validation_batch["event_index"]),
            "validation_population_events": len(data.val_dataset),
            "training_loss_finite": True,
            "training_component_losses": component_losses,
            "intended_gradients_finite_nonzero": groups,
            "checkpoint_saved_reloaded": True,
            "validation_component_losses": val_components,
            "posterior_prediction_finite": True,
            "posterior_samples": SAMPLES,
            "posterior_shape": list(posterior.parents.shape),
            "initialization": init,
            "truth_independent_prediction": True,
            "joint_evaluator": joint_evaluator(posterior, SEED + 3),
            "oracle": oracle,
            "sealed_test_used": False,
        }

    shared_hashes = {
        value["initialization"]["shared_initialization_hash"]
        for value in report.values()
    }
    if len(shared_hashes) != 1:
        raise ValueError(
            "M/C/J common initialization differs under the matched seed: "
            f"{sorted(shared_hashes)}"
        )
    common_parameter_counts = {
        value["initialization"]["shared_parameters"] for value in report.values()
    }
    if len(common_parameter_counts) != 1:
        raise ValueError("M/C/J do not expose the same shared parameter set")

    result = {
        "source_config": str(CONFIG_DIR),
        "sample": "existing 144-event train/validation adaptation; no test rows",
        "optimizer_steps_per_variant": 2,
        "posterior_sample_policy": "4 coherent samples per validation event; ancestral for C and joint for J",
        "shared_initialization_hash": next(iter(shared_hashes)),
        "parameter_counts": {
            variant: value["initialization"] for variant, value in report.items()
        },
        "variants": report,
        "sealed_test_used": False,
    }
    (OUTPUT / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(OUTPUT / "validation.json"),
        "variants": {
            key: {
                "train_events_seen": value["train_events_seen"],
                "train_population_events": value["train_population_events"],
                "validation_events_seen": value["validation_events_seen"],
                "validation_population_events": value["validation_population_events"],
                "initialization": value["initialization"],
                "training_component_losses": value["training_component_losses"],
                "intended_gradients_finite_nonzero": value["intended_gradients_finite_nonzero"],
                "checkpoint_saved_reloaded": value["checkpoint_saved_reloaded"],
                "posterior_samples": value["posterior_samples"],
                "joint_evaluator": value["joint_evaluator"],
                "oracle": value["oracle"],
            }
            for key, value in report.items()
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
