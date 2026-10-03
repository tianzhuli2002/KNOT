"""Train or predict with one frozen KNOT M/C/J experiment."""

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.knot import KnotModule

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "configs/experiments/knot_scientific_closure/matched_production_freeze.json"
SHARED_PREFIXES = (
    "event_model.visible.",
    "event_model.topology_queries.",
    "event_model.reconstruction.",
)


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_value(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def parameter_summary(module):
    parameters = dict(module.named_parameters())
    shared_names = sorted(
        name for name in parameters if name.startswith(SHARED_PREFIXES)
    )
    digest = hashlib.sha256()
    component_hashes = {}
    for name in shared_names:
        tensor = parameters[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes())
    for prefix in SHARED_PREFIXES:
        component = hashlib.sha256()
        for name in shared_names:
            if not name.startswith(prefix):
                continue
            tensor = parameters[name].detach().cpu().contiguous()
            component.update(name.encode())
            component.update(str(tensor.dtype).encode())
            component.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
            component.update(tensor.numpy().tobytes())
        component_hashes[prefix.removeprefix("event_model.").rstrip(".")] = component.hexdigest()
    total = sum(parameter.numel() for parameter in parameters.values())
    shared = sum(parameters[name].numel() for name in shared_names)
    return {
        "shared_initialization_sha256": digest.hexdigest(),
        "shared_component_sha256": component_hashes,
        "total_parameters": int(total),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in parameters.values() if parameter.requires_grad)
        ),
        "shared_parameters": int(shared),
        "variant_specific_parameters": int(total - shared),
    }


def event_id_hash(ids):
    ids = np.asarray(ids, dtype="<i8")
    return hashlib.sha256(np.sort(ids).tobytes()).hexdigest()


def membership_summary(data):
    result = {}
    for name, dataset in (("train", data.train_dataset), ("validation", data.val_dataset)):
        ids = data._selected_ids(name, np.arange(len(dataset), dtype=np.int64))
        result[name] = {"events": len(ids), "event_id_membership_sha256_sorted_little_endian_int64": event_id_hash(ids)}
    return result


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def truth_free_batch(batch):
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


def gradient_norm(module, prefixes):
    squared = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes) and parameter.grad is not None
    ]
    return 0.0 if not squared else float(torch.stack(squared).sum().sqrt())


class TrainingIntegrity(Callback):
    def __init__(self, *, smoke, variant):
        self.smoke = smoke
        self.variant = variant
        self.last_gradient_norms = {}

    def on_before_backward(self, trainer, module, loss):
        del trainer, module
        if not torch.isfinite(loss).all():
            raise FloatingPointError("KNOT training loss is nonfinite")

    def on_before_optimizer_step(self, trainer, module, optimizer):
        del trainer, optimizer
        gradients = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("KNOT gradients are absent or nonfinite")
        continuous_prefix = (
            "joint_continuous." if self.variant == "J" else "parent_generator."
        )
        groups = {
            "visible": ("event_model.visible.",),
            "topology_queries": ("event_model.topology_queries.",),
            "assignment": ("event_model.reconstruction.",),
            "continuous": (continuous_prefix,),
        }
        if self.variant != "J":
            groups["invisible"] = ("event_model.invisible.",)
        self.last_gradient_norms = {
            name: gradient_norm(module, prefixes) for name, prefixes in groups.items()
        }
        if self.smoke and any(value <= 0 for value in self.last_gradient_norms.values()):
            raise FloatingPointError(
                f"KNOT intended gradient missing in GPU smoke: {self.last_gradient_norms}"
            )


def make_module(config, visible, research, data):
    knot_config = copy.deepcopy(config)
    if knot_config.model.get("parent_cfm") is not None:
        knot_config.model.parent_cfm.enabled = False
    torch.manual_seed(int(config.experiment.seed))
    return KnotModule(
        knot_config,
        visible,
        research=research,
        invisible_scaler=data.invisible_scaler,
        partonic_scaler=data.partonic_scaler,
        synthetic_data=data.structured_collator.invisible_store.synthetic,
    )


def validate_frozen_configuration(args, config, research, freeze):
    variant = research["variant"]
    paths = freeze["configs"][variant]
    if Path(args.config).resolve() != (ROOT / paths["experiment"]).resolve():
        raise ValueError(f"KNOT {variant} public config differs from matched freeze")
    if Path(args.research).resolve() != (ROOT / paths["research"]).resolve():
        raise ValueError(f"KNOT {variant} research config differs from matched freeze")
    common = freeze["training"]["common"]
    actual = {
        "seed": int(config.experiment.seed),
        "subset_seed": int(config.data.subset_seed),
        "batch_size": int(config.training.batch_size),
        "epochs": int(config.training.epochs),
        "precision": str(config.training.precision),
        "optimizer": "AdamW",
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "fused_adamw_requested": bool(config.training.fused_adamw),
        "gradient_clip": float(config.training.gradient_clip),
        "workers": int(config.training.workers),
        "accumulation_steps": int(config.training.accumulate_grad_batches),
        "data_loader": {
            "pin_memory": bool(config.data.loader.pin_memory),
            "persistent_workers": bool(config.data.loader.persistent_workers),
            "prefetch_factor": int(config.data.loader.prefetch_factor),
        },
        "scheduler": {
            "type": "ReduceLROnPlateau" if bool(config.training.scheduler.enabled) else None,
            "factor": float(config.training.scheduler.factor),
            "patience": int(config.training.scheduler.patience),
            "minimum_learning_rate": float(config.training.scheduler.min_lr),
        },
        "early_stopping_enabled": bool(config.training.early_stopping.enabled),
        "inference_flow_steps": int(config.inference.invisible.flow_matching.steps),
        "inference_solver": str(config.inference.invisible.flow_matching.solver),
    }
    expected = {
        "seed": int(freeze["training"]["shared_initialization"]["seed"]),
        "subset_seed": int(freeze["training"]["shared_initialization"]["seed"]),
        "batch_size": int(common["batch_size"]),
        "epochs": int(common["epochs"]),
        "precision": str(common["precision"]),
        "optimizer": str(common["optimizer"]),
        "learning_rate": float(common["learning_rate"]),
        "weight_decay": float(common["weight_decay"]),
        "fused_adamw_requested": bool(common["fused_adamw_requested"]),
        "gradient_clip": float(common["gradient_clip"]),
        "workers": int(common["workers"]),
        "accumulation_steps": int(common["gradient_accumulation_steps"]),
        "data_loader": {
            "pin_memory": bool(common["data_loader"]["pin_memory"]),
            "persistent_workers": bool(common["data_loader"]["persistent_workers"]),
            "prefetch_factor": int(common["data_loader"]["prefetch_factor"]),
        },
        "scheduler": {
            "type": str(common["scheduler"]["type"]),
            "factor": float(common["scheduler"]["factor"]),
            "patience": int(common["scheduler"]["patience"]),
            "minimum_learning_rate": float(common["scheduler"]["minimum_learning_rate"]),
        },
        "early_stopping_enabled": False,
        "inference_flow_steps": int(freeze["posterior_sampling"]["continuous_policy"]["steps"]),
        "inference_solver": str(freeze["posterior_sampling"]["continuous_policy"]["solver"]),
    }
    if actual != expected:
        raise ValueError(f"KNOT {variant} config changed from frozen campaign: {actual}")
    frozen_data = freeze["data"]
    input_paths = {
        "measured_input": Path(config.data.input).resolve(),
        "tensor_cache": Path(config.data.tensor_cache).resolve(),
        "visible_truth": Path(config.data.visible_truth).resolve(),
        "invisible_truth": Path(config.data.invisible.path).resolve(),
        "partonic_truth": Path(config.data.partonic.path).resolve(),
    }
    expected_paths = {
        "measured_input": Path(frozen_data["measured_input"]).resolve(),
        "tensor_cache": Path(frozen_data["shared_measured_cache"]).resolve(),
        "visible_truth": Path(frozen_data["authoritative_supervision"]).resolve(),
        "invisible_truth": Path(frozen_data["authoritative_supervision"]).resolve(),
        "partonic_truth": Path(frozen_data["authoritative_supervision"]).resolve(),
    }
    if input_paths != expected_paths:
        raise ValueError(f"KNOT {variant} input paths differ from matched freeze: {input_paths}")
    if research["training_conditioning"] != "truth":
        raise ValueError(f"KNOT {variant} truth-conditioning differs from frozen campaign")
    expected_losses = (
        {"assignment": 1.0, "invisible": 1.0, "parent": 1.0, "joint": 0.0}
        if variant in {"M", "C"}
        else {"assignment": 1.0, "invisible": 0.0, "parent": 0.0, "joint": 1.0}
    )
    if research["loss_weights"] != expected_losses:
        raise ValueError(f"KNOT {variant} loss weights differ from the matched definition")
    if int(research["steps"]) != int(freeze["posterior_sampling"]["continuous_policy"]["steps"]):
        raise ValueError(f"KNOT {variant} flow-step budget differs from frozen campaign")


def run_smoke_prediction(module, data, output, variant):
    batch = next(iter(data.val_dataloader()))
    device = torch.device("cuda")
    module = module.to(device).eval()
    batch_device = move_to_device(batch, device)
    clean = truth_free_batch(batch_device)
    generator = torch.Generator(device=device).manual_seed(42011)
    posterior = module.posterior(clean, samples=2, generator=generator)
    posterior.write(output / "smoke_posterior.h5")
    joint_n = posterior.invisible[:, :, 0, 3]
    joint_p = posterior.parents[:, :, 0, 3]
    joint_covariance = ((joint_n - joint_n.mean(1, keepdim=True)) * (joint_p - joint_p.mean(1, keepdim=True))).mean(1)
    result = {
        "posterior_prediction": True,
        "posterior_samples": 2,
        "posterior_factorisation": posterior.factorisation,
        "joint_evaluator": {
            "executed": True,
            "finite_samplewise_N_P_covariance": bool(torch.isfinite(joint_covariance).all()),
            "mean_samplewise_N_P_covariance": float(joint_covariance.mean().cpu()),
        },
    }
    if variant == "C":
        result["oracle_execution"] = {}
        for endpoint in (
            "truth_assignment_truth_neutrino",
            "predicted_assignment_truth_neutrino",
        ):
            oracle = module.posterior(
                batch_device,
                samples=2,
                generator=torch.Generator(device=device).manual_seed(42011),
                oracle=endpoint,
            )
            if not torch.isfinite(oracle.parents).all():
                raise FloatingPointError(f"KNOT-C oracle {endpoint} is nonfinite")
            result["oracle_execution"][endpoint] = True
        deployed = module.posterior(
            clean,
            samples=2,
            generator=torch.Generator(device=device).manual_seed(42011),
        )
        result["oracle_execution"]["deployed_predicted_assignment_neutrino"] = bool(
            torch.isfinite(deployed.parents).all()
        )
        if not result["oracle_execution"]["deployed_predicted_assignment_neutrino"]:
            raise FloatingPointError("KNOT-C deployed prediction is nonfinite")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--research", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--steps", type=int, help="Bounded optimizer-step cap")
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.steps is not None and args.steps <= 0) or args.batches <= 0:
        parser.error("Positive steps/batches required")
    if args.smoke and (args.action != "train" or args.accelerator != "gpu"):
        parser.error("--smoke requires train --accelerator gpu")
    if args.resume and args.action != "train":
        parser.error("--resume is only valid for training")

    torch.set_num_threads(1)
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT / "results/experiments/knot"):
        raise ValueError("Use isolated results/experiments/knot output")
    config = load_experiment(args.config)
    config.training.output = str(output)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    variant = str(research["variant"])
    if variant not in {"M", "C", "J"}:
        raise ValueError(f"Invalid KNOT variant in research config: {variant}")
    if args.action == "train":
        freeze = json.loads(FREEZE_PATH.read_text())
        validate_frozen_configuration(args, config, research, freeze)
        if args.resume:
            if not output.is_dir():
                raise FileNotFoundError(f"Cannot resume missing KNOT output: {output}")
            manifest_path = output / "run_manifest.json"
            last_checkpoint = output / "checkpoints/last.ckpt"
            if not manifest_path.is_file() or not last_checkpoint.is_file():
                raise FileNotFoundError("Resume requires run_manifest.json and checkpoints/last.ckpt")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("training_status") == "completed":
                raise ValueError("Refusing to resume a completed KNOT run")
            if manifest.get("source_commit") != git_value("rev-parse", "HEAD"):
                raise ValueError("Cannot resume KNOT with a different source commit")
            if manifest.get("variant") != variant:
                raise ValueError("Cannot resume a different KNOT variant")
            manifest.setdefault("resume_utc", []).append(utc_now())
        else:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite KNOT output: {output}")
            manifest = None

        input_path = Path(config.data.input).expanduser().resolve()
        with h5py.File(input_path, "r") as source:
            if "SPLIT" not in source or not np.isin(source["SPLIT"][:], (0, 1)).all():
                raise ValueError(
                    "Frozen KNOT production input must contain only train/validation SPLIT rows"
                )
        data = build_data(config, load_visible_config(config), device="cpu", complete_assignments=True)
        data.setup("fit")
        membership = membership_summary(data)
        for split in ("train", "validation"):
            expected = freeze["data"][split]
            observed = membership[split]
            if observed["events"] != expected["events"] or observed["event_id_membership_sha256_sorted_little_endian_int64"] != expected["event_id_membership_sha256_sorted_little_endian_int64"]:
                raise ValueError(f"KNOT {split} EVENT_ID membership differs from frozen production")
        visible = load_visible_config(config)
        torch.manual_seed(int(config.experiment.seed))
        module = make_module(config, visible, research, data)
        initialization = parameter_summary(module)
        expected_shared = freeze["training"]["shared_initialization"]["parameter_tensor_sha256"]
        if initialization["shared_initialization_sha256"] != expected_shared:
            raise ValueError(
                "KNOT shared initialization differs from matched production freeze: "
                f"expected={expected_shared}, observed={initialization['shared_initialization_sha256']}"
            )
        expected_components = freeze["training"]["shared_initialization"]["components"]
        for observed_name, expected_name in (
            ("visible", "measured_encoder_and_shared_projections"),
            ("topology_queries", "topology_queries"),
            ("reconstruction", "assignment"),
        ):
            if initialization["shared_component_sha256"][observed_name] != expected_components[expected_name]:
                raise ValueError(f"KNOT shared initialization mismatch in {observed_name}")
        expected_counts = freeze["training"]["parameter_counts"][variant]
        if (
            initialization["total_parameters"] != expected_counts["total"]
            or initialization["shared_parameters"] != expected_counts["shared"]
            or initialization["variant_specific_parameters"] != expected_counts["variant_specific"]
        ):
            raise ValueError(f"KNOT {variant} parameter counts differ from matched freeze")
        if manifest is None:
            manifest = {
                "variant": variant,
                "source_commit": git_value("rev-parse", "HEAD"),
                "source_branch": git_value("branch", "--show-current"),
                "config": str(Path(args.config).resolve()),
                "research_config": str(Path(args.research).resolve()),
                "resolved_config_yaml": OmegaConf.to_yaml(config, resolve=True),
                "membership": membership,
                "split_metadata": data.split_metadata,
                "initialization": initialization,
                "seed": int(config.experiment.seed),
                "variant_initialization_seed": int(
                    freeze["training"]["variant_initialization_seeds"][variant]
                ),
                "batch_size": int(config.training.batch_size),
                "epochs": int(config.training.epochs),
                "precision": str(config.training.precision),
                "optimizer": "AdamW",
                "learning_rate": float(config.training.learning_rate),
                "weight_decay": float(config.training.weight_decay),
                "scheduler": OmegaConf.to_container(config.training.scheduler, resolve=True),
                "posterior_sampling": freeze["posterior_sampling"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0) if args.accelerator == "gpu" else None,
                "sealed_test_used": False,
                "training_status": "running",
                "start_utc": utc_now(),
                "smoke": bool(args.smoke),
            }
        else:
            if manifest.get("membership") != membership:
                raise ValueError("Resume EVENT_ID membership differs from starting run")
            manifest["training_status"] = "running"
        output.mkdir(parents=True, exist_ok=True)
        manifest["last_update_utc"] = utc_now()
        manifest_path = output / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

        checkpoint = ModelCheckpoint(
            dirpath=output / "checkpoints",
            filename="best-{epoch:02d}-{val_selection_loss:.6f}",
            auto_insert_metric_name=False,
            monitor="val_selection_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        integrity = TrainingIntegrity(smoke=args.smoke, variant=variant)
        logger = CSVLogger(save_dir=str(output), name="logs", version="")
        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_steps=args.steps if args.steps is not None else -1,
            max_epochs=int(config.training.epochs),
            precision=str(config.training.precision),
            num_sanity_val_steps=0,
            limit_val_batches=1 if args.smoke else None,
            val_check_interval=args.steps if args.smoke else 1.0,
            check_val_every_n_epoch=None if args.smoke else 1,
            logger=logger,
            callbacks=[checkpoint, integrity],
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=output,
            gradient_clip_val=float(config.training.gradient_clip),
        )
        torch.manual_seed(int(config.experiment.seed))
        started = time.perf_counter()
        trainer.fit(
            module,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
            ckpt_path=str(output / "checkpoints/last.ckpt") if args.resume else None,
        )
        trainer.save_checkpoint(output / "final.ckpt")
        if args.smoke:
            best_path = Path(checkpoint.best_model_path)
            if not best_path.is_file() or not (output / "checkpoints/last.ckpt").is_file():
                raise RuntimeError("KNOT GPU smoke did not save best and last checkpoints")
            saved = torch.load(best_path, map_location="cpu", weights_only=False)
            restored = make_module(config, visible, research, data)
            restored.validate_checkpoint_metadata(saved)
            restored.load_state_dict(saved["state_dict"])
            manifest["smoke_validation"] = {
                "best_checkpoint_reload": True,
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(output / "checkpoints/last.ckpt"),
                "last_intended_gradient_norms": integrity.last_gradient_norms,
                "prediction_and_evaluator": run_smoke_prediction(restored, data, output, variant),
            }
        manifest["training_status"] = "completed"
        manifest["end_utc"] = utc_now()
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["best_checkpoint"] = checkpoint.best_model_path
        manifest["last_checkpoint"] = str(output / "checkpoints/last.ckpt")
        manifest["optimizer_execution"] = module.hparams.get("optimizer_execution")
        manifest["maximum_cuda_allocated_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if args.accelerator == "gpu" else None
        )
        manifest["sealed_test_used"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite KNOT prediction output: {output}")
        if not args.checkpoint:
            parser.error("--checkpoint required for prediction")
        output.mkdir(parents=True)
        config = load_experiment(args.config)
        config.training.output = str(output)
        visible = load_visible_config(config)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        module = KnotModule(
            config,
            visible,
            research=OmegaConf.to_container(OmegaConf.load(args.research), resolve=True),
            invisible_scaler=InvisibleScaler.from_dict(saved["invisible_scaler"]),
            partonic_scaler=PartonicScaler.from_dict(saved["partonic_scaler"]),
            synthetic_data=bool(config.data.invisible.allow_synthetic),
        )
        module.validate_checkpoint_metadata(saved)
        module.load_state_dict(saved["state_dict"])
        module.eval()
        data = build_data(config, visible, include_truth=False, device="cpu", complete_assignments=True)
        data.setup("fit")
        generator = torch.Generator().manual_seed(int(config.experiment.seed))
        for index, batch in enumerate(data.val_dataloader()):
            if index >= args.batches:
                break
            module.posterior(batch, samples=args.samples, generator=generator).write(
                output / f"posterior_batch_{index}.h5"
            )


if __name__ == "__main__":
    main()
