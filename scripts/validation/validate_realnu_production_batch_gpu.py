"""Measure one real-data optimiser update at the configured production batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
)
from joint_event_reconstruction.training.joint import JointEventModule


class BatchProbe:
    def __init__(self) -> None:
        self.started: float | None = None
        self.parameter: torch.nn.Parameter | None = None
        self.parameter_before: torch.Tensor | None = None
        self.batch_events: int | None = None
        self.gradient_count = 0
        self.gradient_squared_norm = 0.0
        self.gradient_finite = True
        self.loss: float | None = None
        self.elapsed_seconds: float | None = None
        self.parameter_changed = False

    def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
        if batch_idx != 0:
            return
        torch.cuda.synchronize()
        self.started = time.perf_counter()
        self.batch_events = len(batch["tokens"])

    def on_before_optimizer_step(self, trainer, module, optimizer) -> None:
        if self.started is None:
            return
        gradients = []
        for parameter in module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            self.gradient_count += int(gradient.numel())
            gradients.append((parameter, gradient))
        if not gradients:
            return
        finite = torch.stack([torch.isfinite(gradient).all() for _, gradient in gradients])
        nonzero = torch.stack([(gradient != 0).any() for _, gradient in gradients])
        norm_terms = torch.stack(
            [gradient.detach().float().square().sum() for _, gradient in gradients]
        )
        self.gradient_finite = bool(finite.all().item())
        self.gradient_squared_norm = float(norm_terms.sum().cpu())
        nonzero_positions = torch.nonzero(nonzero, as_tuple=False).flatten()
        if len(nonzero_positions):
            self.parameter, _ = gradients[int(nonzero_positions[0].item())]
            self.parameter_before = self.parameter.detach().clone()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
        if batch_idx != 0:
            return
        torch.cuda.synchronize()
        if self.started is None:
            raise RuntimeError("Production batch probe did not observe batch start")
        self.elapsed_seconds = time.perf_counter() - self.started
        loss = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs["loss"].detach()
        self.loss = float(loss.float().cpu())
        if self.parameter is None or self.parameter_before is None:
            raise RuntimeError("Production batch probe found no non-zero trainable gradient")
        self.parameter_changed = not torch.equal(self.parameter, self.parameter_before)


def _dataset_event_hash(data) -> str:
    dataset = data.train_dataset
    subset_positions = np.arange(len(dataset), dtype=np.int64)
    while isinstance(dataset, torch.utils.data.Subset):
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        subset_positions = subset_indices[subset_positions]
        dataset = dataset.dataset
    payload = getattr(dataset, "payload", None)
    if payload is None:
        payload = getattr(dataset, "cache_payload", None)
    if not hasattr(dataset, "positions") or payload is None:
        raise TypeError(
            "Production probe requires the validated cached train dataset; "
            f"observed {type(dataset).__name__}"
        )
    cache_positions = np.asarray(dataset.positions, dtype=np.int64)[subset_positions]
    event_source = payload.get("event_id", payload["indices"])
    positions = torch.as_tensor(cache_positions, dtype=torch.long)
    event_ids = event_source.index_select(0, positions).cpu().numpy().astype(np.int64, copy=False)
    if event_ids.shape != (512,):
        raise ValueError(f"Production probe train exposure has {len(event_ids)} events; expected 512")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("Production probe train exposure contains duplicate event IDs")
    digest = hashlib.sha256(np.sort(event_ids).tobytes()).hexdigest()
    return digest


def run(config_path: Path, output_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The realnu production batch probe requires an allocated CUDA GPU")
    config = load_experiment(
        config_path,
        ["training.scheduler.enabled=false", f"training.output={output_path.parent / 'probe_run'}"],
    )
    if int(config.training.batch_size) != 512:
        raise ValueError(f"Production batch probe requires batch_size=512, got {config.training.batch_size}")

    import lightning as L
    from lightning.pytorch.callbacks import Callback

    class CallbackAdapter(Callback):
        def __init__(self, probe: BatchProbe) -> None:
            self.probe = probe

        def on_train_batch_start(self, trainer, module, batch, batch_idx) -> None:
            self.probe.on_train_batch_start(trainer, module, batch, batch_idx)

        def on_before_optimizer_step(self, trainer, module, optimizer) -> None:
            self.probe.on_before_optimizer_step(trainer, module, optimizer)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx) -> None:
            self.probe.on_train_batch_end(trainer, module, outputs, batch, batch_idx)

    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(config.experiment.seed), workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cuda")
    data.setup("fit")
    train_event_ids_sha256 = _dataset_event_hash(data)
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        synthetic_data=(
            bool(data.structured_collator.invisible_store.synthetic)
            if data.structured_collator.invisible_store is not None
            else bool(
                data.structured_collator.partonic_store is not None
                and data.structured_collator.partonic_store.synthetic
            )
        ),
        partonic_scaler=data.partonic_scaler,
    )
    probe = BatchProbe()
    torch.cuda.reset_peak_memory_stats()
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        precision=str(config.training.precision),
        gradient_clip_val=float(config.training.gradient_clip),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[CallbackAdapter(probe)],
    )
    trainer.fit(module, datamodule=data)
    torch.cuda.synchronize()
    if probe.batch_events != 512:
        raise RuntimeError(f"Production optimizer update used {probe.batch_events} events; expected 512")
    if probe.loss is None or not np.isfinite(probe.loss):
        raise RuntimeError(f"Production batch probe loss is missing or non-finite: {probe.loss}")
    if not probe.gradient_finite or probe.gradient_count == 0:
        raise RuntimeError("Production batch probe gradients were absent or non-finite")
    if not np.isfinite(probe.gradient_squared_norm) or probe.gradient_squared_norm <= 0.0:
        raise RuntimeError(f"Production batch probe gradient norm is invalid: {probe.gradient_squared_norm}")
    if not probe.parameter_changed:
        raise RuntimeError("Production batch probe optimizer did not update a trainable parameter")
    result = {
        "status": "passed",
        "config": str(config_path.resolve()),
        "mode": str(config.experiment.name),
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": int(config.training.batch_size),
        "optimizer_batch_events": probe.batch_events,
        "train_event_count": len(data.train_dataset),
        "train_event_ids_sha256_sorted": train_event_ids_sha256,
        "precision": str(config.training.precision),
        "loss": probe.loss,
        "gradient_elements": probe.gradient_count,
        "gradient_l2_norm": float(np.sqrt(probe.gradient_squared_norm)),
        "gradient_finite": probe.gradient_finite,
        "parameter_updated": probe.parameter_changed,
        "optimizer_step_seconds": probe.elapsed_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
