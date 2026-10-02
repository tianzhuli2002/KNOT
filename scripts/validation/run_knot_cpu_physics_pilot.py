"""Matched full-architecture CPU pilot, with a measured budget and no test rows."""

import argparse
import copy
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from run_knot_real_cpu_smoke import initialization_summary

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.knot import KnotModule

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/experiments/knot/cpu-physics-pilot-v1"
SEED = 42011


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def setup():
    config = load_experiment(
        ROOT
        / "configs/experiments/knot_scientific_closure/knot_m_production.public.yaml"
    )
    config.data.max_events.train = 25000
    config.data.max_events.val = 5000
    config.training.workers = 0
    config.training.precision = "32-true"
    config.training.output = str(OUT / "runtime")
    # CPU batch remains identical across variants; encoder architecture is unchanged.
    config.training.batch_size = 128
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu", complete_assignments=True)
    data.setup("fit")
    return config, visible, data


def model(config, visible, data, variant):
    config = copy.deepcopy(config)
    config.model.parent_cfm.enabled = False
    research = OmegaConf.to_container(
        OmegaConf.load(
            ROOT / f"configs/research/knot_{variant.lower()}_production.yaml"
        ),
        resolve=True,
    )
    torch.manual_seed(SEED)
    return KnotModule(
        config,
        visible,
        research=research,
        invisible_scaler=data.invisible_scaler,
        partonic_scaler=data.partonic_scaler,
        synthetic_data=False,
    )


def checkpoint(module, optimizer, step, path):
    payload = {
        "state_dict": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "hyper_parameters": dict(module.hparams),
    }
    module.on_save_checkpoint(payload)
    torch.save(payload, path)


def validation(module, loader, max_batches=8):
    module.eval()
    values = []
    # Validation randomness is fixed and does not alter optimization RNG.
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(SEED + 100)
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            output = module(batch)
            continuous = (
                output["joint_loss"]
                if module.variant == "J"
                else (
                    module.n_invisible * output["invisible_loss"]
                    + 8 * output["partonic_cfm_loss"]
                )
                / (module.n_invisible + 8)
            )
            values.append(float(output["assignment_loss"] + continuous))
    module.train()
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["benchmark", "train"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True, parents=True)
    start = time.perf_counter()
    config, visible, data = setup()
    print(f"data setup seconds={time.perf_counter() - start:.1f}", flush=True)
    membership = {}
    for split, dataset in [
        ("train", data.train_dataset),
        ("validation", data.val_dataset),
    ]:
        ids = np.array(
            [int(dataset[i]["event_index"]) for i in range(len(dataset))],
            dtype=np.int64,
        )
        np.save(OUT / f"{split}_event_ids.npy", ids)
        membership[split] = {
            "events": len(ids),
            "sha256": hashlib.sha256(np.sort(ids).astype("<i8").tobytes()).hexdigest(),
        }
    save_json(
        OUT / "membership.json",
        {
            **membership,
            "selection": "maintained deterministic EVENT_ID subset, seed 42011",
            "sealed_test_used": False,
        },
    )
    benchmark = (
        {}
        if args.action == "benchmark"
        else json.loads((OUT / "benchmark.json").read_text())
    )
    steps = (
        50
        if args.action == "benchmark"
        else json.loads((OUT / "budget.json").read_text())["optimizer_steps"]
    )
    hashes = []
    for variant in ["M", "C", "J"]:
        module = model(config, visible, data, variant)
        init = initialization_summary(module)
        init["variant_initialization_seed"] = SEED
        hashes.append(init["shared_initialization_hash"])
        optimizer = torch.optim.AdamW(
            [p for p in module.parameters() if p.requires_grad],
            lr=module.learning_rate,
            weight_decay=module.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=8, min_lr=1e-6
        )
        train_loader = data.train_dataloader()
        train_loader.sampler.generator = torch.Generator().manual_seed(SEED)
        torch.manual_seed(SEED)
        iterator = iter(train_loader)
        module.train()
        history = []
        directory = OUT / variant
        directory.mkdir(exist_ok=True)
        OmegaConf.save(config, directory / "config.yaml")
        best = float("inf")
        begin = time.perf_counter()
        for step in range(1, steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(SEED + step)
            output = module(batch)
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError(f"{variant} step {step}: nonfinite loss")
            output["loss"].backward()
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f"{variant}: invalid gradient {norm}")
            optimizer.step()
            row = {
                "step": step,
                "loss": float(output["loss"].detach()),
                "gradient_norm": float(norm),
                "seconds": time.perf_counter() - begin,
            }
            if args.action == "train" and (step % 100 == 0 or step == steps):
                row["validation_loss"] = validation(module, data.val_dataloader())
                scheduler.step(row["validation_loss"])
                if row["validation_loss"] < best:
                    best = row["validation_loss"]
                    checkpoint(module, optimizer, step, directory / "best.ckpt")
            history.append(row)
            if step % 10 == 0:
                print(
                    f"{args.action} {variant} {step}/{steps} loss={row['loss']:.4g} seconds={row['seconds']:.1f}",
                    flush=True,
                )
        elapsed = time.perf_counter() - begin
        summary = {
            "wall_seconds": elapsed,
            "seconds_per_step": elapsed / steps,
            "steps_per_second": steps / elapsed,
            "events_per_second": steps * 128 / elapsed,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "initialization": init,
            "optimizer_steps": steps,
            "batch_size": 128,
        }
        if args.action == "benchmark":
            benchmark[variant] = summary
            save_json(OUT / "benchmark.json", benchmark)
        else:
            checkpoint(module, optimizer, steps, directory / "final.ckpt")
            save_json(directory / "history.json", history)
            save_json(directory / "training_summary.json", summary)
    if len(set(hashes)) != 1:
        raise ValueError("M/C/J shared initialization differs")
    if args.action == "benchmark":
        # Reserve 25 minutes for validation, prediction and diagnostics.
        seconds = sum(v["seconds_per_step"] for v in benchmark.values())
        budget = max(100, min(600, int(4500 / seconds) // 100 * 100))
        save_json(
            OUT / "budget.json",
            {
                "optimizer_steps": budget,
                "batch_size": 128,
                "posterior_samples": 16,
                "prediction_events": 1024,
                "expected_training_seconds": budget * seconds,
                "benchmark_steps_per_model": 50,
                "reason": "matched budget from summed CPU timings, <=75 minutes training; reserve prediction/diagnostics time",
            },
        )


if __name__ == "__main__":
    main()
