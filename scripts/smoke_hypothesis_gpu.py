#!/usr/bin/env python3
"""Run one real GPU smoke of the selected visible/invisible hypothesis path."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import torch

from joint_event_reconstruction.cli.run import (
    build_data,
    load_trained,
)
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.hypothesis import write_hypothesis_group
from joint_event_reconstruction.prediction import (
    decode_batch,
    node_indices_for_hypothesis,
    predict_hdf5,
    to_device,
)
from joint_event_reconstruction.prediction_schema import (
    validate_hypothesis_group,
    validate_prediction,
)
from joint_event_reconstruction.topology.physics import (
    HypothesisEvaluator,
    LearnedHypothesisScorer,
    PhysicsFeatureScaler,
)


ROOT = Path(__file__).resolve().parents[1]
VISIBLE_EXPERIMENT = ROOT / (
    "results/studies/encoder_backbone/ttbar2L/lgatr/seed_42/experiment.yaml"
)
VISIBLE_CHECKPOINT = ROOT / (
    "results/studies/encoder_backbone/ttbar2L/lgatr/seed_42/train/checkpoints/best-loss.ckpt"
)
CFM_EXPERIMENT = ROOT / (
    "results/studies/generator_formulations/ttbar2L/cfm/seed_42/experiment.yaml"
)
CFM_CHECKPOINT = ROOT / (
    "results/studies/generator_formulations/ttbar2L/cfm/seed_42/train/checkpoints/best-loss.ckpt"
)


def _overrides() -> list[str]:
    return [
        "data.events.train=16",
        "data.events.validation=16",
        "data.events.test=16",
        "training.batch_size=16",
        "training.workers=0",
        "inference.seed=424242",
        "inference.assignment.top_k=5",
        "inference.invisible.samples_per_assignment=8",
        "inference.invisible.flow.steps=4",
        "inference.invisible.flow.solver=heun",
    ]


def _load_module(experiment_path: Path, checkpoint: Path, device: str):
    config = load_experiment(experiment_path, _overrides())
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("predict")
    module = load_trained(config, visible, data, str(checkpoint)).to(device).eval()
    return config, data, module


def _physics_feature_names(spec) -> tuple[str, ...]:
    names: list[str] = []
    for observable in spec.physics.observables:
        if observable.kind == "transverse_sum_residual":
            names.extend((f"{observable.name}.px", f"{observable.name}.py"))
        else:
            names.append(observable.name)
    return tuple(names)


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")


def _assert_probability(name: str, value: torch.Tensor) -> None:
    _finite_tensor(name, value)
    if not torch.allclose(value.sum(dim=1), torch.ones(value.shape[0], device=value.device), atol=2e-5):
        raise RuntimeError(f"{name} is not normalised over valid hypotheses")


def run(output_dir: Path, device: str, expected_source_sha: str | None = None) -> None:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The hypothesis smoke requires a CUDA device")
    for path in (VISIBLE_EXPERIMENT, VISIBLE_CHECKPOINT, CFM_EXPERIMENT, CFM_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise RuntimeError(
            f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {source_sha}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"GPU smoke requires a clean checkout; found {dirty!r}")
    print(f"SOURCE_SHA={source_sha}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(0)}", flush=True)

    visible_config, visible_data, visible_module = _load_module(
        VISIBLE_EXPERIMENT, VISIBLE_CHECKPOINT, device
    )
    del visible_config, visible_data, visible_module
    gc.collect()
    torch.cuda.empty_cache()
    print("PASS: selected L-GATr checkpoint strict load", flush=True)

    config, data, module = _load_module(CFM_EXPERIMENT, CFM_CHECKPOINT, device)
    if str(module.event_model.invisible.assignment_mode) != "none":
        raise RuntimeError("The CFM smoke checkpoint is not unconditioned")
    print("PASS: selected CFM checkpoint strict load", flush=True)

    prediction_path = output_dir / "prediction.h5"
    sampling_calls = 0
    original_sample = module.event_model.invisible.sample

    def counted_sample(*args, **kwargs):
        nonlocal sampling_calls
        sampling_calls += 1
        return original_sample(*args, **kwargs)

    module.event_model.invisible.sample = counted_sample
    started = time.perf_counter()
    predict_hdf5(
        module,
        data,
        data.invisible_scaler,
        prediction_path,
        split="validation",
        top_k=5,
        posterior_samples=8,
        device=device,
        prediction_seed=424242,
        checkpoint_path=str(CFM_CHECKPOINT),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    validate_prediction(prediction_path)

    with h5py.File(prediction_path, "r") as source:
        event_ids = source["EVENT_ID"][:]
        roots = source["VISIBLE/root_candidate_indices"][:]
        visible_scores = source["VISIBLE/assignment_log_scores"][:]
        valid = source["VISIBLE/valid_hypotheses"][:]
        truth = source["TRUTH/topk_assignment"][:]
        physical_samples = source["INVISIBLE/four_vectors"][:]
        unique_branches = int(source["INVISIBLE"].attrs["unique_assignment_conditioning_branches"])
        recorded_nfe = int(source["INVISIBLE"].attrs["actual_nfe"])
    if sampling_calls != 1:
        raise RuntimeError(f"Expected one unconditioned flow call, observed {sampling_calls}")
    if unique_branches != 1:
        raise RuntimeError(f"Expected one posterior branch, observed {unique_branches}")
    if physical_samples.shape[1:4] != (5, 8):
        raise RuntimeError(f"Unexpected physical posterior shape {physical_samples.shape}")
    print("PASS: one unconditioned posterior branch reused for K=5", flush=True)

    batch = next(iter(data.val_dataloader()))
    batch = to_device(batch, device)
    if not torch.equal(batch["event_index"].detach().cpu(), torch.from_numpy(event_ids)):
        raise RuntimeError("Prediction EVENT_ID order disagrees with validation batch")
    with torch.inference_mode():
        visible_output = module(batch)
        decoded = decode_batch(module.spec, batch, visible_output, 5)
    node_indices = torch.stack(
        [
            torch.stack(
                [
                    node_indices_for_hypothesis(module.spec, batch, event, hypothesis)
                    for hypothesis in event
                ],
            )
            for event, hypotheses in enumerate(decoded)
        ]
    )
    if node_indices.shape[:2] != (len(event_ids), 5):
        raise RuntimeError(f"Unexpected full hypothesis state shape {node_indices.shape}")

    feature_names = _physics_feature_names(module.spec)
    synthetic_train_residuals = torch.arange(
        2 * len(feature_names), dtype=torch.float32
    ).reshape(2, len(feature_names))
    scaler = PhysicsFeatureScaler.fit(
        synthetic_train_residuals,
        feature_names,
        topology_signature=module.spec.signature,
        physics_signature=module.spec.physics_signature,
        source_data_identity="gpu-smoke-synthetic-train-contract",
    )
    scaler = PhysicsFeatureScaler.from_dict(scaler.as_dict())
    physical = torch.from_numpy(physical_samples).to(device=device, dtype=torch.float32)
    score = torch.from_numpy(visible_scores).to(device=device, dtype=torch.float32)
    valid_tensor = torch.from_numpy(valid).to(device=device, dtype=torch.bool)
    physics_evaluator = HypothesisEvaluator(
        module.spec, scaler=scaler, mode="physics", alpha=1.0
    )
    analytic = physics_evaluator.evaluate(
        score,
        batch["structured"]["pools"],
        batch["physics_vectors"],
        node_indices,
        invisible_four_vectors=physical,
        observed_met=batch["met_xy"],
        valid_hypotheses=valid_tensor,
    )
    _finite_tensor("analytic physics score", analytic.analytic_score[valid_tensor])
    _assert_probability("analytic final probability", analytic.final_probability)
    print("PASS: topology kinematics, physics features, reducer and analytic score", flush=True)

    scorer = LearnedHypothesisScorer(3 * len(feature_names) + 1).to(device)
    learned_evaluator = HypothesisEvaluator(
        module.spec, scaler=scaler, mode="learned", scorer=scorer
    )
    learned = learned_evaluator.evaluate(
        score,
        batch["structured"]["pools"],
        batch["physics_vectors"],
        node_indices,
        invisible_four_vectors=physical,
        observed_met=batch["met_xy"],
        valid_hypotheses=valid_tensor,
    )
    loss = -learned.final_log_score[valid_tensor].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in scorer.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Learned hypothesis scorer produced non-finite gradients")
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1.0e-3)
    optimizer.step()
    scorer_path = output_dir / "learned_scorer.pt"
    torch.save(scorer.state_dict(), scorer_path)
    reloaded = LearnedHypothesisScorer(3 * len(feature_names) + 1).to(device)
    reloaded.load_state_dict(torch.load(scorer_path, map_location=device, weights_only=True))
    with torch.no_grad():
        reduced = learned.reduction.as_tensor()
        if not torch.allclose(scorer(reduced), reloaded(reduced), atol=1e-6, rtol=1e-6):
            raise RuntimeError("Reloaded learned scorer is not deterministic")
    _assert_probability("learned final probability", learned.final_probability)
    print("PASS: learned scorer forward/backward/checkpoint reload", flush=True)

    hypotheses_path = output_dir / "prediction_with_hypotheses.h5"
    shutil.copy2(prediction_path, hypotheses_path)
    with h5py.File(hypotheses_path, "r+") as destination:
        destination.attrs["hypothesis_smoke_source_sha"] = source_sha
        write_hypothesis_group(
            destination,
            states=node_indices,
            truth=torch.from_numpy(truth).to(device=device, dtype=torch.bool),
            evaluation=learned,
            scaler_identity=scaler.identity_sha256,
            scorer_type="learned_residual_smoke",
            alpha=1.0,
        )
        validate_hypothesis_group(destination, count=len(event_ids), top_k=5)
    validate_prediction(hypotheses_path)
    with h5py.File(hypotheses_path, "r") as source:
        validate_hypothesis_group(source, count=len(event_ids), top_k=5)
        physics_signature = source["HYPOTHESES"].attrs["physics_signature"]
    if str(physics_signature) != str(module.spec.physics_signature):
        raise RuntimeError("Reloaded HYPOTHESES physics signature does not match topology")
    print("PASS: HYPOTHESES HDF5 write, schema validation and reload", flush=True)

    print(json.dumps({
        "elapsed_seconds": elapsed,
        "gpu_peak_memory_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "K": 5,
        "M": 8,
        "flow_steps": recorded_nfe,
        "posterior_branches": unique_branches,
        "prediction": str(prediction_path),
        "prediction_with_hypotheses": str(hypotheses_path),
        "physics_signature": str(module.spec.physics_signature),
        "status": "PASS",
    }, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-source-sha")
    args = parser.parse_args()
    run(
        args.output_dir.expanduser().resolve(),
        args.device,
        expected_source_sha=args.expected_source_sha,
    )


if __name__ == "__main__":
    main()
