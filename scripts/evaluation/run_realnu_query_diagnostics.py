#!/usr/bin/env python3
"""Export topology queries and run fixed-source query interventions on validation data."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
OUTPUT_ROOT = PRODUCTION_ROOT / "evaluation_v3_diagnostics/query_diagnostics"
MODES = ("PCFM_Q", "Q1")
EVENT_COUNT = 256
STEPS = 8


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _wrap_phi(values: torch.Tensor) -> torch.Tensor:
    return torch.remainder(values + torch.pi, 2.0 * torch.pi) - torch.pi


def _derive_visible(event_model, encoder_output: dict, query_states: torch.Tensor, batch: dict) -> dict:
    block = event_model.topology_queries
    if block is None:
        raise RuntimeError("Query diagnostic requires the shared topology-query block")
    mean_query = query_states.mean(dim=1)
    global_context = encoder_output["pooled"] + block.event_context(mean_query)
    node_context = {name: query_states[:, index] for name, index in block.node_query_indices}
    invisible_source = query_states[:, block.invisible_query_indices]
    if len(block.invisible_query_indices):
        if block.invisible_projection is None:
            raise RuntimeError("Invisible role queries lack their projection")
        invisible_context = (
            block.invisible_projection(invisible_source)
            + block.invisible_role_embedding.unsqueeze(0)
        )
    else:
        invisible_context = query_states.new_empty((len(query_states), 0, block.d_model))
    parent_context = query_states[:, block.parent_query_indices]
    result = dict(encoder_output)
    result.update({
        "query_states": query_states,
        "query_summary": block.summary(mean_query),
        "query_context": global_context,
        "node_context": node_context,
        "invisible_role_names": block.invisible_role_names,
        "invisible_role_context": invisible_context,
        "parent_role_names": block.parent_role_names,
        "parent_role_context": parent_context,
        "pooled": global_context,
    })
    if "physics_vectors" in batch:
        result["physics_vectors"] = batch["physics_vectors"]
    if "met_xy" in batch:
        result["met_xy"] = batch["met_xy"]
    return result


def _intervened_states(block, states: torch.Tensor, intervention: str, role_means=None) -> torch.Tensor:
    result = states.clone()
    roles = ("top", "antitop")
    indices = tuple(block.query_names.index(name) for name in roles)
    global_indices = [
        index for index, role in enumerate(block.query_roles)
        if str(role["kind"]) == "global"
    ]
    if len(global_indices) != 1:
        raise ValueError(f"Expected one global query, found {global_indices}")
    global_index = global_indices[0]
    if intervention == "normal":
        return result
    if intervention == "zero_role_states":
        result[:, indices] = 0.0
    elif intervention == "shuffle_role_states_between_events":
        for index in indices:
            result[:, index] = torch.roll(result[:, index], shifts=1, dims=0)
    elif intervention == "swap_top_antitop_states":
        result[:, indices[0]] = states[:, indices[1]]
        result[:, indices[1]] = states[:, indices[0]]
    elif intervention == "replace_role_states_with_dataset_mean":
        if role_means is None:
            raise ValueError("Dataset-mean query intervention requires fixed query means")
        result[:, indices] = role_means.unsqueeze(0)
    elif intervention == "replace_role_states_with_global_context":
        result[:, indices] = states[:, global_index].unsqueeze(1)
    elif intervention == "remove_role_identity":
        # Operational definition: keep each event's average role content but
        # suppress the top-versus-antitop distinction in the query states.
        common = states[:, indices].mean(dim=1, keepdim=True)
        result[:, indices] = common
    else:
        raise ValueError(f"Unknown query intervention {intervention!r}")
    return result


def _query_statistics(states: np.ndarray, block) -> dict:
    names = list(block.query_names)
    top = states[:, names.index("top")].astype(np.float64)
    antitop = states[:, names.index("antitop")].astype(np.float64)
    top_norm = np.linalg.norm(top, axis=1)
    antitop_norm = np.linalg.norm(antitop, axis=1)
    cosine = np.sum(top * antitop, axis=1) / np.maximum(top_norm * antitop_norm, 1e-12)
    between_antitop = np.roll(antitop, 1, axis=0)
    return {
        "query_names": names,
        "top_norm": {"mean": float(top_norm.mean()), "std": float(top_norm.std()), "q16": float(np.quantile(top_norm, 0.16)), "q84": float(np.quantile(top_norm, 0.84))},
        "antitop_norm": {"mean": float(antitop_norm.mean()), "std": float(antitop_norm.std()), "q16": float(np.quantile(antitop_norm, 0.16)), "q84": float(np.quantile(antitop_norm, 0.84))},
        "top_event_to_event_variance_mean": float(top.var(axis=0).mean()),
        "antitop_event_to_event_variance_mean": float(antitop.var(axis=0).mean()),
        "cos_top_antitop": {"mean": float(cosine.mean()), "std": float(cosine.std()), "q16": float(np.quantile(cosine, 0.16)), "q84": float(np.quantile(cosine, 0.84))},
        "within_event_role_distance": float(np.linalg.norm(top - antitop, axis=1).mean()),
        "between_event_role_distance": float(np.linalg.norm(top - between_antitop, axis=1).mean()),
    }


def _metric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "bias": float("nan"), "std": float("nan"), "sigma68": float("nan"), "mae": float("nan"), "rmse": float("nan")}
    q16, q84 = np.quantile(values, [0.16, 0.84])
    return {
        "n": int(len(values)),
        "bias": float(values.mean()),
        "std": float(values.std()),
        "sigma68": float((q84 - q16) / 2.0),
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def _parent_observables(parents: torch.Tensor) -> dict[str, torch.Tensor]:
    pair = parents[:, 0] + parents[:, 1]
    def rapidity(vector):
        return 0.5 * torch.log(
            ((vector[:, 0] + vector[:, 3]).clamp_min(1e-8))
            / ((vector[:, 0] - vector[:, 3]).clamp_min(1e-8))
        )
    return {
        "mttbar": (pair[:, 0].square() - pair[:, 1:].square().sum(-1)).clamp_min(0).sqrt(),
        "top_pT": torch.linalg.vector_norm(parents[:, 0, 1:3], dim=-1),
        "antitop_pT": torch.linalg.vector_norm(parents[:, 1, 1:3], dim=-1),
        "top_y": rapidity(parents[:, 0]),
        "antitop_y": rapidity(parents[:, 1]),
        "top_phi": torch.atan2(parents[:, 0, 2], parents[:, 0, 1]),
        "antitop_phi": torch.atan2(parents[:, 1, 2], parents[:, 1, 1]),
    }


def _neutrino_observables(momentum: torch.Tensor) -> dict[str, torch.Tensor]:
    pt = torch.linalg.vector_norm(momentum[..., :2], dim=-1)
    phi = torch.atan2(momentum[..., 1], momentum[..., 0])
    eta = torch.asinh(momentum[..., 2] / pt.clamp_min(1e-8))
    return {"pt": pt, "pz": momentum[..., 2], "phi": phi, "eta": eta}


def _load_truth(ids: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(EVENT_TABLE, "r") as source:
        if source.attrs.get("schema") != "realnu_event_metrics_v1" or bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("Truth metrics require the validation-only accepted R2 table")
        source_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        if len(source_ids) != 100_342 or len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("Accepted R2 table has invalid validation EVENT_ID support")
        order = np.argsort(source_ids, kind="stable")
        positions = np.searchsorted(source_ids[order], ids)
        if np.any(positions >= len(order)) or not np.array_equal(source_ids[order[positions]], ids):
            raise ValueError("Query diagnostic EVENT_ID join failed")
        rows = order[positions]
        return {
            "parents": np.asarray(source["truth/parents"][rows], dtype=np.float64),
            "neutrinos": np.asarray(source["truth/neutrinos"][rows], dtype=np.float64),
            "parent_support": np.asarray(source["support/parent_roles"][rows], dtype=bool).all(axis=1),
            "neutrino_support": np.asarray(source["support/neutrino_roles"][rows], dtype=bool),
            "assignment_truth": np.asarray(source["models/Q1/assignment_truth_candidate_indices"][rows], dtype=np.int64),
            "assignment_role_support": np.asarray(source["models/Q1/assignment_role_supported"][rows], dtype=bool),
            "assignment_support": np.asarray(source["models/Q1/assignment_supported"][rows], dtype=bool),
        }


def _write_query_export(path: Path, ids: np.ndarray, visible: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as target:
        target.attrs["schema"] = "realnu_query_diagnostic_v1"
        target.attrs["split"] = "validation"
        target.attrs["truth_free_conditioning"] = True
        target.attrs["sealed_test_used"] = False
        target.attrs["query_names"] = json.dumps(list(visible["query_names"]))
        target.attrs["parent_roles"] = json.dumps(list(visible["parent_role_names"]))
        target.attrs["invisible_roles"] = json.dumps(list(visible["invisible_role_names"]))
        target.create_dataset("EVENT_ID", data=ids)
        for name, key in (
            ("QUERY_STATES", "query_states"),
            ("GLOBAL_CONTEXT", "query_context"),
            ("PARENT_ROLE_CONTEXT", "parent_role_context"),
            ("INVISIBLE_ROLE_CONTEXT", "invisible_role_context"),
        ):
            value = visible[key].detach().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f"Refusing to export non-finite {name}")
            target.create_dataset(name, data=value, compression="gzip", shuffle=True)


def _mode(mode: str) -> dict:
    os.environ["JER_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(SOURCE_ROOT)
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT / "scripts/production"))
    from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config
    import run_realnu_production as runner

    config, _, _ = runner.load_and_validate_config(
        mode, runner.MODE_CONFIGS[mode], PRODUCTION_ROOT / mode
    )
    visible_config = load_visible_config(config)
    visible_config.general.mode = "predict"
    visible_config.dataset.val_batch_size = EVENT_COUNT
    visible_config.dataset.val_num_workers = 0
    visible_config.dataset.pin_memory = False
    visible_config.dataset.persistent_workers = False
    data = build_data(config, visible_config, include_truth=False, device="cpu")
    data.setup("predict")
    data.val_dataset = Subset(data.val_dataset, range(EVENT_COUNT))
    batch = next(iter(data.val_dataloader()))
    ids = batch["event_index"].cpu().numpy().astype(np.int64, copy=False)
    with h5py.File(EVENT_TABLE, "r") as source:
        if not np.array_equal(ids, np.asarray(source["EVENT_ID"][:EVENT_COUNT], dtype=np.int64)):
            raise ValueError(f"{mode}: fixed validation EVENT_ID order differs from R2")
    record = json.loads((PRODUCTION_ROOT / mode / "production_training.json").read_text())
    module = load_trained(config, visible_config, data, str(Path(record["checkpoint"]).resolve())).cpu().eval()
    model = module.event_model
    block = model.topology_queries
    if block is None:
        raise RuntimeError(f"{mode}: topology query module is not enabled")
    with torch.no_grad():
        raw = model.visible(batch)
        query_output = block(raw["pooled"], raw["tokens"], raw["padding_mask"])
        original_states = query_output.query_states
        visible_normal = _derive_visible(model, raw, original_states, batch)
    visible_normal["query_names"] = block.query_names
    export_path = OUTPUT_ROOT / f"{mode}_query_states_validation256.h5"
    _write_query_export(export_path, ids, visible_normal)
    stats = _query_statistics(original_states.cpu().numpy(), block)
    role_means = original_states[:, [block.query_names.index("top"), block.query_names.index("antitop")]].mean(dim=0)
    interventions = (
        "normal",
        "zero_role_states",
        "shuffle_role_states_between_events",
        "swap_top_antitop_states",
        "replace_role_states_with_dataset_mean",
        "replace_role_states_with_global_context",
        "remove_role_identity",
    )
    fixed_parent_source = torch.randn(
        (EVENT_COUNT, 1, 8), generator=torch.Generator(device="cpu").manual_seed(20260929)
    )
    fixed_neutrino_source = None
    if mode == "Q1":
        fixed_neutrino_source = torch.randn(
            (
                EVENT_COUNT,
                1,
                model.invisible.coordinates.token_count,
                model.invisible.coordinates.state_dim,
            ),
            generator=torch.Generator(device="cpu").manual_seed(314159),
        )
    truth = _load_truth(ids)
    outcome = {"query_export": str(export_path), "query_statistics": stats, "interventions": {}}
    for intervention in interventions:
        with torch.no_grad():
            query_states = _intervened_states(block, original_states, intervention, role_means)
            visible = _derive_visible(model, raw, query_states, batch)
            if mode == "PCFM_Q":
                memory = torch.cat((visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1)
                mask = torch.cat((torch.zeros((EVENT_COUNT, 1), dtype=torch.bool), visible["padding_mask"]), dim=1)
                prediction = model.parent_cfm.sample(
                    memory,
                    mask,
                    visible["parent_role_context"],
                    samples=1,
                    steps=STEPS,
                    solver="heun",
                    source_state=fixed_parent_source,
                )
                parents = prediction["parents"][:, 0]
                reconstructed = _parent_observables(parents)
                target = torch.as_tensor(truth["parents"], dtype=torch.float32)
                truth_observables = _parent_observables(target)
                metrics = {}
                valid = truth["parent_support"]
                for name, estimate in reconstructed.items():
                    residual = (estimate - truth_observables[name]).cpu().numpy()
                    if name.endswith("phi"):
                        residual = (residual + np.pi) % (2.0 * np.pi) - np.pi
                    metrics[name] = _metric(residual[valid])
                outcome["interventions"][intervention] = {"parent_metrics": metrics}
            else:
                assignment = model.reconstruction(
                    visible["tokens"],
                    visible["pooled"],
                    batch["structured"],
                    node_context=visible.get("node_context"),
                    calculate_loss=False,
                )
                roles = ("top", "antitop")
                predicted_candidates = np.stack([
                    (assignment["assignment_probabilities"][role].argmax(1) - 1).cpu().numpy()
                    for role in roles
                ], axis=1)
                correct = predicted_candidates == truth["assignment_truth"]
                role_efficiency = {
                    role: float(correct[:, index][truth["assignment_role_support"][:, index]].mean())
                    for index, role in enumerate(roles)
                }
                complete_efficiency = float(
                    correct.all(axis=1)[truth["assignment_support"]].mean()
                )
                role_parent = model.invisible.role_parent_context(
                    assignment,
                    batch["structured"],
                    visible["pooled"],
                    assignment.get("node_context", visible.get("node_context")),
                    mode=model.invisible.assignment_mode,
                    query_role_context=visible.get("invisible_role_context"),
                    query_role_names=tuple(visible.get("invisible_role_names", ())),
                )
                parent = model.invisible.coordinates.parent_context(role_parent, visible["pooled"])
                neutrino_samples, nfe = model.invisible.sample(
                    visible,
                    parent,
                    samples=1,
                    budget=min(STEPS, int(model.invisible.inference_budget)),
                    sampler=str(model.invisible.sampler),
                    source_state=fixed_neutrino_source,
                )
                neutrino = model.invisible.decode_state(neutrino_samples, batch["met_xy"], data.invisible_scaler)[:, 0]
                truth_neutrino = torch.as_tensor(truth["neutrinos"], dtype=torch.float32)
                predicted_kin = _neutrino_observables(neutrino)
                truth_kin = _neutrino_observables(truth_neutrino)
                support = truth["neutrino_support"]
                neutrino_metrics = {}
                for index, role in enumerate(("top.nu", "antitop.nu")):
                    for variable in ("pz", "pt"):
                        residual = (predicted_kin[variable][:, index] - truth_kin[variable][:, index]).cpu().numpy()
                        neutrino_metrics[f"{role}_{variable}"] = _metric(residual[support[:, index]])
                    delta_phi = _wrap_phi(predicted_kin["phi"][:, index] - truth_kin["phi"][:, index])
                    delta_eta = predicted_kin["eta"][:, index] - truth_kin["eta"][:, index]
                    delta_r = torch.sqrt(delta_eta.square() + delta_phi.square()).cpu().numpy()
                    neutrino_metrics[f"{role}_deltaR"] = _metric(delta_r[support[:, index]])
                outcome["interventions"][intervention] = {
                    "assignment": {
                        "supported_complete_event_efficiency": complete_efficiency,
                        "role_efficiency": role_efficiency,
                        "n_supported": int(truth["assignment_support"].sum()),
                    },
                    "neutrino_metrics": neutrino_metrics,
                    "neutrino_sampling": {"solver": str(model.invisible.sampler), "steps": min(STEPS, int(model.invisible.inference_budget)), "nfe": int(nfe)},
                }
    normal = outcome["interventions"]["normal"]
    for name, values in outcome["interventions"].items():
        if name == "normal":
            continue
        if mode == "PCFM_Q":
            for observable, metric in values["parent_metrics"].items():
                reference = normal["parent_metrics"][observable]
                metric["delta_from_normal"] = {
                    key: float(metric[key] - reference[key]) for key in ("bias", "std", "sigma68", "mae", "rmse")
                }
        else:
            values["assignment"]["delta_complete_efficiency"] = float(
                values["assignment"]["supported_complete_event_efficiency"]
                - normal["assignment"]["supported_complete_event_efficiency"]
            )
            values["assignment"]["delta_role_efficiency"] = {
                role: float(values["assignment"]["role_efficiency"][role] - normal["assignment"]["role_efficiency"][role])
                for role in ("top", "antitop")
            }
            for observable, metric in values["neutrino_metrics"].items():
                reference = normal["neutrino_metrics"][observable]
                metric["delta_from_normal"] = {
                    key: float(metric[key] - reference[key]) for key in ("bias", "std", "sigma68", "mae", "rmse")
                }
    outcome["event_count"] = EVENT_COUNT
    outcome["event_id_sha256"] = _sha256_ids(ids)
    outcome["intervention_definition"] = {
        "remove_role_identity": "replace both top and antitop query states with their per-event mean; parent-CFM learned role embeddings remain intact",
        "source_state": "same fixed explicit source for every intervention",
        "truth": "used only after prediction for validation metrics; never used for conditioning",
    }
    return outcome


def run() -> dict:
    torch.set_num_threads(4)
    if not EVENT_TABLE.is_file():
        raise FileNotFoundError(f"Accepted R2 event table is missing: {EVENT_TABLE}")
    result = {
        "status": "passed",
        "split": "validation",
        "event_count_per_mode": EVENT_COUNT,
        "parent_roles": ["top", "antitop"],
        "neutrino_roles": ["top.nu", "antitop.nu"],
        "sealed_test_used": False,
        "truth_free_query_export": True,
        "models": {mode: _mode(mode) for mode in MODES},
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "query_interventions.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["output"] = str(path)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
