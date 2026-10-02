"""Stream train+validation and check truth support using KNOT's candidate state."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
)
from scripts.data.adapt_partonic_ttbar2l_v1 import measured_event_arrays
from scripts.validation.validate_knot_real_truth_contract import splitmix64

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "results/experiments/knot/scientific-closure-run2/real_sample/truth_contract.json"
CONFIG = ROOT / "configs/experiments/realnu_v1/j_an.public.yaml"
OUTPUT = ROOT / "results/experiments/knot/scientific-closure-run2/real_sample/model_native_assignment_support_train_validation.json"
SPLIT_SEED = 20260924
CHUNK_SIZE = 2048
MAX_UNSUPPORTED_IDS = 20


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def type_layout(data):
    values = []
    for source_name, source in data.schema.sources.items():
        if source_name == "GLOBAL":
            continue
        width = int(source.num_particles or 1)
        values.extend([int(data.schema.source_type_ids[source_name])] * width)
    if data.schema.has_global_token:
        values.append(int(data.schema.source_type_ids["GLOBAL"]))
    return torch.as_tensor(values, dtype=torch.long)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    prior = json.loads(CONTRACT.read_text())
    source_path = Path(prior["source"]).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Required KNOT source is missing: {source_path}")
    if sha256(source_path) != prior["source_sha256"]:
        raise ValueError(f"Source identity changed since the bounded truth contract: {source_path}")

    torch.set_num_threads(1)
    config = load_experiment(CONFIG)
    data = build_data(
        config,
        load_visible_config(config),
        device="cpu",
        complete_assignments=True,
    )
    collator = data.structured_collator
    type_ids_1d = type_layout(data)
    total_tokens = len(type_ids_1d)
    token_width = int(data.schema.dim)
    jet_width = int(data.schema.sources["JET"].num_particles)
    lepton_width = int(data.schema.sources["LEPTON"].num_particles)
    source_ids = dict(data.schema.source_type_ids)
    jet_slots = (type_ids_1d == source_ids["JET"]).nonzero().flatten()
    lepton_slots = (type_ids_1d == source_ids["LEPTON"]).nonzero().flatten()
    met_slots = (type_ids_1d == source_ids["MET"]).nonzero().flatten()
    if len(jet_slots) != jet_width or len(lepton_slots) != lepton_width:
        raise RuntimeError("Resolved KNOT input widths do not match its token layout")

    counts = {"train": 0, "validation": 0}
    supported_total = 0
    unsupported_causes = {
        "candidate_truncation": 0,
        "missing_truth_role": 0,
        "feasibility_topology_exclusion": 0,
        "jet_multiplicity_candidate_availability": 0,
        "other": 0,
    }
    source_events = None
    unsupported_ids = []

    with h5py.File(source_path, "r") as source:
        required = (
            "META/eventNumber", "INPUTS/JET", "INPUTS/ELECTRON", "INPUTS/MUON",
            "INPUTS/GLOBAL", "LABELS/JET", "LABELS/ELECTRON", "LABELS/MUON",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{source_path}: missing required model-support inputs {missing}")
        source_events = len(source["META/eventNumber"])
        if source_events != int(prior["source_events"]):
            raise ValueError("Source event count changed since the bounded truth-contract check")

        for start in range(0, source_events, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, source_events)
            event_ids = np.asarray(source["META/eventNumber"][start:stop], dtype=np.int64)
            bucket = (splitmix64(event_ids, SPLIT_SEED) % 100).astype(np.uint8)
            selected = np.flatnonzero(bucket < 95)
            if not len(selected):
                continue
            rows = selected + start
            ids = event_ids[selected]
            split_values = bucket[selected]
            jets = source["INPUTS/JET"][rows, :jet_width]
            all_jet_labels = np.asarray(source["LABELS/JET"][rows], dtype=np.float32)
            jet_labels = all_jet_labels[:, :jet_width]
            electrons = source["INPUTS/ELECTRON"][rows]
            muons = source["INPUTS/MUON"][rows]
            globals_chunk = source["INPUTS/GLOBAL"][rows]
            electron_labels = np.asarray(source["LABELS/ELECTRON"][rows], dtype=np.float32)
            muon_labels = np.asarray(source["LABELS/MUON"][rows], dtype=np.float32)

            leptons, _ = measured_event_arrays(electrons, muons, globals_chunk)
            raw_lepton_labels = np.concatenate((electron_labels, muon_labels), axis=1)
            source_type = np.rint(leptons["source_type"]).astype(np.int64)
            source_index = np.rint(leptons["source_index"]).astype(np.int64)
            ordered_lepton_labels = np.empty((len(rows), lepton_width), dtype=np.float32)
            for slot in range(lepton_width):
                column = source_index[:, slot] + np.where(source_type[:, slot] == 1, 0, 2)
                ordered_lepton_labels[:, slot] = raw_lepton_labels[np.arange(len(rows)), column]

            jet_valid = np.isfinite(jets["pt"]) & (jets["pt"] > 0.0)
            lepton_valid = np.isfinite(leptons["pt"]) & (leptons["pt"] > 0.0)
            padding = np.zeros((len(rows), total_tokens), dtype=np.bool_)
            padding[:, jet_slots.numpy()] = ~jet_valid
            padding[:, lepton_slots.numpy()] = ~lepton_valid
            raw_labels = np.full((len(rows), total_tokens), np.nan, dtype=np.float32)
            raw_labels[:, jet_slots.numpy()] = jet_labels
            raw_labels[:, lepton_slots.numpy()] = ordered_lepton_labels
            if len(met_slots):
                padding[:, met_slots.numpy()] = False

            batch = {
                "tokens": torch.zeros((len(rows), total_tokens, token_width), dtype=torch.float32),
                "type_ids": type_ids_1d.unsqueeze(0).expand(len(rows), -1).clone(),
                "padding_mask": torch.from_numpy(padding),
                "reco_raw_labels": torch.from_numpy(raw_labels),
                "event_index": torch.as_tensor(ids, dtype=torch.long),
            }
            structured = collator(batch)["structured"]
            assignments = structured["assignments"]
            roots = assignments["roots"]
            root_support = torch.stack(
                [structured["pools"][root]["reconstructable"] for root in roots], dim=1
            )
            full_truth_roots = torch.stack(
                [structured["pools"][root]["truth_index"] for root in roots], dim=1
            )
            matches = (
                assignments["root_candidate_indices"].unsqueeze(0)
                == full_truth_roots.unsqueeze(1)
            ).all(dim=2) & assignments["valid_mask"]
            exact_match_count = matches.sum(dim=1)
            supported = root_support.all(dim=1) & (exact_match_count == 1)
            supported_total += int(supported.sum())
            counts["train"] += int((split_values < 90).sum())
            counts["validation"] += int(((split_values >= 90) & (split_values < 95)).sum())

            unsupported = ~supported
            if not unsupported.any():
                continue
            n_kept = jet_valid.sum(axis=1)
            all_top = (all_jet_labels == 1.0).any(axis=1)
            all_antitop = (all_jet_labels == 2.0).any(axis=1)
            kept_top = (jet_labels == 1.0).any(axis=1)
            kept_antitop = (jet_labels == 2.0).any(axis=1)
            causes = np.full(len(rows), "other", dtype=object)
            truncated = (~kept_top & all_top) | (~kept_antitop & all_antitop)
            causes[truncated] = "candidate_truncation"
            missing_role = (
                (~all_top | ~all_antitop)
                | ~np.isin(ordered_lepton_labels, (1.0, 2.0)).all(axis=1)
            )
            causes[unsupported.cpu().numpy() & missing_role & ~truncated] = "missing_truth_role"
            low_multiplicity = n_kept < 2
            causes[
                unsupported.cpu().numpy() & low_multiplicity & ~truncated & ~missing_role
            ] = "jet_multiplicity_candidate_availability"
            topology_excluded = (
                unsupported.cpu().numpy()
                & ~truncated
                & ~missing_role
                & ~low_multiplicity
                & root_support.all(dim=1).cpu().numpy()
            )
            causes[topology_excluded] = "feasibility_topology_exclusion"
            for cause, number in zip(*np.unique(causes[unsupported.cpu().numpy()], return_counts=True)):
                unsupported_causes[str(cause)] += int(number)
            if len(unsupported_ids) < MAX_UNSUPPORTED_IDS:
                unsupported_ids.extend(
                    ids[unsupported.cpu().numpy()].tolist()[
                        : MAX_UNSUPPORTED_IDS - len(unsupported_ids)
                    ]
                )

    total = counts["train"] + counts["validation"]
    if total == 0:
        raise ValueError("The declared train+validation split is empty")
    result = {
        "source": str(source_path),
        "source_sha256": prior["source_sha256"],
        "source_events": source_events,
        "assignment_implementation": "StructuredCollator + AssignmentTemplate.batch + actual structured candidate masks",
        "topology": str(Path(config.topology_config).resolve()),
        "max_jets": jet_width,
        "train_events_checked": counts["train"],
        "validation_events_checked": counts["validation"],
        "train_validation_events_checked": total,
        "truth_assignment_supported": supported_total,
        "truth_assignment_unsupported": total - supported_total,
        "model_native_truth_support_fraction": supported_total / total,
        "unsupported_causes": unsupported_causes,
        "unsupported_event_id_examples": unsupported_ids,
        "split": prior["split"],
        "split_seed": SPLIT_SEED,
        "reserved_test_used": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
