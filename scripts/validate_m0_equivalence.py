#!/usr/bin/env python3
"""Compare the isolated M0 implementation with the frozen reference on real data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


def _worker(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    if args.implementation == "reference":
        from src.data.DataModuleTransformer import TransformerDataModule
        from src.models.Transformer import TransformerModel
        from src.utils.TransformerPredict import _decode_batch_topologies
    else:
        from joint_event_reconstruction.m0.data.module import TransformerDataModule
        from joint_event_reconstruction.m0.model import TransformerModel
        from joint_event_reconstruction.m0.prediction import _decode_batch_topologies

    torch.manual_seed(1729)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["hyper_parameters"])
    # Historical tuning settings were operational metadata and can conflict with
    # loading the selected production checkpoint. They do not affect a forward pass.
    config.pop("optimise", None)
    config.dataset.cache.enabled = False
    for split in ("train", "val", "test"):
        config.dataset[f"{split}_batch_size"] = args.events
        config.dataset[f"{split}_num_workers"] = 0
    config.dataset.pin_memory = False
    config.dataset.persistent_workers = False

    data = TransformerDataModule(config)
    data.setup("validate")
    batch = next(iter(data.val_dataloader()))
    model = TransformerModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(
            batch["tokens"], batch["type_ids"], batch["padding_mask"],
            labels=batch["labels"], reco=batch["reco"],
        )
    payload = {
        "event_index": batch["event_index"],
        "tokens": batch["tokens"],
        "type_ids": batch["type_ids"],
        "padding_mask": batch["padding_mask"],
        "labels": batch["labels"],
        "classification_logits": output["classification_logits"],
        "encoded_tokens": output["encoded_tokens"],
        "pooled": output["pooled"],
        "loss": output["loss"],
        "classification_loss": output["classification_loss"],
        "reconstruction_loss": output["reconstruction_loss"],
        "decode": _decode_batch_topologies(output, batch, str(config.general.topology)),
        "pools": {},
    }
    for name, pool in batch["reco"]["pools"].items():
        payload["pools"][name] = {
            "candidate_indices": pool["candidate_indices"],
            "candidate_mask": pool["candidate_mask"],
            "truth_class": pool["truth_class"],
            "event_active": pool["event_active"],
            "candidate_embeddings": output["candidate_embeddings"][name],
            "candidate_attention": output["candidate_attention"][name],
            "logits": output["reconstruction_logits"][name],
        }
    torch.save(payload, args.output)


def _compare(reference: object, isolated: object, path: str, results: list[dict]) -> None:
    if torch.is_tensor(reference):
        if reference.dtype.is_floating_point:
            difference = float((reference.double() - isolated.double()).abs().max()) if reference.numel() else 0.0
            equal = torch.equal(reference, isolated)
        else:
            difference = 0.0
            equal = torch.equal(reference, isolated)
        results.append({"path": path, "exact": equal, "max_abs": difference})
        return
    if isinstance(reference, dict):
        if reference.keys() != isolated.keys():
            raise AssertionError(f"Key mismatch at {path}: {reference.keys()} != {isolated.keys()}")
        for key in reference:
            _compare(reference[key], isolated[key], f"{path}.{key}", results)
        return
    if isinstance(reference, (list, tuple)):
        if len(reference) != len(isolated):
            raise AssertionError(f"Length mismatch at {path}: {len(reference)} != {len(isolated)}")
        for index, (left, right) in enumerate(zip(reference, isolated)):
            _compare(left, right, f"{path}[{index}]", results)
        return
    if isinstance(reference, float):
        equal = (reference == isolated) or (reference != reference and isolated != isolated)
        results.append({"path": path, "exact": equal, "max_abs": 0.0 if equal else abs(reference - isolated)})
        return
    if reference != isolated:
        raise AssertionError(f"Value mismatch at {path}: {reference!r} != {isolated!r}")


def _controller(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    repository = script.parents[1]
    with tempfile.TemporaryDirectory(prefix="m0-equivalence-") as temporary:
        temporary = Path(temporary)
        outputs = {}
        for implementation, python_path in (
            ("reference", Path(args.reference_root)),
            ("isolated", repository / "src"),
        ):
            output = temporary / f"{implementation}.pt"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = (
                str(python_path) if implementation == "reference"
                else os.pathsep.join((str(python_path), str(repository)))
            )
            subprocess.run(
                [sys.executable, str(script), "--worker", "--implementation", implementation,
                 "--checkpoint", str(args.checkpoint), "--events", str(args.events),
                 "--output", str(output)],
                check=True, cwd=repository, env=environment,
            )
            outputs[implementation] = torch.load(output, map_location="cpu", weights_only=False)
        results: list[dict] = []
        _compare(outputs["reference"], outputs["isolated"], "batch", results)
        failed = [result for result in results if not result["exact"]]
        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "events": args.events,
            "tensor_comparisons": len(results),
            "non_exact": failed,
            "maximum_absolute_difference": max((result["max_abs"] for result in results), default=0.0),
            "event_indices": outputs["isolated"]["event_index"].tolist(),
            "loss": float(outputs["isolated"]["loss"]),
            "reconstruction_loss": float(outputs["isolated"]["reconstruction_loss"]),
            "decoded_valid": sum(bool(item["reco_valid"]) for item in outputs["isolated"]["decode"]),
            "decoded_truth_correct": sum(bool(item["complete_topology_truth_correct"]) for item in outputs["isolated"]["decode"]),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if failed:
            raise SystemExit("M0 reference equivalence failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--implementation", choices=("reference", "isolated"), help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
    else:
        if not args.reference_root:
            parser.error("--reference-root is required")
        _controller(args)


if __name__ == "__main__":
    main()
