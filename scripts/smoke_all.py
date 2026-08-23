#!/usr/bin/env python3
"""Real-data smoke matrix for visible, generative and coupled configurations."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import validate_experiment
from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_complete
from joint_event_reconstruction.training.joint import JointEventModule


MODES = (
    "visible_m1", "visible_recurrent_m2", "visible_feedback_m3",
    "standard_no_assignment", "edm_no_assignment", "standard_soft_c1", "edm_soft_c1",
    "edm_hard_diagnostic", "edm_truth_oracle", "edm_c2_detached", "topk_edm",
    "joint_compatibility",
)


def move(value, device):
    if torch.is_tensor(value): return value.to(device)
    if isinstance(value, dict): return {key: move(item, device) for key, item in value.items()}
    return value


def select_events(batch, selection):
    selected = {}
    event_count = len(batch["tokens"])
    for key, value in batch.items():
        if key == "structured":
            pools = {}
            for name, pool in value["pools"].items():
                pools[name] = {
                    field: (item[selection] if torch.is_tensor(item) and item.ndim and item.shape[0] == event_count
                            and field in {"candidate_mask", "truth_index", "reconstructable"} else item)
                    for field, item in pool.items()
                }
            selected[key] = {"pools": pools}
        elif torch.is_tensor(value) and value.ndim and value.shape[0] == event_count:
            selected[key] = value[selection]
        else:
            selected[key] = value
    return selected


def configure(base, mode):
    config = copy.deepcopy(base)
    config.model.visible.layers = 1
    config.model.reasoning.block_layers = 1
    config.model.reasoning.train_steps = [1, 2]
    config.model.reasoning.eval_steps = 2
    config.model.invisible.denoiser.d_model = 32
    config.model.invisible.denoiser.n_heads = 4
    config.model.invisible.denoiser.layers = 1
    config.model.invisible.denoiser.d_feedforward = 64
    config.model.invisible.denoiser.zero_init_output = False
    config.model.invisible.standard.training_steps = 20
    config.model.invisible.standard.inference_steps = 3
    config.model.invisible.edm.sigma_max = 10.0
    config.model.invisible.edm.inference_nfe = 5
    recurrent = mode in {"visible_recurrent_m2", "visible_feedback_m3", "standard_soft_c1", "edm_soft_c1", "edm_c2_detached", "joint_compatibility"}
    feedback = mode in {"visible_feedback_m3", "standard_soft_c1", "edm_soft_c1", "edm_c2_detached", "joint_compatibility"}
    invisible = mode not in {"visible_m1", "visible_recurrent_m2", "visible_feedback_m3"}
    config.model.reasoning.enabled = recurrent
    config.model.reconstruction.feedback.enabled = feedback
    config.model.invisible.enabled = invisible
    config.model.invisible.formulation = "standard" if mode.startswith("standard") else "edm"
    assignment = "none"
    if mode in {"standard_soft_c1", "edm_soft_c1", "edm_c2_detached", "topk_edm", "joint_compatibility"}: assignment = "soft"
    if mode == "edm_hard_diagnostic": assignment = "hard"
    if mode == "edm_truth_oracle": assignment = "truth"
    config.model.invisible.conditioning.assignment = assignment
    config.model.invisible.conditioning.allow_truth_oracle = mode == "edm_truth_oracle"
    config.model.coupling.rounds = 2 if mode == "edm_c2_detached" else 1
    config.model.coupling.invisible_to_reconstruction.enabled = mode == "edm_c2_detached"
    config.model.coupling.invisible_to_reconstruction.posterior_samples = 2
    config.model.compatibility.enabled = mode == "joint_compatibility"
    config.inference.top_k_assignments = 2 if mode == "topk_edm" else 1
    config.inference.posterior_samples = 2 if mode == "topk_edm" else 1
    validate_experiment(config)
    return config


def decode_batch(module, batch, output, k):
    decoded = []
    for event in range(len(batch["tokens"])):
        scored = {}
        for root in module.spec.roots:
            pool = batch["structured"]["pools"][root]
            values = []
            for index in pool["candidate_mask"][event].nonzero(as_tuple=False).flatten().tolist():
                owned = pool["leaf_ownership"][index].nonzero(as_tuple=False).flatten().tolist()
                record = CandidateRecord(
                    root, index, (), (), frozenset(("TOKEN", int(token)) for token in owned),
                    bool(index == int(pool["truth_index"][event])),
                )
                values.append(ScoredCandidate(
                    record, float(output["assignment_probabilities"][root][event, index + 1].detach().cpu()),
                ))
            scored[root] = tuple(values)
        result = decode_complete(module.spec, scored, k=k)
        if len(result.assignments) < k:
            raise RuntimeError(f"Event {event} has only {len(result.assignments)} complete assignments")
        decoded.append(result.assignments)
    return decoded


def run_mode(base, visible, batch, scaler, mode, device):
    config = configure(base, mode)
    if mode == "edm_truth_oracle":
        valid = torch.ones(len(batch["tokens"]), dtype=torch.bool, device=batch["tokens"].device)
        spec_module = JointEventModule(config, visible, invisible_scaler=scaler, synthetic_data=True)
        for role in spec_module.spec.invisible_roles:
            valid &= batch["structured"]["pools"][role.node]["reconstructable"]
        del spec_module
        if not valid.any():
            raise RuntimeError("No reconstructable event is available for the truth-oracle smoke")
        batch = select_events(batch, valid)
    module = JointEventModule(config, visible, invisible_scaler=scaler, synthetic_data=True).to(device)
    module.train()
    module.zero_grad(set_to_none=True)
    output = module(batch, recurrent_steps=2) if config.model.reasoning.enabled else module(batch)
    if not torch.isfinite(output["loss"]): raise RuntimeError("non-finite total loss")
    output["loss"].backward()
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing or non-finite gradients")
    summary = {"loss": float(output["loss"].detach()), "parameters": sum(p.numel() for p in module.parameters())}
    if mode == "visible_m1":
        module.event_model.reconstruction.pair_encoders.eval()
        differences = []
        generator = torch.Generator(device=device).manual_seed(4)
        for encoder in module.event_model.reconstruction.pair_encoders.values():
            members = torch.randn((2, 3, 2, int(config.model.visible.d_model)), generator=generator, device=device)
            differences.append(float((encoder(members) - encoder(members.flip(2))).abs().max().detach()))
        if any(value != 0 for value in differences): raise RuntimeError("pair exchange symmetry is not exact")
        summary["pair_symmetry_max_abs"] = max(differences, default=0.0)
    if mode == "visible_recurrent_m2":
        module.eval()
        with torch.inference_mode():
            for steps in (1, 2, 4, 8):
                probe = module(batch, recurrent_steps=steps)
                if probe["recurrent_steps"] != steps or not torch.isfinite(probe["loss"]):
                    raise RuntimeError(f"recurrent depth {steps} failed")
        summary["tested_steps"] = [1, 2, 4, 8]
    if mode == "visible_feedback_m3":
        feedback_gradient = sum(
            float(parameter.grad.abs().sum().detach()) for parameter in module.event_model.feedback.parameters()
            if parameter.grad is not None
        )
        if feedback_gradient <= 0: raise RuntimeError("soft feedback has no gradient")
        summary["feedback_gradient_l1"] = feedback_gradient
    if mode.startswith("standard"):
        noise = output["invisible_training"]["noise"]
        if not torch.isfinite(noise).all(): raise RuntimeError("Gaussian noise is non-finite")
        module.eval()
        with torch.inference_mode():
            generated, nfe = module.event_model.invisible.sample(
                output["visible"], output["invisible_parent_context"], samples=1, budget=3,
                generator=torch.Generator(device=device).manual_seed(8),
            )
        if nfe != 3 or not torch.isfinite(generated).all(): raise RuntimeError("DDIM sampling failed")
        summary.update({
            "nfe": nfe, "noise_mean": float(noise.mean().detach()),
            "noise_std": float(noise.std().detach()),
        })
    if mode.startswith("edm") or mode == "topk_edm":
        if "sigma" in output["invisible_training"] and not torch.isfinite(output["invisible_training"]["sigma"]).all():
            raise RuntimeError("EDM sigma is non-finite")
        summary["formulation"] = "edm"
    if mode in {"standard_soft_c1", "edm_soft_c1"}:
        gradient = sum(
            float(parameter.grad.abs().sum().detach()) for parameter in module.event_model.reconstruction.parameters()
            if parameter.grad is not None
        )
        if gradient <= 0: raise RuntimeError("soft C1 has no reconstruction gradient")
        summary["reconstruction_gradient_l1"] = gradient
    if mode == "edm_truth_oracle": summary["ORACLE"] = True
    if mode == "edm_c2_detached":
        posterior = output["posterior_feedback"]
        if not posterior["detached"] or posterior["samples"].grad_fn is not None or posterior["nfe"] != 5:
            raise RuntimeError("C2 posterior feedback is not detached or has wrong NFE")
        summary["posterior_shape"] = list(posterior["samples"].shape)
    if mode == "topk_edm":
        module.eval()
        with torch.inference_mode():
            decoded = decode_batch(module, batch, output, 2)
            contexts = module.event_model.invisible.assignment_parent_context(
                output, batch["structured"], output["visible"]["pooled"], decoded,
                output.get("node_context"),
            )
            if base.experiment.name == "ttbar2L_joint":
                role_order = [role.key for role in module.spec.invisible_roles]
                if role_order != ["top_e.nu", "top_mu.nu"] or contexts.shape[2] != 2:
                    raise RuntimeError(f"Dilepton invisible role association is wrong: {role_order}")
                embedding = module.event_model.invisible.formulation.denoiser.role_embedding
                if torch.equal(embedding[0], embedding[1]):
                    raise RuntimeError("Electron and muon neutrinos do not have distinct role embeddings")
            generated, nfe = module.event_model.invisible.sample_hypotheses(
                output["visible"], contexts, samples=2, budget=5,
                generator=torch.Generator(device=device).manual_seed(9),
            )
        if generated.shape[:3] != (len(batch["tokens"]), 2, 2) or nfe != 5:
            raise RuntimeError("K x M generation has incorrect shape or NFE")
        summary.update({"joint_shape": list(generated.shape), "nfe": nfe})
    if mode == "joint_compatibility":
        gradient = sum(
            float(parameter.grad.abs().sum().detach())
            for parameter in module.event_model.compatibility.parameters() if parameter.grad is not None
        )
        if gradient <= 0 or not torch.isfinite(output["compatibility_loss"]):
            raise RuntimeError("Compatibility scoring has no finite training gradient")
        summary.update({"compatibility_loss": float(output["compatibility_loss"].detach()), "compatibility_gradient_l1": gradient})
    return summary


def run_experiment_smoke(
    experiment_path: Path, *, device: str, events: int, modes=None,
    continue_on_failure=False, real_data: bool = False,
):
    torch.set_num_threads(1)
    base = OmegaConf.load(experiment_path)
    # The synthetic software smoke deliberately remains on the compact frozen
    # visible cache. Physics-facing runs use canonical real data.
    if base.experiment.name == "ttbar1L_joint" and not real_data:
        base.visible_config = "configs/m0_ttbar1L.yaml"
        base.data.tensor_cache = (
            "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/SB_paper/results/"
            "transformer_full_campaign_joint_first_v2/cache/"
            "transformer_ttbar1L_semantic_reconstruction_full__jets8_lep1_met1__tok17__all__filelabels__458b8dcf1959.pt"
        )
    for split in ("train", "val", "test"): base.data.max_events[split] = events
    if not real_data:
        sidecar = Path("data/synthetic") / f"{base.experiment.name}_{events}.h5"
        subprocess.run([
            sys.executable, "scripts/data/make_synthetic_invisible.py", "--experiment", str(experiment_path),
            "--output", str(sidecar), "--max-events", str(events), "--seed", str(base.experiment.seed),
            "--visible-config", str(base.visible_config), "--tensor-cache", str(base.data.tensor_cache),
        ], check=True)
        base.data.invisible.path = str(sidecar.resolve())
        base.data.invisible.allow_synthetic = True
    visible = OmegaConf.load(base.visible_config)
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = events
        visible.dataset[f"{split}_num_workers"] = 0
    visible.dataset.pin_memory = False
    visible.dataset.persistent_workers = False
    data_config = configure(base, "edm_no_assignment")
    data = StructuredDataModule(
        visible, data_config.topology_config, tensor_cache=data_config.data.tensor_cache,
        invisible_path=data_config.data.invisible.path,
        allow_synthetic=bool(data_config.data.invisible.allow_synthetic),
        max_events={"train": events, "val": events, "test": events}, subset_seed=int(base.experiment.seed),
    )
    data.setup("fit")
    batch = move(next(iter(data.train_dataloader())), torch.device(device))
    selected = MODES if modes is None else modes
    results = {}
    for mode in selected:
        try:
            results[mode] = {"status": "PASS", **run_mode(base, visible, batch, data.invisible_scaler, mode, device)}
        except Exception as error:
            results[mode] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
            if not continue_on_failure:
                print(json.dumps(results, indent=2, sort_keys=True))
                raise
    print(json.dumps(results, indent=2, sort_keys=True))
    if any(result["status"] != "PASS" for result in results.values()):
        raise RuntimeError("One or more smoke modes failed")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, choices=("ttbar1L", "ttbar2L", "ttH1L"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--mode", action="append", choices=MODES)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    run_experiment_smoke(
        Path(f"configs/experiments/{args.topology}.yaml"), device=args.device, events=args.events,
        modes=args.mode, continue_on_failure=args.continue_on_failure, real_data=args.real,
    )


if __name__ == "__main__": main()
