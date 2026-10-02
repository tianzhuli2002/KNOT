"""Common M/C/J posterior, reconstruction, spin and C oracle evaluation."""

import argparse
import json
import time
from dataclasses import fields

import h5py
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_knot_cpu_physics_pilot import OUT, SEED, model, save_json, setup
from run_knot_real_cpu_smoke import prediction_batch

from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.knot import EventPosterior
from joint_event_reconstruction.knot_physics import (
    SPIN_DEFINITION,
    joint_energy_score,
    joint_metric,
    mass,
    reconstruction_metrics,
    spin_observables,
)


def concatenate(posteriors):
    arguments = {}
    for field in fields(EventPosterior):
        values = [getattr(p, field.name) for p in posteriors]
        arguments[field.name] = (
            torch.cat(values) if torch.is_tensor(values[0]) else values[0]
        )
    return EventPosterior(**arguments).validate()


def leptons_from_batch(batch, data):
    positions = (
        (batch["type_ids"][0] == data.schema.source_type_ids["LEPTON"])
        .nonzero()
        .flatten()
    )
    if len(positions) != 2:
        raise ValueError("Pilot requires exactly two charge-ordered lepton slots")
    return batch["physics_vectors"][:, positions].numpy()


def spin_summary(parents, leptons, truth):
    coherent = spin_observables(parents, leptons[:, None])
    # Break top-antitop pairing; a whole-pair permutation cannot change a P-only observable.
    permutation = np.argsort(
        np.random.default_rng(SEED + 3).random(parents.shape[:2]), axis=1
    )
    shuffled_parents = parents.copy()
    shuffled_parents[:, :, 1] = np.take_along_axis(
        parents[:, :, 1], permutation[..., None], axis=1
    )
    shuffled = spin_observables(shuffled_parents, leptons[:, None])
    exact = spin_observables(truth, leptons)
    result = {}
    for name in exact:
        difference = coherent[name].mean(axis=1) - shuffled[name].mean(axis=1)
        result[name] = {
            "truth_mean": float(exact[name].mean()),
            "coherent_mean": float(coherent[name].mean()),
            "shuffled_mean": float(shuffled[name].mean()),
            "coherent_event_mean_RMSE": float(
                np.sqrt(np.mean((coherent[name].mean(axis=1) - exact[name]) ** 2))
            ),
            "shuffled_event_mean_RMSE": float(
                np.sqrt(np.mean((shuffled[name].mean(axis=1) - exact[name]) ** 2))
            ),
            "paired_event_difference_RMS": float(np.sqrt(np.mean(difference**2))),
            "paired_mean_difference_SE": float(
                difference.std() / np.sqrt(len(difference))
            ),
        }
    return result, exact, coherent, shuffled


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["M", "C", "J"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    config, visible, data = setup()
    budget = json.loads((OUT / "budget.json").read_text())
    count, samples = budget["prediction_events"], budget["posterior_samples"]
    validation_ids = np.load(OUT / "validation_event_ids.npy")
    evaluation_positions = deterministic_subset_positions(
        validation_ids, count, SEED + 200
    )
    np.save(OUT / "prediction_event_ids.npy", validation_ids[evaluation_positions])
    loader = DataLoader(
        Subset(data.val_dataset, evaluation_positions.tolist()),
        batch_size=32,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
    )
    summary_path = OUT / "pilot_metrics.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for variant in args.variants:
        begin = time.perf_counter()
        module = model(config, visible, data, variant)
        checkpoint = torch.load(
            OUT / variant / "final.ckpt", map_location="cpu", weights_only=False
        )
        module.validate_checkpoint_metadata(checkpoint)
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()
        draws, truth_n, truth_p, lepton_rows, support_rows = [], [], [], [], []
        oracles = {
            key: []
            for key in [
                "truth_assignment_truth_neutrino",
                "predicted_assignment_truth_neutrino",
            ]
        }
        (
            truth_states,
            truth_candidates,
            per_role_correct,
            nll,
            complete_correct,
            multiplicities,
        ) = [], [], [], [], [], []
        leaf_accuracy = {
            role.key: [] for role in module.event_model.spec.visible_leaf_roles
        }
        channel_counts = {channel: 0 for channel in ["emu", "ee", "mumu"]}
        for index, batch in enumerate(loader):
            clean = prediction_batch(batch)
            posterior = module.posterior(
                clean,
                samples=samples,
                generator=torch.Generator().manual_seed(SEED + index),
            )
            draws.append(posterior)
            truth = module.event_model.invisible.decode_physical_state(
                batch["invisible_truth_scaled"][:, None],
                batch["met_xy"],
                module.event_model.invisible.scaler,
            )["four_vector"][:, 0]
            truth_n.append(truth.numpy())
            truth_p.append(batch["partonic_truth"].numpy())
            lepton_rows.append(leptons_from_batch(batch, data))
            encoded = module.event_model.encode_visible(clean)
            result, space = module.assignment(clean, encoded)
            probabilities = space.state_probabilities(result["assignment_logits"])
            true_state = batch["structured"]["assignments"]["truth_state"]
            true_candidate = space.node_indices(true_state)
            mode = probabilities.argmax(-1)
            candidate_mode = space.node_indices(mode)
            support = batch["visible_assignment_support"].numpy()
            support_rows.append(support)
            truth_states.append(true_state.numpy())
            truth_candidates.append(true_candidate.numpy())
            per_role_correct.append((candidate_mode == true_candidate).numpy())
            for role in module.event_model.spec.visible_leaf_roles:
                node_index = space.nodes.index(role.node)
                table = batch["structured"]["pools"][role.node]["leaf_indices"][
                    role.key
                ]
                predicted_index = candidate_mode[:, node_index]
                target_index = true_candidate[:, node_index]
                selected_leaf = table[predicted_index.clamp_min(0)]
                selected_leaf = torch.where(
                    predicted_index[:, None] >= 0, selected_leaf, -1
                )
                target_leaf = table[target_index.clamp_min(0)]
                leaf_accuracy[role.key].append(
                    (selected_leaf == target_leaf).all(-1).numpy()
                )
            complete_correct.append((mode == true_state).numpy())
            nll.append(
                -probabilities.gather(1, true_state[:, None])
                .clamp_min(1e-30)
                .log()[:, 0]
                .numpy()
            )
            multiplicities.append((~batch["padding_mask"][:, :8]).sum(-1).numpy())
            lepton_mask = batch["type_ids"] == data.schema.source_type_ids["LEPTON"]
            lepton_ids = batch["tokens"][..., data.schema.names.index("id:none")][
                lepton_mask
            ].reshape(-1, 2)
            electron_count = (lepton_ids == 2).sum(-1)
            for channel, number in [("emu", 1), ("ee", 2), ("mumu", 0)]:
                channel_counts[channel] += int((electron_count == number).sum())
            if variant == "C":
                for endpoint, values in oracles.items():
                    oracle = module.posterior(
                        batch,
                        samples=samples,
                        generator=torch.Generator().manual_seed(SEED + index),
                        oracle=endpoint,
                    )
                    values.append(oracle.parents.numpy())
            if index % 4 == 0:
                print(
                    f"predict {variant} batch {index + 1}/{len(loader)} elapsed={time.perf_counter() - begin:.1f}",
                    flush=True,
                )
        posterior = concatenate(draws)
        n, p = posterior.invisible.numpy(), posterior.parents.numpy()
        tn, tp, leptons = (
            np.concatenate(truth_n),
            np.concatenate(truth_p),
            np.concatenate(lepton_rows),
        )
        support = np.concatenate(support_rows)
        posterior_path = OUT / variant / "posterior_validation.h5"
        posterior.write(posterior_path)
        with h5py.File(posterior_path, "a") as f:
            f.create_dataset("TRUTH/neutrinos", data=tn)
            f.create_dataset("TRUTH/parents", data=tp)
            f.create_dataset(
                "TRUTH/assignment_state", data=np.concatenate(truth_states)
            )
            f.create_dataset(
                "TRUTH/assignment_candidates", data=np.concatenate(truth_candidates)
            )
            f.create_dataset("TRUTH/assignment_support", data=support)
            f.create_dataset("measured/leptons", data=leptons)
            f.attrs["spin_definition"] = json.dumps(SPIN_DEFINITION)
            f.attrs["sealed_test_used"] = False
        metrics = reconstruction_metrics(n, p, tn, tp)
        metrics["assignment"] = {
            "complete_accuracy": float(
                np.concatenate(complete_correct)[support].mean()
            ),
            "NLL": float(np.concatenate(nll)[support].mean()),
            "per_node_accuracy": dict(
                zip(
                    posterior.assignment_nodes,
                    np.concatenate(per_role_correct)[support].mean(axis=0).tolist(),
                )
            ),
            "unsupported_evaluation_events": int((~support).sum()),
        }
        metrics["joint"] = joint_metric(n, p)
        metrics["joint_energy_score"] = joint_energy_score(n, p, tn, tp)
        metrics["assignment"]["per_role_accuracy"] = {
            name: float(np.concatenate(values)[support].mean())
            for name, values in leaf_accuracy.items()
        }
        spin, exact, coherent, shuffled = spin_summary(p, leptons, tp)
        with h5py.File(posterior_path, "a") as saved:
            for name in exact:
                saved.create_dataset(f"TRUTH/spin/{name}", data=exact[name])
                saved.create_dataset(f"derived/spin/{name}", data=coherent[name])
                saved.create_dataset(f"SHUFFLED_PARENT_PAIR/spin/{name}", data=shuffled[name])
            saved.attrs["spin_shuffle"] = (
                "antitop draws permuted relative to top within event, seed 42014"
            )
        metrics["spin"] = spin
        metrics["prediction"] = {
            "events": len(tn),
            "samples": samples,
            "seconds": time.perf_counter() - begin,
            "EVENT_ID_sha256": __import__("hashlib")
            .sha256(posterior.event_id.numpy().astype("<i8").tobytes())
            .hexdigest(),
            "semantics": {"M": "marginal", "C": "ancestral", "J": "joint"}[variant],
        }
        metrics["parameter_counts"] = {
            "total": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in module.parameters()
                if parameter.requires_grad
            ),
        }
        metrics["population"] = {
            "channel_counts": channel_counts,
            "jet_multiplicity_counts": {
                str(j): int((np.concatenate(multiplicities) == j).sum())
                for j in range(2, 9)
            },
            "truth_mttbar_quantiles_GeV": np.quantile(
                mass(tp.sum(axis=1)), [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
        }
        if variant == "C":
            ladder = {
                endpoint: reconstruction_metrics(
                    n[support],
                    np.concatenate(values)[support],
                    tn[support],
                    tp[support],
                )
                for endpoint, values in oracles.items()
            }
            ladder["deployed"] = reconstruction_metrics(
                n[support], p[support], tn[support], tp[support]
            )
            for endpoint in ladder.values():
                for side in ["top", "antitop"]:
                    endpoint[side] = {
                        key: value for key, value in endpoint[side].items()
                        if key.startswith("parent_")
                    }
            save_json(OUT / "oracle_ladder.json", ladder)
            with h5py.File(posterior_path, "a") as f:
                for endpoint, values in oracles.items():
                    f.create_dataset(
                        f"ORACLE/{endpoint}/parents", data=np.concatenate(values)
                    )
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for axis, observable in zip(
            axes, ["delta_phi_ll", "helicity_product", "cos_opening"]
        ):
            bounds = (0, np.pi) if observable == "delta_phi_ll" else (-1, 1)
            axis.hist(
                exact[observable],
                bins=25,
                range=bounds,
                density=True,
                histtype="step",
                label="truth parents + measured leptons",
            )
            axis.hist(
                coherent[observable].ravel(),
                bins=25,
                range=bounds,
                density=True,
                histtype="step",
                label=f"{variant} paired",
            )
            axis.hist(
                shuffled[observable].ravel(),
                bins=25,
                range=bounds,
                density=True,
                histtype="step",
                label="side-shuffled",
            )
            axis.set_xlabel(observable)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / variant / "spin_diagnostics.png", dpi=140)
        plt.close(fig)
        summary[variant] = metrics
        save_json(summary_path, summary)
        save_json(OUT / "spin_definition.json", SPIN_DEFINITION)
        print(f"{variant} evaluation complete", flush=True)

    paths = [OUT / variant / "posterior_validation.h5" for variant in ["M", "C", "J"]]
    if all(path.is_file() for path in paths):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        common_ids = None
        for variant, path in zip(["M", "C", "J"], paths):
            with h5py.File(path) as source:
                event_ids = source["EVENT_ID"][:]
                if common_ids is None:
                    common_ids = event_ids
                elif not np.array_equal(common_ids, event_ids):
                    raise ValueError("M/C/J posterior populations differ")
                truth = source["TRUTH/parents"][:].sum(axis=1)
                for axis, name, truth_values, bounds in zip(
                    axes,
                    ["m_parent_pair", "pT_parent_pair"],
                    [mass(truth), np.linalg.norm(truth[:, 1:3], axis=-1)],
                    [(250, 1500), (0, 600)],
                ):
                    if variant == "M":
                        axis.hist(
                            truth_values,
                            bins=30,
                            range=bounds,
                            weights=np.full(len(truth_values), 1 / len(truth_values)),
                            histtype="step",
                            label="truth",
                        )
                    axis.hist(
                        source[f"derived/{name}"][:].ravel(),
                        bins=30,
                        range=bounds,
                        weights=np.full(source[f"derived/{name}"].size,
                                        1 / source[f"derived/{name}"].size),
                        histtype="step",
                        label=variant,
                    )
                    axis.set_xlabel(name + " [GeV]")
                    axis.set_ylabel("Probability / bin")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(OUT / "ttbar_distributions.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
