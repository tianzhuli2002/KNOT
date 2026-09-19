#!/usr/bin/env python3
"""Focused contracts that protect study identity and event selection."""

from __future__ import annotations

import json
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import _load_initial_components
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.training.joint import (
    JointEventModule,
    active_architecture,
)


def _assert_close(first, second, name: str, tolerance: float = 2.0e-5) -> None:
    difference = (first - second).abs().max().item()
    if difference > tolerance:
        raise AssertionError(f"{name} changed under a symmetry transformation: {difference}")


def _architecture(overrides: list[str] | None = None) -> str:
    config = load_experiment("configs/experiments/ttbar2L.yaml", overrides or [])
    return json.dumps(active_architecture(config.model), sort_keys=True)


def check_architecture_signature_contract() -> None:
    reference = _architecture()
    inference_only = (
        "inference.seed=7",
        "inference.assignment.top_k=5",
        "inference.invisible.samples_per_assignment=64",
        "inference.invisible.flow.steps=16",
        "inference.invisible.flow.solver=euler",
        "inference.sample_chunk_size=4",
        "inference.invisible.edm.nfe=33",
        "inference.compatibility.weight=2.0",
    )
    for override in inference_only:
        if _architecture([override]) != reference:
            raise AssertionError(f"Inference-only override changed architecture: {override}")

    architecture_changes = (
        "model.encoder.layers=5",
        "model.assignment.refinement.train_iterations=[1,2]",
        "model.assignment.refinement.feedback=soft",
        "model.invisible.generator.type=cfm",
    )
    for override in architecture_changes:
        if _architecture([override]) == reference:
            raise AssertionError(f"Training/architecture override was omitted: {override}")

    c2_base = [
        "model.assignment.refinement.train_iterations=[1,2]",
        "model.assignment.refinement.inference_iterations=2",
        "model.invisible.generator.type=cfm",
        "model.feedback.invisible_to_assignment.enabled=true",
    ]
    c2_reference = _architecture(c2_base)
    if _architecture(
        [*c2_base, "model.feedback.invisible_to_assignment.training_sampler.steps=8"]
    ) != c2_reference:
        raise AssertionError("Equivalent C2 sampler override changed architecture")
    if _architecture(
        [*c2_base, "model.feedback.invisible_to_assignment.training_sampler.steps=9"]
    ) == c2_reference:
        raise AssertionError("C2 training sampler change was omitted from identity")
    if _architecture(
        [
            *c2_base,
            "inference.invisible.flow.steps=17",
            "inference.invisible.flow.solver=euler",
        ]
    ) != c2_reference:
        raise AssertionError("Prediction sampler changed C2 architecture identity")


def check_training_identity_contract() -> None:
    """Training identity must ignore inference settings but include C2 semantics."""
    from scripts.run_study import _training_config_identity

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    changed_inference = copy.deepcopy(base)
    changed_inference.inference.seed = int(base.inference.seed) + 1
    changed_inference.inference.top_k_assignments = 5
    changed_inference.inference.posterior_samples = 64
    changed_inference.inference.joint_flow.sample_chunk_size = 2
    changed_inference.training.runtime.compile = True
    changed_inference.training.output = "results/another-location"
    if _training_config_identity(base) != _training_config_identity(changed_inference):
        raise AssertionError("Inference/runtime settings changed training identity")

    c2_overrides = [
        "model.assignment.refinement.train_iterations=[1,2]",
        "model.assignment.refinement.inference_iterations=2",
        "model.invisible.generator.type=cfm",
        "model.feedback.invisible_to_assignment.enabled=true",
    ]
    c2_base = load_experiment("configs/experiments/ttbar2L.yaml", c2_overrides)
    c2_changed = load_experiment(
        "configs/experiments/ttbar2L.yaml",
        [*c2_overrides, "model.feedback.invisible_to_assignment.training_sampler.steps=9"],
    )
    if _training_config_identity(c2_base) == _training_config_identity(c2_changed):
        raise AssertionError("C2 training sampler change was omitted from training identity")


def check_coordinate_contract() -> None:
    """All maintained coordinate systems must preserve their explicit contract."""
    from scripts.smoke_synthetic import build_synthetic_batch
    from joint_event_reconstruction.models.invisible.coordinates import (
        build_invisible_coordinates,
    )
    from joint_event_reconstruction.topology.compiler import load_topology

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    _, batch, _ = build_synthetic_batch(base, events=2, device="cpu")
    role_names = tuple(
        role.key
        for role in load_topology(
            base.topology_config,
            OmegaConf.to_container(base.data.truth_bindings, resolve=True),
        ).invisible_roles
    )
    # Keep the role ordering from the topology rather than deriving it from
    # tensor dimensions; prediction metadata uses this exact ordering.
    truth = batch["invisible_truth"]
    met = batch["met_xy"]
    for coordinate_type in ("cartesian", "met_residual", "met_exact"):
        config = copy.deepcopy(base.model.invisible.coordinates)
        config.type = coordinate_type
        coordinates = build_invisible_coordinates(config, role_names)
        state = coordinates.encode(truth, met)
        inactive = ~coordinates.active_mask
        if not torch.equal(state[..., inactive], torch.zeros_like(state[..., inactive])):
            raise AssertionError(f"{coordinate_type} emitted non-zero inactive state components")
        decoded = coordinates.decode(state, met)
        if coordinate_type != "met_exact" and not torch.allclose(decoded, truth, atol=2.0e-5):
            raise AssertionError(f"{coordinate_type} coordinate round-trip changed truth")
        if coordinate_type == "met_exact":
            shifted_met = met + torch.tensor([[1.0, -2.0], [0.5, 1.5]])
            projected = coordinates.decode(coordinates.encode(truth, shifted_met), shifted_met)
            if not torch.allclose(projected[..., :, :2].sum(dim=-2), shifted_met, atol=2.0e-5):
                raise AssertionError("MET-exact projection did not close to measured MET")
            diagnostics = coordinates.projection_diagnostics(truth, shifted_met)
            if diagnostics["per_role_transverse_projection_magnitude"].shape != truth.shape[:2]:
                raise AssertionError("MET-exact projection diagnostics lost the role axis")


def check_mass_shell_contract() -> None:
    """Physical p4 conversion is explicit, differentiable and coordinate-agnostic."""
    from joint_event_reconstruction.models.invisible.coordinates import (
        build_invisible_coordinates,
    )
    from joint_event_reconstruction.models.invisible.mass_shell import (
        InvisibleMassShell,
    )

    torch.manual_seed(713)
    spatial = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)
    massless = InvisibleMassShell(mode="fixed", value=0.0)
    massless_p4 = massless.to_four_vector(spatial)
    spatial_norm = spatial.square().sum(dim=-1).sqrt()
    if not torch.allclose(massless_p4[..., 0], spatial_norm, atol=1.0e-12):
        raise AssertionError("Fixed massless energy is not |p|")
    if not torch.allclose(
        massless_p4[..., 0].square() - massless_p4[..., 1:].square().sum(dim=-1),
        torch.zeros_like(spatial_norm),
        atol=1.0e-12,
    ):
        raise AssertionError("Fixed massless invisible vector is not on shell")

    massive = InvisibleMassShell(mode="fixed", value=4.5)
    massive_p4 = massive.to_four_vector(spatial.detach())
    massive_shell = massive_p4[..., 0].square() - massive_p4[..., 1:].square().sum(dim=-1)
    if not torch.allclose(massive_shell, torch.full_like(massive_shell, 4.5**2), atol=1.0e-10):
        raise AssertionError("Fixed massive invisible vector is not on shell")

    generated = InvisibleMassShell(
        mode="generated", value=None, parameterization="log_mass"
    )
    generated_latent = torch.cat(
        (spatial.detach(), torch.randn(3, 2, 1, dtype=torch.float64)), dim=-1
    ).requires_grad_()
    generated_p4 = generated.to_four_vector(generated_latent)
    generated_mass = generated.raw_to_mass(generated_latent[..., 3])
    generated_shell = generated_p4[..., 0].square() - generated_p4[..., 1:].square().sum(dim=-1)
    if (generated_mass <= 0).any() or not torch.allclose(
        generated_shell, generated_mass.square(), atol=1.0e-10
    ):
        raise AssertionError("Generated positive mass is not represented on shell")
    generated_p4.square().sum().backward()
    if generated_latent.grad is None or not torch.isfinite(generated_latent.grad).all():
        raise AssertionError("Generated mass-shell conversion lost finite gradients")
    if generated_p4.device != generated_latent.device or generated_p4.dtype != generated_latent.dtype:
        raise AssertionError("Mass-shell conversion changed device or dtype")

    unconstrained = InvisibleMassShell(
        mode="unconstrained", value=None, parameterization=None, enforcement="unconstrained"
    )
    arbitrary_p4 = torch.tensor(
        [[-2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float64
    )
    if not torch.equal(unconstrained.to_four_vector(arbitrary_p4), arbitrary_p4):
        raise AssertionError("Unconstrained invisible p4 was projected")

    role_names = ("invisible_a", "invisible_b")
    momenta = torch.randn(4, 2, 3, dtype=torch.float64)
    met = torch.randn(4, 2, dtype=torch.float64)
    fixed_config = {
        "enforcement": "exact",
        "mass": {"mode": "fixed", "value": 2.0},
        "roles": {},
    }
    cartesian = build_invisible_coordinates(
        {"type": "cartesian"}, role_names, fixed_config
    )
    physical = cartesian.decode_physical(cartesian.encode(momenta, met), met)
    fixed_shell = physical["four_vector"][..., 0].square() - physical["four_vector"][..., 1:].square().sum(dim=-1)
    if not torch.allclose(fixed_shell, torch.full_like(fixed_shell, 4.0), atol=1.0e-10):
        raise AssertionError("Cartesian coordinates did not preserve a fixed massive shell")

    residual = build_invisible_coordinates(
        {"type": "met_residual"}, role_names, massless
    )
    residual_state = residual.encode(momenta, met)
    if not torch.allclose(residual.decode(residual_state, met), momenta, atol=1.0e-10):
        raise AssertionError("MET-residual coordinates did not compose with mass shells")

    generated_config = {
        "enforcement": "exact",
        "mass": {"mode": "fixed", "value": 0.0},
        "roles": {
            "invisible_b": {
                "mass": {"mode": "generated", "parameterization": "softplus"}
            }
        },
    }
    generated_coordinates = build_invisible_coordinates(
        {"type": "cartesian"}, role_names, generated_config
    )
    physical_masses = torch.tensor(
        [[0.0, 2.0], [0.0, 2.5], [0.0, 3.0], [0.0, 3.5]], dtype=torch.float64
    )
    generated_state = generated_coordinates.encode(
        momenta, met, masses=physical_masses
    )
    generated_output = generated_coordinates.decode_physical(generated_state, met)
    if not torch.allclose(generated_output["mass"], physical_masses, atol=1.0e-10):
        raise AssertionError("Generated mass coordinate did not round-trip its positive mass")
    generated_shell = (
        generated_output["four_vector"][..., 0].square()
        - generated_output["four_vector"][..., 1:].square().sum(dim=-1)
    )
    if not torch.allclose(generated_shell, physical_masses.square(), atol=1.0e-10):
        raise AssertionError("Generated coordinate output is not on its generated shell")


def check_topology_physics_contract() -> None:
    """Protect the compact topology physics grammar and hypothesis evaluator."""
    import h5py
    import inspect

    from joint_event_reconstruction.hypothesis import write_hypothesis_group
    from joint_event_reconstruction.prediction_schema import (
        _iter_hdf5_chunks,
        validate_hypothesis_group,
    )
    from joint_event_reconstruction.topology import (
        HypothesisEvaluator,
        LearnedHypothesisScorer,
        PhysicsFeatureEngine,
        PhysicsFeatureScaler,
        PosteriorReducer,
        TopologyKinematics,
        compile_physics,
        load_topology,
        reuse_unconditioned_posterior,
    )
    from joint_event_reconstruction.topology.compiler import compile_topology
    from joint_event_reconstruction.physics import compose_assignment_state
    from joint_event_reconstruction.prediction import predict_hdf5

    split_parameter = inspect.signature(predict_hdf5).parameters["split"]
    if split_parameter.default is not inspect.Parameter.empty:
        raise AssertionError("Factorized prediction split must be explicit")
    if split_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise AssertionError("Factorized prediction split must be keyword-only")

    current = load_topology("configs/topology/ttbar2L.yaml")
    topology_only = compile_topology(
        {"topology": OmegaConf.to_container(
            OmegaConf.load("configs/topology/ttbar2L.yaml").topology,
            resolve=True,
        )}
    )
    if current.signature != topology_only.signature:
        raise AssertionError("Adding optional physics changed the historical topology signature")
    if current.physics is None or current.physics_signature is None:
        raise AssertionError("ttbar2L physics block did not compile")
    if current.physics_signature != load_topology("configs/topology/ttbar2L.yaml").physics_signature:
        raise AssertionError("Physics signature is not deterministic")
    if load_topology("configs/topology/ttbar1L.yaml").physics is not None:
        raise AssertionError("Topology-only files unexpectedly acquired physics semantics")

    raw_physics = OmegaConf.to_container(
        OmegaConf.load("configs/topology/ttbar2L.yaml").physics,
        resolve=True,
    )
    compiled = compile_physics(
        raw_physics,
        node_names=current.nodes.keys(),
        leaf_names=current.leaf_roles.keys(),
        topology_composite_names=current.composite_roles.keys(),
    )
    if compiled.signature != current.physics_signature:
        raise AssertionError("Physics signature changed during direct compilation")

    invalid_cases = []
    unknown = copy.deepcopy(raw_physics)
    unknown["observables"]["bad"] = {
        "type": "invariant_mass", "object": "not_an_object", "reference": 1.0
    }
    invalid_cases.append((unknown, "unknown object"))
    cycle = {"composites": {"a": {"sum": ["b"]}, "b": {"sum": ["a"]}}}
    invalid_cases.append((cycle, "cycle"))
    bad_kind = copy.deepcopy(raw_physics)
    bad_kind["observables"]["bad"] = {
        "type": "not_supported", "object": "top_e", "reference": 1.0
    }
    invalid_cases.append((bad_kind, "observable type"))
    bad_arity = copy.deepcopy(raw_physics)
    bad_arity["observables"]["bad"] = {
        "type": "mass_difference", "objects": ["top_e"], "reference": 0.0
    }
    invalid_cases.append((bad_arity, "observable arity"))
    bad_met = copy.deepcopy(raw_physics)
    bad_met["observables"]["bad"] = {
        "type": "transverse_sum_residual", "objects": ["top_e.nu"], "measurement": "HT"
    }
    invalid_cases.append((bad_met, "MET measurement"))
    for raw, label in invalid_cases:
        try:
            compile_physics(
                raw,
                node_names=current.nodes.keys(),
                leaf_names=current.leaf_roles.keys(),
                topology_composite_names=current.composite_roles.keys(),
            )
        except ValueError:
            continue
        raise AssertionError(f"Physics compiler accepted invalid {label}")

    pools = {}
    for node_name in current.dag_order:
        pool = {
            "candidate_mask": torch.ones(1, 2, dtype=torch.bool),
            "leaf_indices": {},
            "child_indices": {},
        }
        for role in current.nodes[node_name].leaf_roles:
            if not role.invisible:
                pool["leaf_indices"][role.key] = torch.tensor(
                    [[0] * role.multiplicity, [1] * role.multiplicity],
                    dtype=torch.long,
                )
        pools[node_name] = pool
    vectors = torch.tensor(
        [[[100.0, 30.0, 0.0, 90.0], [100.0, -30.0, 0.0, -90.0],
          [80.0, 10.0, 2.0, 20.0], [70.0, -10.0, -2.0, -20.0]]]
    )
    assignments = torch.tensor(
        [[[0, 0], [1, 1]]], dtype=torch.long
    ).repeat(1, 1, 1)
    posterior = torch.tensor(
        [
            [
                [[30.0, 2.0, 3.0], [20.0, -2.0, -3.0]],
                [[31.0, 2.0, 3.0], [19.0, -2.0, -3.0]],
                [[29.0, 2.0, 3.0], [21.0, -2.0, -3.0]],
            ]
        ]
    )
    kinematics = TopologyKinematics(current)
    objects = kinematics.compose(
        pools,
        vectors,
        assignments,
        invisible_momenta=posterior,
    )
    expected_shapes = {name: (1, 2, 3, 4) for name in objects}
    if {name: tuple(value.shape) for name, value in objects.items()} != expected_shapes:
        raise AssertionError("Topology kinematics did not preserve [B,K,M,4] shapes")
    for hypothesis in range(2):
        for sample in range(posterior.shape[1]):
            reference = compose_assignment_state(
                current,
                pools,
                vectors[0],
                assignments[0, hypothesis],
                {
                    "top_e.nu": posterior[0, sample, 0],
                    "top_mu.nu": posterior[0, sample, 1],
                },
            )
            for name in current.roots:
                if not torch.allclose(objects[name][0, hypothesis, sample], reference[name]):
                    raise AssertionError(f"Vectorized kinematics disagrees for {name}")

    feature_result = PhysicsFeatureEngine(current.physics).evaluate(
        objects, observed_met=torch.zeros(1, 2)
    )
    if feature_result.tensor.shape != (1, 2, 3, 7):
        raise AssertionError("ttbar2L physics feature shape is not [B,K,M,F]")
    if feature_result.feature_names != (
        "W_e_mass", "W_mu_mass", "met_closure.px", "met_closure.py",
        "top_e_mass", "top_mass_balance", "top_mu_mass",
    ):
        raise AssertionError("Physics feature order is not deterministic")
    scaler = PhysicsFeatureScaler.fit(
        feature_result.tensor,
        feature_result.feature_names,
        topology_signature=current.signature,
        physics_signature=current.physics_signature,
        source_data_identity="synthetic-contract",
    )
    restored_scaler = PhysicsFeatureScaler.from_dict(scaler.as_dict())
    scaled = restored_scaler.transform(
        feature_result.tensor,
        feature_names=feature_result.feature_names,
        topology_signature=current.signature,
        physics_signature=current.physics_signature,
    )
    try:
        restored_scaler.transform(scaled, topology_signature="wrong")
    except ValueError:
        pass
    else:
        raise AssertionError("Physics scaler accepted a topology-signature mismatch")

    reducer = PosteriorReducer()
    reduction = reducer.reduce(scaled)
    permuted = reducer.reduce(scaled[:, :, [2, 0, 1]])
    for first, second in zip(
        (reduction.mean, reduction.std, reduction.rms, reduction.analytic_log_evidence),
        (permuted.mean, permuted.std, permuted.rms, permuted.analytic_log_evidence),
    ):
        if not torch.allclose(first, second, atol=1.0e-4, rtol=1.0e-5):
            raise AssertionError("Posterior reduction depends on sample ordering")
    good = reducer.reduce(torch.zeros(1, 2, 2, 1)).analytic_log_evidence
    bad = reducer.reduce(torch.full((1, 2, 2, 1), 10.0)).analytic_log_evidence
    if not (good > bad).all():
        raise AssertionError("Analytic physics scorer ranks a good hypothesis below a poor one")

    scorer = LearnedHypothesisScorer(reduction.as_tensor().shape[-1], hidden=8)
    learned = scorer(reduction.as_tensor())
    loss = learned.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in scorer.parameters()]
    if not torch.isfinite(learned).all() or not all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise AssertionError("Learned hypothesis scorer forward/backward is not finite")
    with tempfile.TemporaryDirectory(prefix="jer-hypothesis-contract-") as directory:
        checkpoint = Path(directory) / "scorer.pt"
        torch.save(scorer.state_dict(), checkpoint)
        restored = LearnedHypothesisScorer(reduction.as_tensor().shape[-1], hidden=8)
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if not torch.allclose(scorer(reduction.as_tensor()), restored(reduction.as_tensor())):
            raise AssertionError("Reloaded learned scorer is not deterministic")

        from joint_event_reconstruction.topology import MatchedVisibleOnlyScorer

        visible_control = MatchedVisibleOnlyScorer(hidden=8)
        visible_input = torch.zeros(1, 2, len(visible_control.input_names))
        visible_output = visible_control(visible_input)
        if visible_output.shape != (1, 2) or not torch.isfinite(visible_output).all():
            raise AssertionError("Matched visible-only scorer has an invalid interface")

        evaluator = HypothesisEvaluator(
            current,
            scaler=restored_scaler,
            mode="learned",
            scorer=restored,
        )
        evaluated = evaluator.evaluate(
            torch.zeros(1, 2),
            pools,
            vectors,
            assignments,
            invisible_momenta=posterior,
            observed_met=torch.zeros(1, 2),
        )
        if not torch.isfinite(evaluated.final_probability).all() or not torch.allclose(
            evaluated.final_probability.sum(dim=1), torch.ones(1)
        ):
            raise AssertionError("Hypothesis evaluator probabilities are invalid")
        padded = assignments.clone()
        padded[0, 1] = -1
        padded_evaluation = evaluator.evaluate(
            torch.tensor([[0.0, -torch.inf]]),
            pools,
            vectors,
            padded,
            invisible_momenta=posterior,
            observed_met=torch.zeros(1, 2),
            valid_hypotheses=torch.tensor([[True, False]]),
        )
        if not torch.allclose(
            padded_evaluation.final_probability, torch.tensor([[1.0, 0.0]])
        ):
            raise AssertionError("Padded assignment hypotheses were not masked safely")
        alpha_zero = HypothesisEvaluator(
            current, scaler=restored_scaler, mode="physics", alpha=0.0
        ).evaluate(
            torch.tensor([[0.0, -torch.inf]]),
            pools,
            vectors,
            padded,
            invisible_momenta=posterior,
            observed_met=torch.zeros(1, 2),
            valid_hypotheses=torch.tensor([[True, False]]),
        )
        if not torch.isfinite(alpha_zero.final_probability).all():
            raise AssertionError("alpha=0 physics evaluation produced non-finite probabilities")
        if not torch.allclose(alpha_zero.final_probability, torch.tensor([[1.0, 0.0]])):
            raise AssertionError("alpha=0 physics evaluation did not preserve visible ranking")
        if alpha_zero.learned_delta[0, 1] != 0 or not torch.isneginf(
            alpha_zero.final_log_score[0, 1]
        ):
            raise AssertionError("alpha=0 padded compatibility correction was not masked")

        all_valid_path = Path(directory) / "hypotheses-all-valid.h5"
        with h5py.File(all_valid_path, "w") as handle:
            write_hypothesis_group(
                handle,
                states=assignments,
                truth=torch.tensor([[True, False]]),
                evaluation=evaluated,
                scaler_identity=restored_scaler.identity_sha256,
                scorer_type="learned",
            )
        with h5py.File(all_valid_path, "r") as handle:
            validate_hypothesis_group(handle, count=1, top_k=2)

        padded_path = Path(directory) / "hypotheses-padded.h5"
        with h5py.File(padded_path, "w") as handle:
            write_hypothesis_group(
                handle,
                states=padded,
                truth=torch.tensor([[True, False]]),
                evaluation=padded_evaluation,
                scaler_identity=restored_scaler.identity_sha256,
                scorer_type="learned",
            )
        with h5py.File(padded_path, "r") as handle:
            validate_hypothesis_group(handle, count=1, top_k=2)

        chunk_path = Path(directory) / "chunked.h5"
        with h5py.File(chunk_path, "w") as handle:
            dataset = handle.create_dataset("large", data=np.arange(40, dtype=np.float32).reshape(10, 4))
            chunks = list(_iter_hdf5_chunks(dataset, target_bytes=32))
        if len(chunks) != 5 or any(chunk.shape[0] != 2 for chunk in chunks):
            raise AssertionError("HDF5 deep-validation chunking is not bounded by target bytes")

    reused, branches = reuse_unconditioned_posterior(posterior, 5)
    if branches != 1 or reused.shape != (1, 5, 3, 2, 3):
        raise AssertionError("Unconditioned posterior was not broadcast across hypotheses")


def check_conditional_lgatr_contract() -> None:
    """CFM keeps its objective while swapping in the conditional L-GATr backbone."""
    from joint_event_reconstruction.data.invisible import InvisibleScaler
    from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
    from joint_event_reconstruction.topology.compiler import load_topology

    base_overrides = ["model.invisible.generator.type=cfm"]
    lgatr_overrides = [
        *base_overrides,
        "model.invisible.generator.backbone.type=lgatr",
        "model.invisible.generator.backbone.layers=1",
        "model.invisible.generator.backbone.n_heads=1",
        "model.invisible.generator.backbone.hidden.multivector_channels=2",
        "model.invisible.generator.backbone.hidden.scalar_channels=8",
    ]
    base = load_experiment("configs/experiments/ttbar2L.yaml", base_overrides)
    spec = load_topology(
        base.topology_config,
        OmegaConf.to_container(base.data.truth_bindings, resolve=True),
    )
    role_names = tuple(role.key for role in spec.invisible_roles)
    batch_size = 2
    width = int(base.model.latent.d_model)
    target = torch.randn(batch_size, len(role_names), 3)
    met = torch.randn(batch_size, 2)
    scaler = InvisibleScaler.fit(target, role_names)
    visible = {
        "pooled": torch.randn(batch_size, width),
        "tokens": torch.randn(batch_size, 4, width),
        "padding_mask": torch.tensor(
            [[False, False, False, True], [False, False, True, True]]
        ),
        "physics_vectors": torch.randn(batch_size, 4, 4) * 20.0,
        "met_xy": met,
    }
    parent = torch.randn(batch_size, len(role_names), width)
    target_scaled = scaler.transform(target)

    for overrides in (base_overrides, lgatr_overrides):
        config = load_experiment("configs/experiments/ttbar2L.yaml", overrides)
        generator = InvisibleGenerator(
            spec,
            width,
            config.model.invisible,
            config.inference.invisible,
            scaler=scaler,
        )
        result = generator.loss(target_scaled, visible, parent)
        if not torch.isfinite(result["loss"]):
            raise AssertionError("CFM invisible generator loss is non-finite")
        result["loss"].backward()
        gradients = [parameter.grad for parameter in generator.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError("CFM invisible generator backward is non-finite")

        with tempfile.TemporaryDirectory(prefix="jer-invisible-backbone-") as directory:
            checkpoint = Path(directory) / "generator.pt"
            torch.save(generator.state_dict(), checkpoint)
            restored = InvisibleGenerator(
                spec,
                width,
                config.model.invisible,
                config.inference.invisible,
                scaler=scaler,
            )
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            generator.eval()
            restored.eval()
            first_seed = torch.Generator().manual_seed(981)
            second_seed = torch.Generator().manual_seed(981)
            first, first_nfe = generator.sample(
                visible, parent, samples=2, budget=2, generator=first_seed
            )
            second, second_nfe = restored.sample(
                visible, parent, samples=2, budget=2, generator=second_seed
            )
            if first_nfe != 4 or second_nfe != 4 or not torch.isfinite(first).all():
                raise AssertionError("CFM generator prediction smoke failed")
            if not torch.allclose(first, second):
                raise AssertionError("Reloaded CFM generator changed deterministic prediction")
            physical = restored.decode_physical_state(first, met, scaler)["four_vector"]
            if physical.shape != (batch_size, 2, len(role_names), 4) or not torch.isfinite(physical).all():
                raise AssertionError("CFM generator did not expose finite physical p4 output")

    massive_config = load_experiment(
        "configs/experiments/ttbar2L.yaml",
        [*base_overrides, "model.invisible.mass_shell.mass.value=3.0"],
    )
    massive_coordinates = InvisibleGenerator(
        spec,
        width,
        massive_config.model.invisible,
        massive_config.inference.invisible,
        scaler=scaler,
    ).coordinates
    massive_state = massive_coordinates.encode(target, met)
    massive_p4 = massive_coordinates.decode_physical(massive_state, met)["four_vector"]
    massive_shell = massive_p4[..., 0].square() - massive_p4[..., 1:].square().sum(dim=-1)
    if not torch.allclose(massive_shell, torch.full_like(massive_shell, 9.0), atol=1.0e-5):
        raise AssertionError("Non-zero fixed-mass CFM smoke is not on shell")


def check_prediction_writer_shape_contract() -> None:
    """HDF5 prediction shapes must not inherit torch's extendible-dimension type."""
    import h5py

    from joint_event_reconstruction.prediction import create_prediction_dataset

    with tempfile.TemporaryDirectory(prefix="jer-prediction-writer-") as directory:
        path = Path(directory) / "prediction.h5"
        with h5py.File(path, "w") as handle:
            dataset = create_prediction_dataset(
                handle,
                "physical_samples",
                torch.Size((2, 2, 4)),
                "<f4",
            )
            if dataset.shape != (0, 2, 2, 4):
                raise AssertionError(f"Unexpected prediction dataset shape: {dataset.shape}")
            if dataset.maxshape != (None, 2, 2, 4):
                raise AssertionError(
                    f"Unexpected prediction dataset maxshape: {dataset.maxshape}"
                )
            dataset.resize(1, axis=0)
            dataset[0] = np.zeros((2, 2, 4), dtype=np.float32)
            if not np.isfinite(dataset[:]).all():
                raise AssertionError("Prediction writer produced non-finite smoke output")


def check_physical_posterior_shape_contract() -> None:
    """The physical posterior contract covers every rank and axis explicitly."""
    from joint_event_reconstruction.prediction_schema import (
        validate_physical_posterior_shape,
        validate_prediction,
    )

    valid = np.zeros((16, 10, 8, 2, 4), dtype=np.float32)
    validate_physical_posterior_shape(
        valid,
        event_count=16,
        top_k=10,
        posterior_samples=8,
        n_invisible_particles=2,
    )
    malformed = (
        (16, 10, 8, 2),
        (16, 9, 8, 2, 4),
        (16, 10, 7, 2, 4),
        (16, 10, 8, 3, 4),
        (16, 10, 8, 2, 3),
    )
    for shape in malformed:
        try:
            validate_physical_posterior_shape(
                np.zeros(shape, dtype=np.float32),
                event_count=16,
                top_k=10,
                posterior_samples=8,
                n_invisible_particles=2,
            )
        except ValueError:
            continue
        raise AssertionError(f"Malformed physical posterior shape was accepted: {shape}")

    existing = (
        ROOT
        / "results/validation/gpu-smoke/c0093bff402cbe7ac83919b866f8a6c26b6f5cb3"
        / "attempt-001/prediction.h5"
    )
    if not existing.is_file():
        raise AssertionError(f"Known failed-smoke prediction artifact is missing: {existing}")
    validate_prediction(existing)
    import h5py

    with h5py.File(existing, "r") as handle:
        physical = handle["INVISIBLE/four_vectors"][:]
        validate_physical_posterior_shape(
            physical,
            event_count=len(handle["EVENT_ID"]),
            top_k=int(handle.attrs["K"]),
            posterior_samples=int(handle.attrs["M"]),
            n_invisible_particles=int(handle.attrs["invisible_role_count"]),
            label="known failed-smoke INVISIBLE/four_vectors",
        )
        if str(handle.attrs["physics_vector_representation"]) != "E_px_py_pz_GeV_v1":
            raise AssertionError("Known failed-smoke artifact does not contain physical E_px_py_pz")
        mass_squared = physical[..., 0] ** 2 - (physical[..., 1:] ** 2).sum(axis=-1)
        if not np.allclose(mass_squared, 0.0, atol=5.0e-2, rtol=2.0e-4):
            raise AssertionError("Known failed-smoke physical posterior is not massless")


def check_flow_coupling_context_contract() -> None:
    """Conditional OT must use role contexts before coordinate-token expansion."""
    import torch

    from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
    from joint_event_reconstruction.topology.compiler import load_topology

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    base = copy.deepcopy(base)
    base.model.invisible.enabled = True
    base.model.invisible.formulation = "flow_matching"
    base.model.invisible.flow_matching.coupling = {
        "method": "conditional_ot",
        "condition_weight": 1.0,
    }
    spec = load_topology(
        base.topology_config,
        OmegaConf.to_container(base.data.truth_bindings, resolve=True),
    )
    role_count = len(spec.invisible_roles)
    width = int(base.model.latent.d_model)
    pooled = torch.full((2, width), 3.0)
    role_context = torch.arange(
        1, 1 + 2 * role_count * width, dtype=torch.float32
    ).reshape(2, role_count, width)
    visible = {"pooled": pooled}

    for coordinate_type in ("met_residual", "met_exact"):
        config = copy.deepcopy(base)
        config.model.invisible.coordinates.type = coordinate_type
        generator = InvisibleGenerator(
            spec,
            width,
            config.model.invisible,
            config.inference.invisible,
        )
        coordinate_context = generator.coordinates.parent_context(
            role_context, pooled
        )
        condition = generator.coupling_condition(
            visible,
            coordinate_context,
            role_context=role_context,
        )
        expected = torch.cat((pooled, role_context.mean(dim=1)), dim=1)
        if not torch.allclose(condition, expected):
            raise AssertionError(
                f"{coordinate_type} conditional-OT context did not use role mean"
            )
        expanded_mean = coordinate_context.mean(dim=1)
        if coordinate_type == "met_residual" and torch.allclose(
            condition[:, width:], expanded_mean
        ):
            raise AssertionError(
                f"{coordinate_type} conditional-OT context still uses expanded tokens"
            )
        if generator.coordinates.token_count != role_count:
            try:
                generator.coupling_condition(visible, coordinate_context)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{coordinate_type} accepted expanded contexts without role contexts"
                )


def check_single_root_runtime_contract() -> None:
    """Snapshots are idempotent and source/data paths remain distinct."""
    from joint_event_reconstruction.runtime_paths import (
        RuntimePaths,
        ensure_source_snapshot,
        verify_source_snapshot,
    )

    with tempfile.TemporaryDirectory(prefix="jer-runtime-contract-") as directory:
        root = Path(directory)
        repository = root / "repo"
        for relative in (
            "src/joint_event_reconstruction",
            "scripts",
            "configs/site",
            "configs/experiments",
        ):
            (repository / relative).mkdir(parents=True, exist_ok=True)
        (repository / "src/joint_event_reconstruction/__init__.py").write_text("")
        (repository / "src/joint_event_reconstruction/runtime_paths.py").write_text(
            (ROOT / "src/joint_event_reconstruction/runtime_paths.py").read_text()
        )
        (repository / "scripts/run_study.py").write_text("raise SystemExit(0)\n")
        (repository / "configs/site/csf3.yaml").write_text(
            (ROOT / "configs/site/csf3.yaml").read_text()
        )
        (repository / "configs/experiments/ttbar2L.yaml").write_text(
            "experiment: ttbar2L\n"
        )
        (repository / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "jer-contract@example.invalid"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "JER contract"],
            cwd=repository,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repository, check=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        snapshot = ensure_source_snapshot(commit, project_root=repository)
        if ensure_source_snapshot(commit, project_root=repository) != snapshot:
            raise AssertionError("Source snapshot creation is not idempotent")
        verify_source_snapshot(commit, project_root=repository, source_root=snapshot)
        metadata = json.loads((snapshot / "SOURCE_SNAPSHOT.json").read_text())
        if metadata.get("commit_sha") != commit or (snapshot / ".git").exists():
            raise AssertionError("Source snapshot metadata or immutability contract failed")
        paths = RuntimePaths(
            project_root=repository,
            source_root=snapshot,
            data_root=repository / "data",
            cache_root=repository / "cache",
            results_root=repository / "results",
            logs_root=repository / "logs",
            runtime_root=repository / "runtime",
            source_sha=commit,
        )
        if paths.project("data/input.h5") != repository / "data/input.h5":
            raise AssertionError("Project data path did not resolve from project_root")
        if paths.source("configs/experiments/ttbar2L.yaml") != snapshot / "configs/experiments/ttbar2L.yaml":
            raise AssertionError("Source config path did not resolve from source_root")


def check_slurm_snapshot_contract() -> None:
    """Workers use an immutable source snapshot while project checkout changes."""
    import scripts.submit_study_slurm as scheduler
    from joint_event_reconstruction.runtime_paths import ensure_source_snapshot

    with tempfile.TemporaryDirectory(prefix="jer-slurm-contract-") as directory:
        root = Path(directory)
        repository = root / "repo"
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "jer-contract@example.invalid"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "JER contract"],
            cwd=repository,
            check=True,
        )
        fake_runner = repository / "scripts" / "run_study.py"
        fake_runner.parent.mkdir(parents=True)
        fake_runner.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
        experiment = repository / "configs" / "experiments" / "ttbar2L.yaml"
        experiment.parent.mkdir(parents=True)
        experiment.write_text("experiment: ttbar2L\n")
        package = repository / "src" / "joint_event_reconstruction"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (package / "runtime_paths.py").write_text(
            (ROOT / "src/joint_event_reconstruction/runtime_paths.py").read_text()
        )
        (repository / "configs" / "site").mkdir(parents=True)
        (repository / "configs/site/csf3.yaml").write_text(
            (ROOT / "configs/site/csf3.yaml").read_text()
        )
        (repository / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repository, check=True)
        initial_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        source_root = ensure_source_snapshot(initial_commit, project_root=repository)

        manifest = root / "study_manifest.yaml"
        manifest.write_text("study: contract\n")
        task_file = root / "tasks.tsv"
        task_file.write_text("ttbar2L\tvariant\t42\n")
        expected_manifest = scheduler._sha256_file(manifest)
        source_experiment = source_root / "configs/experiments/ttbar2L.yaml"
        expected_experiment = scheduler._sha256_file(source_experiment)
        worker = root / "worker.sh"
        original_root = scheduler.ROOT
        scheduler.ROOT = repository
        try:
            scheduler._worker_script(
                worker,
                task_file=task_file,
                study_path=manifest,
                stage="train",
                output_root=root / "results",
                device="cpu",
                global_overrides=(),
                expected_commit=initial_commit,
                source_root=source_root,
                project_root=repository,
                expected_manifest_sha256=expected_manifest,
                expected_experiments={
                    "ttbar2L": {
                        "path": str(source_experiment),
                        "sha256": expected_experiment,
                    }
                },
            )
        finally:
            scheduler.ROOT = original_root
        if f"export JER_SOURCE_ROOT={source_root}" not in worker.read_text():
            raise AssertionError("Slurm worker does not pin the submitted source snapshot")

        environment = dict(os.environ, SLURM_ARRAY_TASK_ID="0")
        accepted = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if accepted.returncode != 0:
            raise AssertionError(f"Unchanged source snapshot was rejected: {accepted.stderr}")

        # A changed normal checkout must not affect a queued worker.
        (repository / "configs/experiments/ttbar2L.yaml").write_text(
            "experiment: changed-in-checkout\n"
        )
        accepted_checkout = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if accepted_checkout.returncode != 0:
            raise AssertionError("Worker still depends on the mutable checkout")

        manifest.write_text("study: changed\n")
        changed_manifest = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if changed_manifest.returncode != 3 or "manifest snapshot changed" not in changed_manifest.stderr:
            raise AssertionError("Changed Slurm manifest was not rejected before work")

        manifest.write_text("study: contract\n")
        source_experiment.write_text("experiment: changed-in-snapshot\n")
        changed_source = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if changed_source.returncode != 3 or "Base experiment changed" not in changed_source.stderr:
            raise AssertionError("Changed source experiment was not rejected before work")


def check_postprocess_recovery_contract() -> None:
    """Prediction-level postprocessing skips complete products deterministically."""
    import scripts.run_study as runner
    import scripts.submit_study_slurm as scheduler
    from joint_event_reconstruction.studies import load_study

    study = load_study("configs/studies/08_generator_inference.yaml")
    variant = study.variant_map["edm"]
    with tempfile.TemporaryDirectory(prefix="jer-postprocess-contract-") as directory:
        root = Path(directory)
        source = root / "source.h5"
        source.write_bytes(b"validated by the runner contract")
        for prediction in variant.predictions:
            target = (
                root / study.name / "ttbar2L" / variant.name / "seed_42"
                / "predictions" / f"{prediction.name}.h5"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)

        complete_evaluation = (
            root / study.name / "ttbar2L" / variant.name / "seed_42"
            / "evaluation" / "nfe5.json"
        )
        complete_evaluation.parent.mkdir(parents=True, exist_ok=True)
        complete_evaluation.write_text(
            json.dumps({"prediction_file": "nfe5.h5", "events": 4}) + "\n"
        )
        complete_plot = (
            root / study.name / "ttbar2L" / variant.name / "seed_42"
            / "plots" / "nfe5" / "summary.png"
        )
        complete_plot.parent.mkdir(parents=True, exist_ok=True)
        complete_plot.write_bytes(b"plot")

        original_validator = runner._validate_prediction_artifact
        runner._validate_prediction_artifact = lambda path: None
        try:
            first = scheduler._postprocess_task_lists(
                study, ("ttbar2L",), (variant,), (42,), root
            )
            second = scheduler._postprocess_task_lists(
                study, ("ttbar2L",), (variant,), (42,), root
            )
        finally:
            runner._validate_prediction_artifact = original_validator
        evaluation_tasks, plot_tasks, skipped_evaluation, skipped_plot = first
        if first != second:
            raise AssertionError("Postprocessing recovery task selection is not idempotent")
        if len(evaluation_tasks) != len(variant.predictions) - 1:
            raise AssertionError("Recovery did not select exactly the missing evaluations")
        if plot_tasks or len(skipped_plot) != 1:
            raise AssertionError("Recovery did not distinguish complete plot products")
        if len(skipped_evaluation) != 1:
            raise AssertionError("Recovery did not skip the complete evaluation product")


def check_campaign_gate_contract() -> None:
    """Pending gates block while the human-selected visible gate resolves."""
    from joint_event_reconstruction.studies import (
        ScientificGatePending,
        load_scientific_gates,
        load_study,
        resolve_study,
    )

    gates = load_scientific_gates()
    expected_gates = {
        "assignment_reasoning",
        "visible_assignment",
        "invisible_generator",
        "invisible_coordinates",
        "invisible_backbone",
        "assignment_coupling",
        "joint_model",
        "assignment_formulation",
        "assignment_conditioning",
        "invisible_feedback",
    }
    if set(gates) != expected_gates:
        raise AssertionError(f"Unexpected scientific gate set: {sorted(gates)}")
    assignment = gates["assignment_reasoning"]
    if (
        assignment["status"] != "locked"
        or assignment["selected_variant"] != "m1"
        or assignment["validated_sha"] != "b0fc3583c94640fa7ec8b281935204c8bda64f16"
    ):
        raise AssertionError("Study-01 assignment_reasoning gate is not locked to m1")

    visible = gates["visible_assignment"]
    if (
        visible["status"] != "locked"
        or visible["selected_variant"] != "lgatr"
        or visible["validated_sha"] != "e00d3ea351e6850e31da610bcb5a79eb1cfed273"
    ):
        raise AssertionError("Study-12 visible_assignment gate is not locked to lgatr")
    expected_visible_patch = {
        "model": {
            "mode": "factorized",
            "encoder": {
                "type": "lgatr",
                "layers": 4,
                "n_heads": 4,
                "hidden": {"multivector_channels": 8, "scalar_channels": 64},
                "symmetry_breaking": {"beam": "xyplane", "time": True},
            },
            "assignment": {"formulation": "discriminative"},
        }
    }
    if visible["selected_patch"] != expected_visible_patch:
        raise AssertionError("Study-12 visible_assignment selected patch changed")
    if any(
        token in json.dumps(visible["selected_patch"])
        for token in ("train_iterations", "inference_iterations", "feedback", "trainable")
    ):
        raise AssertionError("Visible-assignment gate contains training or recurrent settings")

    formulation = gates["assignment_formulation"]
    if (
        formulation["status"] != "locked"
        or formulation["selected_variant"] != "m1"
        or formulation["validated_sha"]
        != "e0ee7b3114ce413b0bb9ea7f96e786e13c4da4f1"
    ):
        raise AssertionError(
            "Study-05 assignment_formulation gate is not locked to m1"
        )
    if formulation["selected_patch"] != {
        "model": {"assignment": {"formulation": "discriminative"}}
    }:
        raise AssertionError("Study-05 assignment gate selected an unexpected patch")
    if set(formulation["artifacts"]["ttbar2L"]) != {
        "seed_42", "seed_43", "seed_44"
    }:
        raise AssertionError("Study-05 assignment gate is missing a model seed artifact")
    for seed in (42, 43, 44):
        artifact = formulation["artifacts"]["ttbar2L"][f"seed_{seed}"]
        if (
            artifact["source_study"] != "discrete_assignment"
            or artifact["variant"] != "m1"
            or int(artifact["model_seed"]) != seed
            or artifact["git_sha"]
            != "752c52df8887dee9f8f04ab16ed0fe1175b8012f"
        ):
            raise AssertionError(f"Study-05 assignment gate has wrong seed-{seed} provenance")

    generator = gates["invisible_generator"]
    expected_generator_patch = {
        "model": {
            "mode": "factorized",
            "invisible": {
                "conditioning": {"assignment": "none"},
                "generator": {"type": "cfm"},
            },
        }
    }
    if (
        generator["status"] != "locked"
        or generator["selected_variant"] != "cfm"
        or generator["validated_sha"]
        != "93d57cc8c54c2f8f3e4ee49c2ee4c5024c4cee16"
        or generator["selected_patch"] != expected_generator_patch
    ):
        raise AssertionError("Study-02 invisible_generator gate is not locked to CFM")
    if set(generator["artifacts"]["ttbar2L"]) != {
        "seed_42", "seed_43", "seed_44"
    }:
        raise AssertionError("Study-02 CFM gate is missing a model seed artifact")
    for seed in (42, 43, 44):
        artifact = generator["artifacts"]["ttbar2L"][f"seed_{seed}"]
        if (
            artifact["source_study"] != "generator_formulations"
            or artifact["variant"] != "cfm"
            or int(artifact["model_seed"]) != seed
            or artifact["git_sha"]
            != "752c52df8887dee9f8f04ab16ed0fe1175b8012f"
            or not artifact["checkpoint"].endswith(
                f"/generator_formulations/ttbar2L/cfm/seed_{seed}/train/checkpoints/best-loss.ckpt"
            )
        ):
            raise AssertionError(f"Study-02 CFM gate has wrong seed-{seed} provenance")

    with tempfile.TemporaryDirectory(prefix="jer-pending-gate-") as directory:
        pending_path = Path(directory) / "gates.yaml"
        pending = copy.deepcopy(gates)
        pending["visible_assignment"].update(
            {
                "status": "pending",
                "selected_variant": None,
                "validated_sha": None,
                "selected_patch": {},
                "artifacts": {},
            }
        )
        pending_path.write_text(OmegaConf.to_yaml(OmegaConf.create(pending)))
        try:
            resolve_study(
                "configs/studies/02_generator_formulations.yaml", pending_path
            )
        except ScientificGatePending as error:
            if "visible_assignment" not in str(error):
                raise AssertionError(
                    f"Pending visible gate raised the wrong barrier: {error}"
                ) from error
        else:
            raise AssertionError("A pending visible_assignment gate resolved Study 02")

    study12 = resolve_study("configs/studies/12_encoder_backbone.yaml")
    if study12.seeds != (42, 43, 44):
        raise AssertionError("Study 12 must use model seeds 42, 43 and 44")
    for variant in study12.variants:
        assignment_patch = variant.patch.get("model", {}).get("assignment", {})
        if "refinement" in assignment_patch:
            raise AssertionError(f"Study 12 inherited visible recurrence: {variant.name}")
        if any(prediction.recurrent_steps is not None for prediction in variant.predictions):
            raise AssertionError(f"Study 12 has recurrent prediction steps: {variant.name}")
    study_paths = {
        load_study(path).name: path
        for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml"))
    }
    study02 = resolve_study(study_paths["generator_formulations"])
    if {variant.name for variant in study02.variants} != {
        "edm",
        "cfm",
        "ot_cfm",
        "conditional_ot_cfm",
    }:
        raise AssertionError("Study 02 does not contain exactly the streamlined objectives")
    from scripts.run_study import _gate_checkpoint, _public_run_config

    for variant in study02.variants:
        if "ddim" in variant.name.lower() or "ddim" in json.dumps(variant.patch).lower():
            raise AssertionError(f"Study 02 retained DDIM: {variant.name}")
        if variant.patch.get("model", {}).get("mode") != "factorized":
            raise AssertionError(f"Study 02:{variant.name} is not factorized")
        for seed in study02.seeds:
            _, _, resolved = _public_run_config(
                study02,
                "ttbar2L",
                variant,
                seed,
                ROOT / "results/engineering/invariant-study02",
                (),
            )
            model = resolved.model
            if str(model.visible.type) != "lgatr":
                raise AssertionError(f"Study 02:{variant.name}:seed_{seed} did not resolve L-GATr")
            if str(model.assignment.formulation) != "discriminative":
                raise AssertionError(f"Study 02:{variant.name}:seed_{seed} did not resolve M1")
            if bool(model.assignment.reasoning.enabled) or bool(model.assignment.feedback.enabled):
                raise AssertionError(f"Study 02:{variant.name}:seed_{seed} resolved recurrence")
            if list(str(value) for value in resolved.training.trainable) != ["invisible"]:
                raise AssertionError(f"Study 02:{variant.name}:seed_{seed} trains more than invisible")
            _gate_checkpoint(study02, "visible_assignment", "ttbar2L", seed)
            artifact = study02.gate_artifact("visible_assignment", "ttbar2L", seed)
            if int(artifact["model_seed"]) != seed or f"seed_{seed}" not in artifact["checkpoint"]:
                raise AssertionError(f"Study 02:{variant.name} used the wrong visible seed artifact")
        if any(prediction.recurrent_steps == 4 for prediction in variant.predictions):
            raise AssertionError(f"Study 02:{variant.name} has recurrent prediction steps")

    study11 = resolve_study(study_paths["data_scaling"])
    if {variant.name for variant in study11.variants} != {"n500k", "m3_n500k"}:
        raise AssertionError("Study 11 is not reduced to the two 500k stress-test variants")
    if any("500000" not in json.dumps(variant.patch) for variant in study11.variants):
        raise AssertionError("Study 11 has a non-500k variant")

    study15 = load_study(study_paths["lgatr_invisible_generator"])
    if {variant.name for variant in study15.training_order()} != {"conditional_lgatr"}:
        raise AssertionError(
            "Study 15 must train only Conditional L-GATr and reuse the conventional gate checkpoint"
        )
    if study15.variant_map["conventional"].checkpoint_from_gate != "invisible_coordinates":
        raise AssertionError("Study 15 conventional arm does not consume the selected coordinate checkpoint")
    if study15.variant_map["conditional_lgatr"].checkpoint_from_gate is not None:
        raise AssertionError("Study 15 Conditional L-GATr arm unexpectedly consumes a gate checkpoint")
    if study15.requires != (
        "visible_assignment", "invisible_generator", "invisible_coordinates", "assignment_formulation"
    ):
        raise AssertionError("Study 15 prerequisites changed unexpectedly")

    study13 = load_study(study_paths["invisible_coordinates"])
    if study13.requires != (
        "visible_assignment", "assignment_formulation", "invisible_generator"
    ):
        raise AssertionError("Study 13 does not explicitly inherit discriminative assignment")
    if study13.variant_map["cartesian"].checkpoint_from_gate != "invisible_generator":
        raise AssertionError("Study 13 Cartesian arm does not reuse the selected CFM checkpoint")
    if {variant.name for variant in study13.training_order()} != {
        "met_residual", "met_exact"
    }:
        raise AssertionError("Study 13 still retrains the selected Cartesian baseline")
    study03 = load_study(study_paths["assignment_conditioning"])
    if "assignment_coupling" not in study03.requires or len(study03.variants) != 3:
        raise AssertionError("Study 03 was not reduced to the gated reverse-pass comparison")
    study10 = load_study(study_paths["compatibility_reranking"])
    if study10.training or study10.variant_map["compatibility"].initialise_from_gate:
        raise AssertionError("Study 10 still schedules obsolete compatibility retraining")

    expected_campaign_states = {
        "04": ("deferred", False),
        "06": ("deferred", False),
        "07": ("deferred", False),
        "09": ("superseded", False),
    }
    for number, (state, enabled) in expected_campaign_states.items():
        path = next(path for path in study_paths.values() if path.name.startswith(f"{number}_"))
        study = load_study(path)
        if study.campaign_state != state or study.campaign_enabled is not enabled:
            raise AssertionError(
                f"Study {number} campaign state is {study.campaign_state}/{study.campaign_enabled}, "
                f"expected {state}/{enabled}"
            )
    from scripts.submit_campaign import ACTIVE_PAPER1

    if not ACTIVE_PAPER1 <= set(study_paths):
        raise AssertionError("The maintained Paper-1 campaign allowlist is incomplete")

    for study_name in (
        "generator_formulations",
        "discrete_assignment",
        "data_scaling",
        "generator_inference",
    ):
        study = resolve_study(study_paths[study_name])
        for variant in study.variants:
            model = variant.patch.get("model", {})
            if model.get("mode") != "factorized":
                raise AssertionError(f"{study_name}:{variant.name} is not factorized")
            if model.get("encoder", {}).get("type") != "lgatr":
                raise AssertionError(f"{study_name}:{variant.name} did not inherit L-GATr")

        if study_name == "discrete_assignment":
            m1 = study.variant_map["m1"]
            if (
                m1.patch.get("model", {}).get("assignment", {}).get("formulation")
                != "discriminative"
            ):
                raise AssertionError("Study 05 M1 did not inherit one-pass discriminative assignment")
        elif study_name in {"generator_formulations", "data_scaling", "generator_inference"}:
            for variant in study.variants:
                assignment = variant.patch.get("model", {}).get("assignment", {})
                if variant.reporting == "primary" and "refinement" in assignment:
                    raise AssertionError(f"{study_name}:{variant.name} inherited visible recurrence")

    for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        study = load_study(path)
        if study.split != "validation":
            raise AssertionError(f"Study selection split is not validation: {study.name}")
    stale_allowed = {"visible_recurrence", "data_scaling"}
    for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        study = load_study(path)
        text = path.read_text()
        if any(token in text for token in ("train_iterations", "inference_iterations", "feedback: soft", "recurrent_steps: 4")) and study.name not in stale_allowed:
            raise AssertionError(f"Unapproved inherited recurrence remains in {path}")


def check_slurm_option_routing_contract() -> None:
    """GPU-only options never leak into the CPU postprocess command."""
    import scripts.submit_study_slurm as scheduler

    worker = ROOT / "scripts/run_study.py"
    gpu = scheduler._sbatch_array(
        worker=worker,
        task_count=1,
        job_name="gpu-contract",
        log_dir=ROOT / "results/engineering",
        partition="gpuL",
        gres="gpu:1",
        cpus=1,
        memory="1G",
        time_limit="00:05:00",
        dependency=None,
        extra=("--account=gpu-project",),
    )
    cpu = scheduler._sbatch_array(
        worker=worker,
        task_count=1,
        job_name="cpu-contract",
        log_dir=ROOT / "results/engineering",
        partition="multicore",
        gres=None,
        cpus=1,
        memory="1G",
        time_limit="00:05:00",
        dependency=None,
        extra=("--account=cpu-project",),
    )
    if "--gres=gpu:1" not in gpu or "--account=gpu-project" not in gpu:
        raise AssertionError(f"GPU Slurm options were not routed: {gpu}")
    if any(value.startswith("--gres=gpu") or value == "--account=gpu-project" for value in cpu):
        raise AssertionError(f"GPU Slurm options leaked into CPU postprocess: {cpu}")
    if "--account=cpu-project" not in cpu:
        raise AssertionError(f"CPU Slurm option was not routed: {cpu}")


def check_campaign_identity_contract() -> None:
    """The campaign identity is deterministic and includes the resolved gate patch."""
    from scripts.submit_campaign import _identity
    from joint_event_reconstruction.studies import resolve_study

    study = resolve_study("configs/studies/12_encoder_backbone.yaml")
    first = _identity(study, "b0fc3583c94640fa7ec8b281935204c8bda64f16")
    second = _identity(study, "b0fc3583c94640fa7ec8b281935204c8bda64f16")
    if first != second or len(first[0]) != 64:
        raise AssertionError("Campaign identity is not deterministic")


def check_release_source_snapshot_contract() -> None:
    """Queued release workers must use the immutable source tree."""
    from joint_event_reconstruction.campaign_controller import (
        _release_source,
        campaign_paths,
    )

    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    source = _release_source(campaign_paths(), sha)
    expected = ROOT / "runtime/source" / sha
    if source != expected or not (source / "SOURCE_SNAPSHOT.json").is_file():
        raise AssertionError(f"Release source handoff is not immutable: {source}")


def check_paper1_controller_contract() -> None:
    """Campaign transitions, gate overlays and test sealing are explicit."""
    from joint_event_reconstruction.campaign_controller import transition_for_observation
    from joint_event_reconstruction.studies import load_scientific_gates

    state = {"state": "WAITING_SMOKE", "stage": None}
    state = transition_for_observation(state, {"smoke": "PASS"})
    state = transition_for_observation(
        state,
        {"study13": "COMPLETE", "decision_status": "AUTO_SELECTED", "decision": "cartesian"},
    )
    if state != {"state": "STUDY15_RUNNING", "stage": "invisible_backbone"}:
        raise AssertionError("Paper-1 controller did not promote Study 13 deterministically")
    review = transition_for_observation(
        {"state": "STUDY13_RUNNING", "stage": "invisible_coordinates"},
        {"study13": "COMPLETE", "decision_status": "REVIEW_REQUIRED"},
    )
    if review["state"] != "STUDY13_REVIEW":
        raise AssertionError("Ambiguous Study-13 result did not stop at REVIEW_REQUIRED")
    failed = transition_for_observation(
        {"state": "STUDY13_RUNNING", "stage": "invisible_coordinates"},
        {"study13": "FAILED"},
    )
    if failed["state"] != "BLOCKED":
        raise AssertionError("Training failure was not fail-closed")

    with tempfile.TemporaryDirectory(prefix="jer-campaign-gate-contract-") as directory:
        gates = copy.deepcopy(load_scientific_gates())
        gate = gates["invisible_coordinates"]
        base_artifact = gates["invisible_generator"]["artifacts"]["ttbar2L"]["seed_42"]
        gate.update(
            {
                "status": "locked",
                "selected_variant": "cartesian",
                "validated_sha": "30da9fce364ed479a2d980c5acc23d3e817c180a",
                "selected_patch": {"model": {"invisible": {"coordinates": {"type": "cartesian"}}}},
                "artifacts": {"ttbar2L": {"seed_42": copy.deepcopy(base_artifact)}},
            }
        )
        path = Path(directory) / "gates.json"
        path.write_text(json.dumps(gates))
        try:
            load_scientific_gates(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Reused checkpoint passed without selection provenance")
        entry = gate["artifacts"]["ttbar2L"]["seed_42"]
        entry["selection_provenance"] = {
            "source_study": "invisible_coordinates",
            "variant": "cartesian",
            "decision_artifact": "results/campaigns/paper1/decisions/invisible_coordinates.json",
            "decision_sha256": "0" * 64,
        }
        entry["training_provenance"] = {
            "source_study": entry["source_study"],
            "variant": entry["variant"],
            "git_sha": entry["git_sha"],
            "architecture_identity": entry["architecture_identity"],
            "identity_sha256": entry["identity_sha256"],
        }
        path.write_text(json.dumps(gates))
        load_scientific_gates(path)

    policy = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/campaigns/paper1.yaml"), resolve=True
    )
    if policy["campaign"]["stop_before_test"] is not True:
        raise AssertionError("Paper-1 campaign does not stop before the real test split")
    for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        study = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["study"]
        if study.get("campaign", {}).get("state") in {"deferred", "superseded"} and study.get("campaign", {}).get("enabled"):
            raise AssertionError(f"Deferred/superseded study is still auto-enabled: {path}")


def _component_signatures(overrides: list[str] | None = None) -> dict[str, str]:
    config = load_experiment("configs/experiments/ttbar2L.yaml", overrides or [])
    from joint_event_reconstruction.training.joint import architecture_component_signatures

    return architecture_component_signatures(config.model)


def check_partial_checkpoint_contract() -> None:
    import torch
    from scripts.smoke_synthetic import build_synthetic_batch

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    visible, _, scaler = build_synthetic_batch(base, events=2, device="cpu")
    parent = JointEventModule(
        base, visible, invisible_scaler=scaler, synthetic_data=True
    )
    with tempfile.TemporaryDirectory(prefix="jer-partial-checkpoint-") as directory:
        checkpoint_path = f"{directory}/parent.ckpt"
        torch.save(
            {
                "state_dict": parent.state_dict(),
                "hyper_parameters": dict(parent.hparams),
                "invisible_scaler": parent.scaler_metadata,
            },
            checkpoint_path,
        )

        target_config = load_experiment(
            "configs/experiments/ttbar2L.yaml", ["model.invisible.generator.type=cfm"]
        )
        target = JointEventModule(
            target_config,
            visible,
            invisible_scaler=scaler,
            synthetic_data=True,
        )
        unselected = {
            key: value.detach().clone()
            for key, value in target.state_dict().items()
            if key.startswith("event_model.invisible.")
        }
        _load_initial_components(target, checkpoint_path, ("visible", "assignment"))
        parent_state = parent.state_dict()
        target_state = target.state_dict()
        for key, value in parent_state.items():
            if key.startswith(("event_model.visible.", "event_model.reconstruction.")):
                if not torch.equal(target_state[key], value):
                    raise AssertionError(f"Selected checkpoint value was not loaded: {key}")
        for key, value in unselected.items():
            if not torch.equal(target_state[key], value):
                raise AssertionError(f"Unselected checkpoint component changed: {key}")

        changed_config = load_experiment(
            "configs/experiments/ttbar2L.yaml", ["model.encoder.dropout=0.2"]
        )
        changed = JointEventModule(
            changed_config,
            visible,
            invisible_scaler=scaler,
            synthetic_data=True,
        )
        try:
            _load_initial_components(changed, checkpoint_path, ("visible",))
        except ValueError as error:
            if "component signatures" not in str(error):
                raise AssertionError(f"Unexpected partial-load error: {error}") from error
        else:
            raise AssertionError("Changed selected component was accepted")


def check_component_signature_contract() -> None:
    reference = _component_signatures()
    inference_only = (
        "inference.seed=7",
        "inference.assignment.top_k=5",
        "inference.invisible.samples_per_assignment=64",
        "inference.invisible.flow.steps=16",
        "inference.invisible.flow.solver=euler",
        "inference.sample_chunk_size=4",
        "inference.invisible.edm.nfe=33",
        "inference.compatibility.weight=2.0",
    )
    for override in inference_only:
        if _component_signatures([override]) != reference:
            raise AssertionError(f"Inference-only override changed component identity: {override}")

    if _component_signatures(["model.encoder.dropout=0.2"])["visible"] == reference["visible"]:
        raise AssertionError("Visible architecture change was omitted from component identity")
    if _component_signatures(["model.assignment.refinement.train_iterations=[1,2]"])["assignment"] == reference["assignment"]:
        raise AssertionError("Assignment architecture change was omitted from component identity")

    c2_base = [
        "model.assignment.refinement.train_iterations=[1,2]",
        "model.assignment.refinement.inference_iterations=2",
        "model.invisible.generator.type=cfm",
        "model.feedback.invisible_to_assignment.enabled=true",
    ]
    c2_reference = _component_signatures(c2_base)
    if _component_signatures([*c2_base, "model.feedback.invisible_to_assignment.training_sampler.steps=9"])["c2"] == c2_reference["c2"]:
        raise AssertionError("C2 training sampler change was omitted from component identity")


def check_nested_subsets() -> None:
    event_ids = np.arange(10_000, dtype=np.uint64) * 17 + 3
    previous: set[int] = set()
    for count in (100, 250, 500, 1000):
        positions = deterministic_subset_positions(event_ids, count, seed=42)
        repeated = deterministic_subset_positions(event_ids, count, seed=42)
        np.testing.assert_array_equal(positions, repeated)
        selected = set(event_ids[positions].tolist())
        if not previous < selected:
            raise AssertionError(f"Stable subsets are not strictly nested at {count}")
        previous = selected


def check_study10_control_contract() -> None:
    """Study-10 controls use train-only fitting and deterministic K/M prefixes."""
    from joint_event_reconstruction.study10 import (
        coupling_efficiency,
        nested_prefix_reduction,
        select_alpha,
        top_k_coverage,
        train_scorers,
        transition_counts,
        visible_only_scorer_input,
    )

    torch.manual_seed(19)
    batch, top_k, samples, feature_count = 3, 10, 64, 4
    posterior_features = torch.randn(batch, top_k, samples, feature_count)
    valid = torch.ones(batch, top_k, dtype=torch.bool)
    valid[1, 8:] = False
    first = nested_prefix_reduction(posterior_features, valid)
    second = nested_prefix_reduction(posterior_features, valid)
    if not torch.equal(first, second) or first.shape != (5, 5, batch, top_k):
        raise AssertionError("Study-10 nested K/M reduction is not deterministic or K=10 capable")
    if not torch.isneginf(first[:, :, :, -1][0, 0, 0]) or not torch.isfinite(first[-1, -1, :, :8]).all():
        raise AssertionError("Study-10 nested prefix padding semantics are incorrect")

    visible_log_score = torch.randn(batch, top_k)
    visible_probability = torch.softmax(visible_log_score, dim=1)
    entropy = torch.rand(batch, top_k)
    margin = torch.rand(batch, top_k)
    visible_input = visible_only_scorer_input(
        visible_log_score,
        visible_probability,
        entropy,
        margin,
        target_features=12,
    )
    changed_physics = visible_input.clone()
    changed_physics[..., 4:] = torch.randn_like(changed_physics[..., 4:])
    if not torch.equal(visible_input[..., :4], changed_physics[..., :4]):
        raise AssertionError("Visible-only Study-10 control does not preserve its explicit input contract")
    if not torch.equal(visible_input[..., 4:], torch.zeros_like(visible_input[..., 4:])):
        raise AssertionError("Visible-only Study-10 control received physics features in padding")

    truth = torch.zeros(batch, top_k, dtype=torch.bool)
    truth[:, 0] = True
    physics = torch.randn(batch, top_k, 12)
    scorers = train_scorers(
        visible_log_score,
        visible_probability,
        physics,
        visible_input,
        truth,
        valid,
        hidden=8,
        epochs=2,
    )
    with torch.no_grad():
        if not torch.isfinite(scorers.physics(physics)).all():
            raise AssertionError("Study-10 learned physics scorer produced non-finite output")
        if not torch.isfinite(scorers.visible_only(visible_input)).all():
            raise AssertionError("Study-10 visible-only control produced non-finite output")
    alpha, scan = select_alpha(visible_log_score, torch.randn(batch, top_k), truth, valid, (0.0, 1.0))
    if alpha not in {0.0, 1.0} or set(scan) != {"0.0", "1.0"}:
        raise AssertionError("Study-10 alpha selection did not use the declared training scan")
    if top_k_coverage(truth, valid, 10).shape != (batch,):
        raise AssertionError("Study-10 top-K coverage has the wrong shape")
    transitions = transition_counts(truth, truth)
    if transitions["baseline_correct_to_remains_correct"] != batch:
        raise AssertionError("Study-10 transition accounting is incorrect")
    coupling = coupling_efficiency(0.50, 0.53, 0.60)
    if abs(float(coupling["Rcoupling"]) - 0.3) > 1.0e-6:
        raise AssertionError("Study-10 Rcoupling calculation is incorrect")


def check_campaign_scheduler_contract() -> None:
    """Pending Slurm arrays and study filters must never look like failures."""
    from scripts.submit_campaign import _classify_array, _parse_study_filter

    cases = (
        (("PENDING", {}, 4), "PENDING"),
        (("CONFIGURING", {}, 4), "PENDING"),
        (("RUNNING", {0: "RUNNING"}, 4), "RUNNING"),
        (("COMPLETED", {}, 4), "PENDING"),
        (("COMPLETED", {0: "COMPLETED", 1: "COMPLETED"}, 2), "COMPLETED"),
        (("COMPLETED", {0: "COMPLETED", 1: "FAILED"}, 2), "FAILED"),
        (("TIMEOUT", {}, 4), "FAILED"),
    )
    for (parent, children, expected_tasks), expected in cases:
        observed, _ = _classify_array(parent, children, expected_tasks)
        if observed != expected:
            raise AssertionError(
                f"Unexpected Slurm array classification for {parent!r}, "
                f"children={children}: {observed!r} != {expected!r}"
            )
    if _parse_study_filter(None, {"a"}) is not None:
        raise AssertionError("Missing study filter must select the maintained campaign")
    if _parse_study_filter("a,b,a", {"a", "b"}) != {"a", "b"}:
        raise AssertionError("Study filter did not normalise selected names")
    try:
        _parse_study_filter("missing", {"a"})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown study filter name was accepted")


def check_loader_split_seed_contract() -> None:
    base = load_experiment("configs/experiments/ttbar2L.yaml")
    changed_model_seed = load_experiment(
        "configs/experiments/ttbar2L.yaml", ["experiment.seed=123"]
    )
    for experiment in (base, changed_model_seed):
        visible = load_visible_config(experiment)
        if int(visible.general.seed) != int(experiment.data.subset_seed):
            raise AssertionError(
                "Visible-loader split seed must mirror data.subset_seed, not "
                "experiment.seed"
            )
    if int(base.experiment.seed) == int(changed_model_seed.experiment.seed):
        raise AssertionError("Loader seed contract test did not change model seed")


def check_public_schema_and_inputs_contract() -> None:
    """Old public spellings fail, Study 12 stays fair, and p4 units are explicit."""
    from joint_event_reconstruction.inputs import (
        assert_study12_fairness,
        study_input_contracts,
    )

    assert_study12_fairness(
        study_input_contracts("configs/studies/12_encoder_backbone.yaml")
    )
    obsolete = (
        "data.files.events=/tmp/events.h5",
        "data.invisible_truth=/tmp/truth.h5",
        "data.split=/tmp/split.npz",
        "inputs.JET.max_count=8",
        "inputs.event_features.njet=none",
        "prediction.seed=7",
        "model.assignment.method=discriminative",
        "model.assignment.recurrence.train_steps=[1,2]",
        "model.invisible.method=cfm",
        "model.encoder.scalar_features=[btag]",
        "model.encoder.spurions.beam=none",
        "model.encoder.type=physics_transformer",
        "model.encoder.type=relational_transformer",
        "model.encoder.inputs.kinematics=four_vector",
    )
    for override in obsolete:
        try:
            load_experiment("configs/experiments/ttbar2L.yaml", [override])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Obsolete public config spelling was accepted: {override}")

    from scripts.smoke_synthetic import build_synthetic_batch

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    _, batch, _ = build_synthetic_batch(base, events=2, device="cpu")
    vectors = batch["physics_vectors"]
    if vectors.shape[-1] != 4 or not torch.isfinite(vectors).all():
        raise AssertionError("Physical vectors are not finite (E,px,py,pz)")
    met_type = batch["type_ids"] == 3
    if not torch.equal(vectors[..., 3][met_type], torch.zeros_like(vectors[..., 3][met_type])):
        raise AssertionError("Detector MET pseudo-vectors must have pz=0")


def check_same_flavour_role_contract() -> None:
    """Flavour never replaces semantic parent identity for invisible roles."""
    from joint_event_reconstruction.topology.compiler import compile_topology

    raw = {
        "topology": {
            "top": {
                "nu": {"source": "NEUTRINO", "truth": "top_neutrino"},
            },
            "antitop": {
                "nu": {"source": "NEUTRINO", "truth": "antitop_neutrino"},
            },
        }
    }
    bindings = {
        "top_neutrino": {"source": "NEUTRINO", "index": 0},
        "antitop_neutrino": {"source": "NEUTRINO", "index": 1},
    }
    spec = compile_topology(raw, bindings)
    if tuple(role.key for role in spec.invisible_roles) != ("top.nu", "antitop.nu"):
        raise AssertionError("Same-flavour invisible roles lost semantic parent ordering")
    if tuple(role.truth_indices[0] for role in spec.invisible_roles) != (0, 1):
        raise AssertionError("Same-flavour invisible truth bindings collided")


def check_truth_oracle_unreconstructable_contract() -> None:
    """Truth-oracle conditioning must mask, rather than index, missing parents."""
    import torch

    from scripts.smoke_synthetic import build_synthetic_batch

    config = load_experiment(
        "configs/experiments/ttbar2L.yaml",
        [
            "model.invisible.generator.type=cfm",
            "model.invisible.conditioning.assignment=truth_oracle",
            "training.trainable=[visible,assignment,invisible]",
            "training.loss_weights.compatibility=0.0",
        ],
    )
    visible, batch, scaler = build_synthetic_batch(config, events=2, device="cpu")
    node = next(iter(batch["structured"]["pools"]))
    pool = batch["structured"]["pools"][node]
    pool["truth_index"][0] = -1
    pool["reconstructable"][0] = False
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise AssertionError("Truth-oracle masking produced a non-finite loss")
    if bool(output["invisible_training_mask"][0]):
        raise AssertionError("Unreconstructable truth-oracle event was not masked")

    from joint_event_reconstruction.physics import compose_assignment_state
    from joint_event_reconstruction.prediction import compose_physics, truth_node_indices

    valid_batch = copy.deepcopy(batch)
    valid_batch["structured"]["pools"][node]["truth_index"][0] = 0
    valid_batch["structured"]["pools"][node]["reconstructable"][0] = True
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    samples = torch.zeros(
        (2, 1, 2, len(module.spec.invisible_roles), 3), dtype=torch.float32
    )
    composites, _, _ = compose_physics(
        module,
        valid_batch,
        [(), ()],
        samples,
        truth_assignment=True,
    )
    truth_indices = truth_node_indices(module.spec, valid_batch, 1)
    oracle_invisible = {
        role.key: samples[1, 0, 0, role_index]
        for role_index, role in enumerate(module.spec.invisible_roles)
    }
    expected = compose_assignment_state(
        module.spec,
        valid_batch["structured"]["pools"],
        valid_batch["physics_vectors"][1],
        truth_indices,
        oracle_invisible,
    )
    for name in module.spec.dag_order:
        expected_values = torch.stack([expected[name], expected[name]])
        if not torch.allclose(
            torch.from_numpy(composites[name][1, 0]),
            expected_values,
            equal_nan=True,
        ):
            raise AssertionError(
                "Truth-oracle composite output did not use the truth assignment"
            )

    alternative, _, _ = compose_physics(
        module,
        valid_batch,
        [[object()], [object()]],
        samples,
        truth_assignment=True,
    )
    for name in module.spec.dag_order:
        if not np.allclose(composites[name], alternative[name], equal_nan=True):
            raise AssertionError(
                "Truth-oracle composite depends on model-predicted assignment"
            )

    changed_samples = samples.clone()
    changed_samples[1, 0, 0, 0, 0] = 1.0
    changed, _, _ = compose_physics(
        module,
        valid_batch,
        [(), ()],
        changed_samples,
        truth_assignment=True,
    )
    if all(
        np.allclose(composites[name][1, 0, 0], changed[name][1, 0, 0])
        for name in module.spec.dag_order
    ):
        raise AssertionError(
            "Truth-oracle composite is insensitive to its generated invisible sample"
        )

    unsupported, _, _ = compose_physics(
        module,
        batch,
        [(), ()],
        samples,
        truth_assignment=True,
    )
    for name in module.spec.dag_order:
        if not np.isnan(unsupported[name][0]).all():
            raise AssertionError(
                "Unsupported truth-oracle composite must be explicitly all-NaN"
            )


def _permuted_same_type_batch(batch: dict, positions) -> dict:
    """Permute same-type token positions and all static candidate indices with them."""
    import torch

    result = copy.deepcopy(batch)
    token_count = batch["tokens"].shape[1]
    permutation = torch.arange(token_count)
    positions = torch.as_tensor(positions, dtype=torch.long)
    permutation[positions] = positions.flip(0)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(token_count)
    for name in ("tokens", "type_ids", "padding_mask", "physics_vectors"):
        if name in result:
            result[name] = result[name][:, permutation]
    for pool in result["structured"]["pools"].values():
        pool["leaf_ownership"] = pool["leaf_ownership"][:, permutation]
        pool["leaf_indices"] = {
            role: torch.where(
                indices >= 0, inverse[indices.clamp_min(0)], indices
            )
            for role, indices in pool["leaf_indices"].items()
        }
    return result


def check_exchange_and_permutation_contract() -> None:
    import torch

    from scripts.smoke_synthetic import build_synthetic_batch
    from joint_event_reconstruction.models.reconstruction.structured import (
        SymmetricPairEncoder,
        SymmetricSetEncoder,
    )

    torch.manual_seed(701)
    pair = SymmetricPairEncoder(8, 0.0).eval()
    pair_members = torch.randn(2, 3, 2, 8)
    _assert_close(pair(pair_members), pair(pair_members.flip(2)), "pair exchange symmetry", 0.0)

    set_encoder = SymmetricSetEncoder(8, 0.0).eval()
    set_members = torch.randn(2, 3, 4, 8)
    _assert_close(
        set_encoder(set_members),
        set_encoder(set_members[:, :, (2, 0, 3, 1)]),
        "set exchange symmetry",
        2.0e-5,
    )

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    visible, batch, scaler = build_synthetic_batch(base, events=2, device="cpu")
    model = JointEventModule(
        base, visible, invisible_scaler=scaler, synthetic_data=True
    ).eval()
    permuted = _permuted_same_type_batch(batch, (0, 1, 2, 3, 4, 5, 6, 7))
    with torch.inference_mode():
        original = model(batch)
        transformed = model(permuted)
    _assert_close(
        original["visible"]["pooled"],
        transformed["visible"]["pooled"],
        "visible same-type permutation equivariance",
    )
    for node in model.spec.dag_order:
        _assert_close(
            original["assignment_logits"][node],
            transformed["assignment_logits"][node],
            f"{node} assignment permutation equivariance",
        )


def _check_one_freeze_case(
    config,
    visible,
    batch,
    scaler,
    expected_prefixes: tuple[str, ...],
    name: str,
) -> None:
    import torch

    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    frozen = {
        key: value.detach().clone()
        for key, value in module.named_parameters()
        if not value.requires_grad
    }
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise AssertionError(f"{name} produced a non-finite loss")
    module.zero_grad(set_to_none=True)
    output["loss"].backward()
    intended = [
        parameter
        for key, parameter in module.named_parameters()
        if parameter.requires_grad
        and key.startswith(expected_prefixes)
        and parameter.grad is not None
    ]
    if not intended or not all(torch.isfinite(parameter.grad).all() for parameter in intended):
        raise AssertionError(f"{name} did not produce finite intended gradients")
    optimiser = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1.0e-4,
        fused=False,
    )
    optimiser.step()
    for key, before in frozen.items():
        if not torch.equal(module.state_dict()[key], before):
            raise AssertionError(f"{name} changed frozen parameter {key}")


def check_freeze_contract() -> None:
    from scripts.smoke_all import configure as configure_factorized
    from scripts.smoke_flows import configure as configure_flow
    from scripts.smoke_synthetic import build_synthetic_batch

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    visible, batch, scaler = build_synthetic_batch(base, events=2, device="cpu")

    assignment = copy.deepcopy(base)
    assignment.training.trainable = ["visible", "assignment"]
    assignment.training.loss_weights.invisible = 0.0
    assignment.training.loss_weights.compatibility = 0.0
    _check_one_freeze_case(
        assignment,
        visible,
        batch,
        scaler,
        ("event_model.visible.", "event_model.reconstruction."),
        "visible+assignment",
    )

    invisible = configure_flow(base, "cfm_no_assignment")
    invisible.training.trainable = ["invisible"]
    invisible.training.loss_weights.assignment = 0.0
    invisible.training.loss_weights.invisible = 1.0
    _check_one_freeze_case(
        invisible,
        visible,
        batch,
        scaler,
        ("event_model.invisible.",),
        "invisible-only",
    )

    c2 = configure_factorized(base, "edm_c2_detached")
    c2.training.trainable = ["assignment", "c2"]
    c2.training.loss_weights.assignment = 1.0
    c2.training.loss_weights.invisible = 0.0
    c2.training.loss_weights.compatibility = 0.0
    _check_one_freeze_case(
        c2,
        visible,
        batch,
        scaler,
        (
            "event_model.reconstruction.",
            "event_model.reasoner.",
            "event_model.feedback.",
            "event_model.posterior_feedback.",
        ),
        "assignment+c2",
    )

    one_pass_c2 = configure_factorized(base, "edm_c2_detached")
    one_pass_c2.model.assignment.reasoning.enabled = False
    one_pass_c2.model.assignment.feedback.enabled = False
    one_pass_c2.training.trainable = ["assignment", "c2"]
    one_pass_c2.training.loss_weights.assignment = 1.0
    one_pass_c2.training.loss_weights.invisible = 0.0
    one_pass_c2.training.loss_weights.compatibility = 0.0
    from joint_event_reconstruction.configuration import validate_experiment

    validate_experiment(one_pass_c2)
    _check_one_freeze_case(
        one_pass_c2,
        visible,
        batch,
        scaler,
        ("event_model.reconstruction.", "event_model.posterior_feedback."),
        "one-pass assignment+c2",
    )

    compatibility = configure_factorized(base, "joint_compatibility")
    compatibility.training.trainable = ["compatibility"]
    compatibility.training.loss_weights.assignment = 0.0
    compatibility.training.loss_weights.invisible = 0.0
    compatibility.training.loss_weights.compatibility = 1.0
    _check_one_freeze_case(
        compatibility,
        visible,
        batch,
        scaler,
        ("event_model.compatibility.",),
        "compatibility-only",
    )

    joint = configure_flow(base, "joint_flow")
    joint.training.trainable = ["assignment", "invisible"]
    joint.training.loss_weights.assignment = 1.0
    joint.training.loss_weights.invisible = 1.0
    _check_one_freeze_case(
        joint,
        visible,
        batch,
        scaler,
        ("event_model.joint_flow.", "event_model.invisible."),
        "joint assignment+invisible",
    )


def main() -> None:
    check_architecture_signature_contract()
    check_training_identity_contract()
    check_component_signature_contract()
    check_partial_checkpoint_contract()
    check_nested_subsets()
    check_study10_control_contract()
    check_loader_split_seed_contract()
    check_public_schema_and_inputs_contract()
    check_same_flavour_role_contract()
    check_coordinate_contract()
    check_mass_shell_contract()
    check_topology_physics_contract()
    check_conditional_lgatr_contract()
    check_prediction_writer_shape_contract()
    check_physical_posterior_shape_contract()
    check_flow_coupling_context_contract()
    check_truth_oracle_unreconstructable_contract()
    check_exchange_and_permutation_contract()
    check_freeze_contract()
    check_single_root_runtime_contract()
    check_slurm_snapshot_contract()
    check_postprocess_recovery_contract()
    check_campaign_gate_contract()
    check_campaign_scheduler_contract()
    check_slurm_option_routing_contract()
    check_campaign_identity_contract()
    check_release_source_snapshot_contract()
    check_paper1_controller_contract()
    print(
        "PASS: architecture, training-identity, partial-checkpoint, split-seed, "
        "nested-subset, coordinate, truth-oracle, exchange, permutation, "
        "freeze, public-input, same-flavour-role, campaign-gate, Slurm-option, "
        "campaign-identity, release-source-snapshot, campaign-scheduler, Paper-1-controller, prediction-writer, mass-shell, Conditional-L-GATr "
        "Slurm-snapshot and resumable-postprocess contracts"
    )


if __name__ == "__main__":
    main()
