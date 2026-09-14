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

    from joint_event_reconstruction.prediction import _create

    with tempfile.TemporaryDirectory(prefix="jer-prediction-writer-") as directory:
        path = Path(directory) / "prediction.h5"
        with h5py.File(path, "w") as handle:
            dataset = _create(
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


def check_slurm_snapshot_contract() -> None:
    """Generated workers accept an unchanged snapshot and reject later edits."""
    import scripts.submit_study_slurm as scheduler

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
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repository, check=True)
        initial_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

        manifest = root / "study_manifest.yaml"
        manifest.write_text("study: contract\n")
        task_file = root / "tasks.tsv"
        task_file.write_text("ttbar2L\tvariant\t42\n")
        expected_manifest = scheduler._sha256_file(manifest)
        expected_experiment = scheduler._sha256_file(experiment)
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
                expected_manifest_sha256=expected_manifest,
                expected_experiments={
                    "ttbar2L": {
                        "path": str(experiment),
                        "sha256": expected_experiment,
                    }
                },
            )
        finally:
            scheduler.ROOT = original_root

        environment = dict(os.environ, SLURM_ARRAY_TASK_ID="0")
        accepted = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if accepted.returncode != 0:
            raise AssertionError(f"Unchanged Slurm snapshot was rejected: {accepted.stderr}")

        manifest.write_text("study: changed\n")
        changed_manifest = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if changed_manifest.returncode != 3 or "manifest snapshot changed" not in changed_manifest.stderr:
            raise AssertionError("Changed Slurm manifest was not rejected before work")

        manifest.write_text("study: contract\n")
        experiment.write_text("experiment: changed\n")
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "changed experiment"], cwd=repository, check=True)
        changed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        changed_checkout = subprocess.run(
            [str(worker)], text=True, capture_output=True, env=environment, check=False
        )
        if changed_checkout.returncode != 3 or "Checkout changed" not in changed_checkout.stderr:
            raise AssertionError("Changed Git checkout was not rejected before work")

        config_worker = root / "config-worker.sh"
        scheduler.ROOT = repository
        try:
            scheduler._worker_script(
                config_worker,
                task_file=task_file,
                study_path=manifest,
                stage="train",
                output_root=root / "results",
                device="cpu",
                global_overrides=(),
                expected_commit=changed_commit,
                expected_manifest_sha256=expected_manifest,
                expected_experiments={
                    "ttbar2L": {
                        "path": str(experiment),
                        "sha256": expected_experiment,
                    }
                },
            )
        finally:
            scheduler.ROOT = original_root
        changed_config = subprocess.run(
            [str(config_worker)],
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        if changed_config.returncode != 3 or "Base experiment changed" not in changed_config.stderr:
            raise AssertionError("Changed base experiment was not rejected before work")


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
    if study15.variant_map["conventional"].checkpoint_from_gate != "invisible_generator":
        raise AssertionError("Study 15 conventional arm does not consume the generator gate checkpoint")
    if study15.variant_map["conditional_lgatr"].checkpoint_from_gate is not None:
        raise AssertionError("Study 15 Conditional L-GATr arm unexpectedly consumes a gate checkpoint")

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
    from joint_event_reconstruction.prediction import _compose_physics, _truth_node_indices

    valid_batch = copy.deepcopy(batch)
    valid_batch["structured"]["pools"][node]["truth_index"][0] = 0
    valid_batch["structured"]["pools"][node]["reconstructable"][0] = True
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    samples = torch.zeros(
        (2, 1, 2, len(module.spec.invisible_roles), 3), dtype=torch.float32
    )
    composites, _, _ = _compose_physics(
        module,
        valid_batch,
        [(), ()],
        samples,
        truth_assignment=True,
    )
    truth_indices = _truth_node_indices(module.spec, valid_batch, 1)
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

    alternative, _, _ = _compose_physics(
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
    changed, _, _ = _compose_physics(
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

    unsupported, _, _ = _compose_physics(
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
    check_loader_split_seed_contract()
    check_public_schema_and_inputs_contract()
    check_same_flavour_role_contract()
    check_coordinate_contract()
    check_mass_shell_contract()
    check_conditional_lgatr_contract()
    check_prediction_writer_shape_contract()
    check_flow_coupling_context_contract()
    check_truth_oracle_unreconstructable_contract()
    check_exchange_and_permutation_contract()
    check_freeze_contract()
    check_slurm_snapshot_contract()
    check_campaign_gate_contract()
    check_slurm_option_routing_contract()
    check_campaign_identity_contract()
    print(
        "PASS: architecture, training-identity, partial-checkpoint, split-seed, "
        "nested-subset, coordinate, truth-oracle, exchange, permutation, "
        "freeze, public-input, same-flavour-role, campaign-gate, Slurm-option, "
        "campaign-identity, prediction-writer, mass-shell, Conditional-L-GATr "
        "and Slurm-snapshot contracts"
    )


if __name__ == "__main__":
    main()
