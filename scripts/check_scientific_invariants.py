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
    """The current gate state resolves Study 12 and blocks downstream science."""
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

    study12 = resolve_study("configs/studies/12_encoder_backbone.yaml")
    if study12.seeds != (42, 43, 44):
        raise AssertionError("Study 12 must use model seeds 42, 43 and 44")
    for variant in study12.variants:
        assignment_patch = variant.patch.get("model", {}).get("assignment", {})
        if "refinement" in assignment_patch:
            raise AssertionError(f"Study 12 inherited visible recurrence: {variant.name}")
        if any(prediction.recurrent_steps is not None for prediction in variant.predictions):
            raise AssertionError(f"Study 12 has recurrent prediction steps: {variant.name}")
    try:
        resolve_study("configs/studies/02_generator_formulations.yaml")
    except ScientificGatePending as error:
        if "visible_assignment" not in str(error):
            raise AssertionError(f"Unexpected Study-02 gate barrier: {error}") from error
    else:
        raise AssertionError("Study 02 resolved while visible_assignment is pending")

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
        "campaign-identity and Slurm-snapshot contracts"
    )


if __name__ == "__main__":
    main()
