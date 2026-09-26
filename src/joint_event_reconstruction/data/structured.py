"""Add topology-derived hierarchical candidate tensors to visible-event batches."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from joint_event_reconstruction.data.invisible import (
    InvisibleScaler,
    InvisibleTruthStore,
    ParentTruthStore,
)
from joint_event_reconstruction.data.partonic import PartonicScaler, PartonicTruthStore
from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.m0.data.module import (
    TransformerDataModule,
    _collate_transformer,
)
from joint_event_reconstruction.models.invisible.coordinates import (
    build_invisible_coordinates,
    met_xy_from_batch,
)
from joint_event_reconstruction.topology.assignments import build_assignment_template
from joint_event_reconstruction.topology.candidates import build_event_candidates
from joint_event_reconstruction.topology.compiler import load_topology

logger = logging.getLogger(__name__)


class VisibleTruthStore:
    """Event-ID joined visible labels kept outside the measured-input cache."""

    def __init__(self, path: str | Path, source_names: tuple[str, ...]):
        self.path = Path(path).expanduser().resolve()
        self.source_names = tuple(str(name) for name in source_names)
        if not self.path.is_file():
            raise FileNotFoundError(f"Visible truth sidecar does not exist: {self.path}")
        with h5py.File(self.path, "r") as source:
            if "EVENT_ID" not in source or "LABELS" not in source:
                raise KeyError(f"{self.path} must contain EVENT_ID and LABELS/<source>")
            event_ids = np.asarray(source["EVENT_ID"][:])
            if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
                raise TypeError(f"{self.path}: EVENT_ID must be a one-dimensional integer array")
            labels = {}
            for name in self.source_names:
                path_name = f"LABELS/{name}"
                if path_name not in source:
                    raise KeyError(f"{self.path}: missing {path_name}")
                values = np.asarray(source[path_name][:])
                if values.ndim != 2 or values.shape[0] != len(event_ids):
                    raise ValueError(
                        f"{self.path}:{path_name} must have shape [events,objects], "
                        f"found {values.shape}"
                    )
                if not np.issubdtype(values.dtype, np.number):
                    raise TypeError(f"{self.path}:{path_name} must be numeric")
                finite = np.isfinite(values)
                if finite.any() and (
                    not np.equal(values[finite], np.rint(values[finite])).all()
                    or not np.isin(values[finite], (0, 1, 2)).all()
                ):
                    raise ValueError(
                        f"{self.path}:{path_name} contains invalid raw labels; "
                        "expected integer labels 0, 1, 2 or NaN padding"
                    )
                labels[name] = values.astype(np.float32, copy=False)
            support_path = "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"
            if support_path not in source:
                raise KeyError(f"{self.path}: missing {support_path}")
            support = np.asarray(source[support_path][:])
            if support.shape != (len(event_ids),) or support.dtype != np.bool_:
                raise TypeError(
                    f"{self.path}:{support_path} must be boolean [{len(event_ids)}]"
                )
            metadata = {}
            for key in (
                "source_manifest_sha256",
                "split_membership_sha256",
                "split_version",
                "adapter_version",
                "source_path",
                "max_jets",
            ):
                value = source.attrs.get(key)
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if value is not None:
                    metadata[key] = str(value)
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{self.path}: EVENT_ID contains duplicates")
        order = np.argsort(event_ids)
        self.event_ids = event_ids[order].astype(np.int64, copy=False)
        self.labels = {name: values[order] for name, values in labels.items()}
        self.support = support[order]
        stat = self.path.stat()
        self.metadata = {
            "path": str(self.path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "source_names": list(self.source_names),
            **metadata,
        }

    def lookup(self, event_ids: torch.Tensor | np.ndarray):
        requested = np.asarray(
            event_ids.detach().cpu() if torch.is_tensor(event_ids) else event_ids,
            dtype=np.int64,
        )
        positions = np.searchsorted(self.event_ids, requested)
        valid = positions < len(self.event_ids)
        valid &= self.event_ids[np.minimum(positions, len(self.event_ids) - 1)] == requested
        if not valid.all():
            raise KeyError(
                f"Visible truth sidecar {self.path} is missing EVENT_ID values "
                f"{requested[~valid][:10].tolist()}"
            )
        return (
            {name: torch.from_numpy(values[positions].copy()) for name, values in self.labels.items()},
            torch.from_numpy(self.support[positions].copy()),
        )


class StructuredCollator:
    """Attach topology-derived candidate/assignment tensors to a fixed token layout."""

    def __init__(
        self,
        topology_path: str | Path,
        source_type_ids: dict[str, int],
        *,
        truth_bindings: dict | None = None,
        build_assignments: bool = False,
        max_assignment_states: int | None = None,
        require_truth: bool = True,
    ):
        self.topology_path = str(Path(topology_path).expanduser().resolve())
        self.source_type_ids = dict(source_type_ids)
        self.truth_bindings = truth_bindings
        self.build_assignments = bool(build_assignments)
        self.max_assignment_states = (
            None if max_assignment_states is None else int(max_assignment_states)
        )
        self.require_truth = bool(require_truth)
        if self.max_assignment_states is not None and self.max_assignment_states <= 0:
            raise ValueError("max_assignment_states must be positive or null")
        self._spec = None
        self._templates = None
        self._assignment_template = None
        self._type_layout = None
        self.visible_store = None
        self.invisible_store = None
        self.invisible_scaler = None
        self.point_scaler = None
        self.parent_store = None
        self.partonic_store = None
        self.partonic_scaler = None
        self.invisible_coordinates = None

    @property
    def spec(self):
        if self._spec is None:
            self._spec = load_topology(self.topology_path, self.truth_bindings)
        return self._spec

    def _initialise_static_layout(
        self, type_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if type_ids.ndim != 2 or not torch.equal(
            type_ids, type_ids[:1].expand_as(type_ids)
        ):
            raise ValueError(
                "Structured reconstruction requires one fixed token-source layout "
                "for every event; type-ID positions vary within the batch"
            )
        self._type_layout = type_ids[0].clone()
        source_positions = {}
        for source in {role.source for role in self.spec.visible_leaf_roles}:
            if source not in self.source_type_ids:
                raise KeyError(
                    f"Visible topology source {source} has no token type ID"
                )
            positions = (self._type_layout == self.source_type_ids[source]).nonzero(
                as_tuple=False
            ).flatten()
            if not positions.numel():
                raise ValueError(
                    f"Token layout contains no slots for visible source {source}"
                )
            source_positions[source] = positions
        self._templates, self._assignment_template = self._build_templates(
            source_positions, type_ids.shape[1]
        )
        return source_positions

    def _validate_static_layout(self, type_ids: torch.Tensor) -> None:
        if self._type_layout is None:
            self._initialise_static_layout(type_ids)
            return
        if type_ids.ndim != 2 or type_ids.shape[1] != len(self._type_layout):
            raise ValueError(
                "Token layout width changed after structured candidate templates were cached"
            )
        if not torch.equal(type_ids, self._type_layout.unsqueeze(0).expand_as(type_ids)):
            raise ValueError(
                "Token-source positions changed after structured candidate templates were cached"
            )

    def __call__(self, items: list[dict]) -> dict:
        # PyTorch calls Dataset.__getitems__ for batched map-style retrieval.
        # The cached dataset returns a tensor dictionary directly; uncached
        # compatibility datasets still pass a list of per-event examples.
        batch = items if isinstance(items, dict) else _collate_transformer(items)
        type_ids = batch["type_ids"]
        padding = batch["padding_mask"]
        raw = batch.get("reco_raw_labels")
        self._validate_static_layout(type_ids)
        visible_assignment_support = None
        if raw is None and self.visible_store is not None:
            source_labels, visible_assignment_support = self.visible_store.lookup(
                batch["event_index"]
            )
            raw = torch.full(
                type_ids.shape,
                float("nan"),
                dtype=torch.float32,
                device=type_ids.device,
            )
            for source_name, values in source_labels.items():
                if source_name not in self.source_type_ids:
                    raise KeyError(
                        f"Visible truth source {source_name!r} has no input source type"
                    )
                positions = (
                    self._type_layout == self.source_type_ids[source_name]
                ).nonzero(as_tuple=False).flatten().to(type_ids.device)
                if values.shape[1] != len(positions):
                    raise ValueError(
                        "Visible truth and measured token widths differ: "
                        f"source={source_name}, truth={values.shape[1]}, "
                        f"tokens={len(positions)}, sidecar={self.visible_store.path}"
                    )
                raw[:, positions] = values.to(device=raw.device)
        if raw is None and self.require_truth:
            raise ValueError(
                "Structured reconstruction requires raw visible labels or a validated "
                "visible-truth sidecar"
            )

        batch_size = len(batch["tokens"])
        truth_by_node = {}
        pools = {}
        for node_name in self.spec.dag_order:
            template = self._templates[node_name]
            ownership = template["leaf_ownership"]
            ownership_edges = template["ownership_edges"].to(device=padding.device)
            blocked = torch.zeros(
                (batch_size, len(ownership)), dtype=torch.int32, device=padding.device
            )
            if ownership_edges.numel():
                candidate_indices = ownership_edges[:, 0]
                token_indices = ownership_edges[:, 1]
                padded_owned = padding[:, token_indices].to(dtype=torch.int32)
                blocked.scatter_add_(
                    1,
                    candidate_indices.unsqueeze(0).expand(batch_size, -1),
                    padded_owned,
                )
            candidate_mask = blocked == 0
            candidate_truth = torch.zeros(
                (batch_size, len(ownership)), dtype=torch.bool, device=padding.device
            )
            if raw is not None:
                candidate_truth.fill_(True)
                for role in self.spec.nodes[node_name].leaf_roles:
                    if role.invisible:
                        continue
                    indices = template["leaf_indices"][role.key]
                    values = raw[:, indices.reshape(-1)].reshape(
                        batch_size, len(indices), role.multiplicity
                    )
                    expected = template["truth_values"][role.key].to(dtype=values.dtype)
                    candidate_truth &= (
                        values.sort(dim=2).values == expected
                    ).all(dim=2)
                for role in self.spec.nodes[node_name].composite_roles:
                    candidate_truth &= truth_by_node[role.child][
                        :, template["child_indices"][role.key]
                    ]
            candidate_truth &= candidate_mask
            truth_count = candidate_truth.sum(dim=1)
            if (truth_count > 1).any():
                event = int((truth_count > 1).nonzero(as_tuple=False)[0])
                raise RuntimeError(
                    f"Node {node_name} has multiple truth candidates; "
                    f"event={int(batch['event_index'][event])}"
                )
            reconstructable = truth_count == 1
            truth_index = candidate_truth.to(torch.int64).argmax(dim=1)
            truth_index = torch.where(
                reconstructable,
                truth_index,
                torch.full_like(truth_index, -1),
            )
            truth_by_node[node_name] = candidate_truth
            pools[node_name] = {
                "candidate_mask": candidate_mask,
                "leaf_ownership": ownership,
                "ownership_edges": ownership_edges,
                "leaf_indices": template["leaf_indices"],
                "child_indices": template["child_indices"],
                "truth_index": truth_index,
                "reconstructable": reconstructable,
            }
        structured = {"pools": pools}
        if visible_assignment_support is not None:
            roots = self.spec.roots
            if set(roots) != {"top", "antitop"}:
                raise ValueError(
                    "Charge-side visible support requires top and antitop roots; "
                    f"found {roots}"
                )
            computed_support = (
                pools["top"]["reconstructable"]
                & pools["antitop"]["reconstructable"]
            )
            observed_support = visible_assignment_support.to(computed_support.device)
            mismatch_mask = computed_support != observed_support
            if mismatch_mask.any():
                mismatch = int(mismatch_mask.nonzero(as_tuple=False)[0])
                raise ValueError(
                    "Visible assignment support disagrees with sidecar labels: "
                    f"EVENT_ID={int(batch['event_index'][mismatch])}, "
                    f"computed={bool(computed_support[mismatch])}, "
                    f"sidecar={bool(observed_support[mismatch])}, "
                    f"truth={self.visible_store.path}"
                )
            batch["visible_assignment_support"] = observed_support
        if self._assignment_template is not None:
            structured["assignments"] = self._assignment_template.batch(pools)
        batch["structured"] = structured
        if "physics_vectors" in batch:
            batch["met_xy"] = met_xy_from_batch(batch, self)
        elif self.invisible_store is not None:
            raise KeyError("Invisible truth collation requires physical MET vectors")
        if self.invisible_store is not None:
            invisible, invisible_support = self.invisible_store.lookup_with_support(
                batch["event_index"]
            )
            batch["invisible_truth"] = invisible
            batch["invisible_support"] = invisible_support
            batch["invisible_all_supported"] = bool(
                getattr(self.invisible_store, "all_supported", False)
            )
            state = self.invisible_coordinates.encode(invisible, batch["met_xy"])
            batch["invisible_target_state"] = state
            batch["invisible_truth_scaled"] = self.invisible_scaler.transform(state)
        if self.point_scaler is not None:
            batch["invisible_truth_point_scaled"] = self.point_scaler.transform(invisible)
            batch["invisible_point_support"] = torch.stack(
                [
                    structured["pools"][role.node]["reconstructable"]
                    for role in self.spec.invisible_roles
                ],
                dim=1,
            ) & invisible_support
        if self.parent_store is not None:
            batch["parent_truth"] = self.parent_store.lookup(batch["event_index"])
        if self.partonic_store is not None:
            values, support = self.partonic_store.lookup(batch["event_index"])
            batch["partonic_truth"] = values
            batch["partonic_support"] = support
            batch["partonic_truth_scaled"] = self.partonic_scaler.transform(values)
        return batch

    def _build_templates(
        self, source_positions: dict[str, torch.Tensor], token_count: int
    ) -> tuple[dict, object | None]:
        available = {
            source: tuple(range(len(positions)))
            for source, positions in source_positions.items()
        }
        records = build_event_candidates(self.spec, available)
        templates = {}
        for node_name in self.spec.dag_order:
            candidates = records[node_name]
            if not candidates:
                raise ValueError(
                    f"Topology node {node_name!r} has no candidates for the configured "
                    f"token layout; available source slots={available}"
                )
            ownership = torch.zeros(
                (len(candidates), token_count), dtype=torch.bool
            )
            leaf_indices = {
                role.key: torch.full(
                    (len(candidates), role.multiplicity), -1, dtype=torch.long
                )
                for role in self.spec.nodes[node_name].leaf_roles
                if not role.invisible
            }
            truth_values = {
                role.key: torch.tensor(
                    sorted(role.truth_indices), dtype=torch.float32
                ).reshape(1, 1, -1)
                for role in self.spec.nodes[node_name].leaf_roles
                if not role.invisible
            }
            child_indices = {
                role.key: torch.full((len(candidates),), -1, dtype=torch.long)
                for role in self.spec.nodes[node_name].composite_roles
            }
            for candidate in candidates:
                for role_key, local_indices in candidate.constituents:
                    source = self.spec.leaf_roles[role_key].source
                    leaf_indices[role_key][candidate.index] = source_positions[source][
                        torch.tensor(local_indices, dtype=torch.long)
                    ]
                for role_key, child_index in candidate.children:
                    child_indices[role_key][candidate.index] = child_index
                for source, local_index in candidate.leaf_ownership:
                    ownership[
                        candidate.index, source_positions[source][local_index]
                    ] = True
            templates[node_name] = {
                "leaf_ownership": ownership,
                "ownership_edges": ownership.nonzero(as_tuple=False),
                "leaf_indices": leaf_indices,
                "truth_values": truth_values,
                "child_indices": child_indices,
            }

        assignment_template = None
        if self.build_assignments:
            assignment_template = build_assignment_template(
                self.spec,
                records,
                max_states=self.max_assignment_states,
            )
        return templates, assignment_template


class StructuredCachedDataset(Dataset):
    """Read the validated visible cache without constructing obsolete M0 pools."""

    def __init__(self, payload: dict, positions):
        self.payload = payload
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = int(self.positions[item])
        result = {
            "tokens": self.payload["tokens"][position],
            "type_ids": self.payload["type_ids"][position],
            "padding_mask": self.payload["padding_mask"][position],
            "labels": self.payload["labels"][position],
            "event_index": self.payload.get("event_id", self.payload["indices"])[position],
            "reco_source_ids": self.payload["type_ids"][position].to(torch.int8),
            "physics_vectors": self.payload["physics_vectors"][position],
        }
        if "reco_raw_labels" in self.payload:
            result["reco_raw_labels"] = self.payload["reco_raw_labels"][position]
        return result

    def __getitems__(self, items):
        positions = torch.as_tensor(
            self.positions[np.asarray(items, dtype=np.int64)], dtype=torch.long
        )
        event_source = self.payload.get("event_id", self.payload["indices"])
        type_ids = self.payload["type_ids"].index_select(0, positions)
        result = {
            "tokens": self.payload["tokens"].index_select(0, positions),
            "type_ids": type_ids,
            "padding_mask": self.payload["padding_mask"].index_select(0, positions),
            "labels": self.payload["labels"].index_select(0, positions),
            "event_index": event_source.index_select(0, positions),
            "reco_source_ids": type_ids.to(torch.int8),
            "physics_vectors": self.payload["physics_vectors"].index_select(0, positions),
        }
        if "reco_raw_labels" in self.payload:
            result["reco_raw_labels"] = self.payload["reco_raw_labels"].index_select(0, positions)
        return result


class StructuredDataModule(TransformerDataModule):
    def __init__(
        self,
        config,
        topology_path: str | Path,
        *,
        truth_bindings: dict | None = None,
        tensor_cache: str | Path | None = None,
        invisible_path: str | Path | None = None,
        point_enabled: bool = False,
        parent_path: str | Path | None = None,
        parent_roles: tuple[str, ...] = (),
        partonic_path: str | Path | None = None,
        partonic_roles: tuple[str, ...] = (),
        partonic_truth_bindings: dict[str, str] | None = None,
        allow_synthetic: bool = False,
        max_events: dict | None = None,
        subset_seed: int = 42,
        build_assignments: bool = False,
        max_assignment_states: int | None = None,
        coordinates_config=None,
        mass_shell_config=None,
        require_truth: bool = True,
        visible_truth_path: str | Path | None = None,
    ):
        self.tensor_cache = (
            None if tensor_cache is None else Path(tensor_cache).expanduser().resolve()
        )
        super().__init__(config)
        self.structured_collator = StructuredCollator(
            topology_path,
            dict(self.schema.source_type_ids),
            truth_bindings=truth_bindings,
            build_assignments=build_assignments,
            max_assignment_states=max_assignment_states,
            require_truth=require_truth,
        )
        self.invisible_path = invisible_path
        self.visible_truth_path = (
            None
            if visible_truth_path is None
            else Path(visible_truth_path).expanduser().resolve()
        )
        self.point_enabled = bool(point_enabled)
        self.parent_path = parent_path
        self.parent_roles = tuple(str(role) for role in parent_roles)
        if self.parent_path is not None and not self.parent_roles:
            raise ValueError("Parent truth loading requires configured parent roles")
        self.partonic_path = partonic_path
        self.partonic_roles = tuple(str(role) for role in partonic_roles)
        self.partonic_truth_bindings = (
            None
            if partonic_truth_bindings is None
            else {str(key): str(value) for key, value in partonic_truth_bindings.items()}
        )
        if self.partonic_path is not None and not self.partonic_roles:
            raise ValueError("Partonic truth loading requires configured partonic roles")
        self.allow_synthetic = allow_synthetic
        self.max_events = max_events
        self.subset_seed = subset_seed
        self.partonic_scaler_fit_metadata = None
        self.invisible_scaler_fit_metadata = None
        self.invisible_scaler = None
        self.point_scaler = None
        self.parent_store = None
        role_names = tuple(
            role.key for role in self.structured_collator.spec.invisible_roles
        )
        if coordinates_config is None:
            coordinates_config = OmegaConf.create({"type": "cartesian"})
        self.invisible_coordinates = build_invisible_coordinates(
            coordinates_config, role_names, mass_shell_config
        )
        self.structured_collator.invisible_coordinates = self.invisible_coordinates
        self._joint_setup_done = False

    def setup(self, stage=None):
        super().setup(stage)
        if self._joint_setup_done:
            return
        if self.tensor_cache is None or self.cache_payload is not None:
            self._apply_event_limits_and_invisible_truth()
            self._joint_setup_done = True
            return
        if not self.tensor_cache.is_file():
            raise FileNotFoundError(
                f"Explicit tensor cache does not exist: {self.tensor_cache}"
            )
        manifest_path = self.tensor_cache.with_suffix(".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Explicit tensor cache has no manifest: {manifest_path}"
            )

        manifest = json.loads(manifest_path.read_text())
        expected = {
            "runtime_input_h5_path": str(Path(self.h5_path).expanduser().resolve()),
            "token_feature_dim": int(self.schema.dim),
            "token_features": list(self.schema.names),
            "physics_vectors_shape": [
                len(self.split_indices["selected"]),
                int(self._configured_total_tokens()),
                4,
            ],
            "physics_vector_representation": "E_px_py_pz_GeV_v1",
            "source_type_ids": dict(self.schema.source_type_ids),
            "widths": {
                key: int(value) for key, value in self.configured_widths.items()
            },
            "reconstruction": OmegaConf.to_container(
                self.reconstruction_cfg, resolve=True
            ),
        }
        observed = {key: manifest.get(key) for key in expected}
        observed["runtime_input_h5_path"] = str(
            Path(observed["runtime_input_h5_path"]).resolve()
        )
        if observed != expected:
            raise ValueError(
                "Explicit tensor cache manifest mismatch: "
                f"expected={expected}, observed={observed}"
            )
        payload = torch.load(
            self.tensor_cache, map_location="cpu", weights_only=False
        )
        required_payload = {
            "indices", "tokens", "type_ids", "padding_mask", "labels",
            "physics_vectors", "event_id",
        }
        if self.reconstruction_enabled:
            required_payload.add("reco_raw_labels")
        missing_payload = sorted(required_payload - set(payload))
        if missing_payload:
            raise ValueError(
                "Explicit tensor cache is missing the required physical-vector "
                f"payload fields {missing_payload}; rebuild the cache"
            )
        payload_indices = torch.as_tensor(payload["indices"], dtype=torch.long).numpy()
        selected = np.asarray(self.split_indices["selected"], dtype=np.int64)
        if not np.array_equal(payload_indices, selected):
            raise ValueError(
                "Explicit tensor cache event indices do not equal the stable "
                "selected split: "
                f"cache={len(payload_indices)}, selected={len(selected)}"
            )
        payload_event_ids = torch.as_tensor(payload["event_id"]).cpu().numpy()
        if payload_event_ids.ndim != 1 or len(payload_event_ids) != len(selected):
            raise ValueError(
                "Explicit tensor cache EVENT_ID has invalid shape: "
                f"expected=({len(selected)},), observed={payload_event_ids.shape}"
            )
        with h5py.File(self.h5_path, "r") as source:
            if "EVENT_ID" not in source:
                raise KeyError(
                    f"Explicit tensor cache requires canonical EVENT_ID values: {self.h5_path}"
                )
            canonical_event_ids = np.asarray(source["EVENT_ID"][selected])
        if not np.array_equal(payload_event_ids, canonical_event_ids):
            mismatch = int(np.flatnonzero(payload_event_ids != canonical_event_ids)[0])
            raise ValueError(
                "Explicit tensor cache EVENT_ID mapping differs from its canonical input: "
                f"cache_row={mismatch}, cache_event_id={int(payload_event_ids[mismatch])}, "
                f"canonical_event_id={int(canonical_event_ids[mismatch])}, input={self.h5_path}"
            )
        expected_shape = (len(selected), self._configured_total_tokens())
        vectors = payload["physics_vectors"]
        if tuple(vectors.shape) != expected_shape + (4,):
            raise ValueError(
                "Explicit tensor cache has invalid physics_vectors shape: "
                f"expected={expected_shape + (4,)}, observed={tuple(vectors.shape)}"
            )
        if not torch.isfinite(vectors).all():
            raise ValueError("Explicit tensor cache physics_vectors contain non-finite values")
        padding = payload["padding_mask"]
        if not torch.equal(vectors[padding], torch.zeros_like(vectors[padding])):
            raise ValueError(
                "Explicit tensor cache padded physics_vectors must be exactly zero"
            )

        def selected_positions(indices):
            indices = np.asarray(indices, dtype=np.int64)
            positions = np.searchsorted(selected, indices)
            if (
                (positions >= len(selected)).any()
                or not np.array_equal(selected[positions], indices)
            ):
                raise ValueError(
                    "Stable split contains an event absent from the explicit tensor cache"
                )
            return positions

        self.cache_payload = payload
        logger.info("CACHE HIT: validated explicit tensor cache %s", self.tensor_cache)
        self.train_dataset = StructuredCachedDataset(
            payload, selected_positions(self.split_indices["train"])
        )
        self.val_dataset = StructuredCachedDataset(
            payload, selected_positions(self.split_indices["validation"])
        )
        self.test_dataset = StructuredCachedDataset(
            payload, selected_positions(self.split_indices["test"])
        )
        self._apply_event_limits_and_invisible_truth()
        self._joint_setup_done = True

    def _selected_ids(self, split_name: str, positions):
        rows = np.asarray(self.split_indices[split_name], dtype=np.int64)[positions]
        if self.cache_payload is not None and "event_id" in self.cache_payload:
            selected = np.asarray(self.split_indices["selected"], dtype=np.int64)
            order = np.argsort(selected)
            locations = np.searchsorted(selected[order], rows)
            if (locations >= len(selected)).any() or not np.array_equal(
                selected[order][locations], rows
            ):
                raise ValueError("Split row is absent from cache EVENT_ID mapping")
            payload_positions = order[locations]
            return self.cache_payload["event_id"][payload_positions].numpy()
        with h5py.File(self.h5_path, "r") as source:
            return source["EVENT_ID"][rows] if "EVENT_ID" in source else rows

    def _apply_event_limits_and_invisible_truth(self):
        split_names = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        selected_positions = {}
        training_dataset = self.train_dataset
        for loader_name, split_name in split_names.items():
            dataset = getattr(self, f"{loader_name}_dataset")
            count = None if self.max_events is None else self.max_events.get(loader_name)
            positions = np.arange(len(dataset), dtype=np.int64)
            if count is not None and int(count) > len(positions):
                raise ValueError(
                    f"Configured data.max_events.{loader_name}={int(count)} exceeds "
                    f"the frozen {split_name} population of {len(positions)} events"
                )
            if count is not None and int(count) < len(positions):
                event_ids = self._selected_ids(split_name, positions)
                positions = deterministic_subset_positions(
                    event_ids,
                    int(count),
                    self.subset_seed,
                )
                setattr(
                    self,
                    f"{loader_name}_dataset",
                    Subset(dataset, positions.tolist()),
                )
            selected_positions[loader_name] = positions

        parent_store = None
        if self.parent_path is not None:
            parent_store = ParentTruthStore(self.parent_path, self.parent_roles)
        self.parent_store = parent_store
        self.structured_collator.parent_store = parent_store

        visible_store = None
        if self.visible_truth_path is not None and self.structured_collator.require_truth:
            source_names = tuple(
                sorted({role.source for role in self.structured_collator.spec.visible_leaf_roles})
            )
            visible_store = VisibleTruthStore(self.visible_truth_path, source_names)
            source_contract = (self.split_metadata or {}).get("source_contract", {})
            for key in (
                "source_manifest_sha256",
                "split_membership_sha256",
                "split_version",
                "max_jets",
            ):
                expected = source_contract.get(key)
                observed = visible_store.metadata.get(key)
                if expected is not None and expected != observed:
                    raise ValueError(
                        "Visible truth sidecar does not match measured-event source/split: "
                        f"key={key}, input={expected!r}, truth={observed!r}, "
                        f"input={self.h5_path}, truth={visible_store.path}"
                    )
        self.visible_store = visible_store
        self.structured_collator.visible_store = visible_store

        partonic_store = None
        if self.partonic_path is not None:
            partonic_store = PartonicTruthStore(
                self.partonic_path,
                self.partonic_roles,
                self.partonic_truth_bindings,
            )
            if partonic_store.metadata.get("partonic_truth_definition") not in (
                None,
                "partonic_after_fsr",
            ):
                raise ValueError(
                    "Partonic CFM requires genuine partonic_after_fsr parent targets; "
                    f"{partonic_store.path} declares "
                    f"{partonic_store.metadata['partonic_truth_definition']!r}"
                )
            source_contract = (self.split_metadata or {}).get("source_contract", {})
            for key in (
                "source_manifest_sha256",
                "split_membership_sha256",
                "split_version",
                "max_jets",
            ):
                expected = source_contract.get(key)
                observed = partonic_store.metadata.get(key)
                if expected is not None and expected != observed:
                    raise ValueError(
                        "Partonic truth sidecar does not match measured-event source/split: "
                        f"key={key}, input={expected!r}, truth={observed!r}, "
                        f"input={self.h5_path}, truth={partonic_store.path}"
                    )
            if partonic_store.synthetic:
                scaler_ids = self._selected_ids("train", selected_positions["train"])
                scaler_population = "selected_train_subset"
            else:
                full_train = np.arange(len(self.split_indices["train"]), dtype=np.int64)
                scaler_ids = self._selected_ids("train", full_train)
                scaler_population = "frozen_train_split"
            train_values, train_support = partonic_store.lookup(scaler_ids)
            self.partonic_scaler = PartonicScaler.fit(
                train_values,
                train_support,
                self.partonic_roles,
            )
            from joint_event_reconstruction.m0.data.splits import index_hash

            self.partonic_scaler_fit_metadata = {
                "population": scaler_population,
                "event_id_hash": index_hash(np.sort(np.asarray(scaler_ids, dtype=np.int64))),
                "event_count": len(scaler_ids),
            }
        else:
            self.partonic_scaler = None
            self.partonic_scaler_fit_metadata = None
        self.partonic_store = partonic_store
        self.structured_collator.partonic_store = partonic_store
        self.structured_collator.partonic_scaler = self.partonic_scaler

        if self.invisible_path is None:
            self.point_scaler = None
            self.structured_collator.point_scaler = None
            return

        role_names = tuple(
            role.key for role in self.structured_collator.spec.invisible_roles
        )
        if self.cache_payload is not None and "invisible_truth" in self.cache_payload:
            cached_ids = self.cache_payload["event_id"].numpy()
            cached_truth = self.cache_payload["invisible_truth"].numpy()
            if cached_truth.shape != (len(cached_ids), len(role_names), 3):
                raise ValueError(
                    "Cached invisible truth shape does not match topology roles: "
                    f"cache={cached_truth.shape}, roles={role_names}"
                )
            order = np.argsort(cached_ids)
            store = InvisibleTruthStore.__new__(InvisibleTruthStore)
            # A caller may provide an already materialised cache payload while
            # deliberately leaving tensor_cache unset (for example a tiny
            # real-data validation fixture).  Keep a concrete provenance path
            # for strict lookup errors instead of constructing Path(None).
            store.path = (
                Path(self.tensor_cache)
                if self.tensor_cache is not None
                else Path(self.h5_path).expanduser().resolve()
            )
            store.role_names = role_names
            store.synthetic = False
            store.metadata = {"source": "tensor_cache"}
            store.event_ids = cached_ids[order]
            cached_support = self.cache_payload.get("invisible_support")
            if cached_support is not None and (
                cached_support.dtype != torch.bool
                or tuple(cached_support.shape) != tuple(cached_truth.shape[:2])
            ):
                raise TypeError(
                    "Cached invisible_support must be boolean [N,R] aligned with targets"
                )
            support = (
                np.ones(cached_truth.shape[:2], dtype=np.bool_)
                if cached_support is None
                else cached_support.numpy().astype(np.bool_, copy=False)
            )
            if support.shape != cached_truth.shape[:2]:
                raise ValueError("Cached invisible support does not align with cached targets")
            support = support[order]
            values = cached_truth[order]
            if not np.isfinite(values[support]).all():
                raise ValueError("Cached supported invisible targets must be finite")
            store.values = np.where(support[..., None], values, 0.0).astype(
                np.float32, copy=False
            )
            store.support = support
            store.all_supported = bool(support.all())
            store.support_policy = (
                "explicit_role_support"
                if cached_support is not None
                else "legacy_all_roles_required"
            )
        else:
            store = InvisibleTruthStore(
                self.invisible_path,
                role_names,
                allow_synthetic=self.allow_synthetic,
            )
        if not store.synthetic:
            source_contract = (self.split_metadata or {}).get("source_contract", {})
            for key in (
                "source_manifest_sha256",
                "split_membership_sha256",
                "split_version",
                "max_jets",
            ):
                expected = source_contract.get(key)
                observed = store.metadata.get(key)
                if isinstance(observed, bytes):
                    observed = observed.decode("utf-8")
                if expected is not None and str(expected) != str(observed):
                    raise ValueError(
                        "Invisible truth sidecar does not match measured-event source/split: "
                        f"key={key}, input={expected!r}, truth={observed!r}, "
                        f"input={self.h5_path}, truth={store.path}"
                    )
        if store.synthetic:
            scaler_ids = self._selected_ids("train", selected_positions["train"])
        else:
            full_train = np.arange(
                len(self.split_indices["train"]), dtype=np.int64
            )
            scaler_ids = self._selected_ids("train", full_train)
        train_truth, train_role_support = store.lookup_with_support(scaler_ids)
        scaler_positions = (
            selected_positions["train"]
            if store.synthetic
            else np.arange(len(training_dataset), dtype=np.int64)
        )
        train_joint_support = train_role_support.all(dim=1)
        if not bool(train_joint_support.any()):
            supported_by_role = train_role_support.sum(dim=0).tolist()
            raise ValueError(
                "No training events have verified support for every joint invisible role; "
                f"sidecar={store.path}, role_counts={supported_by_role}, "
                f"roles={store.role_names}"
            )
        train_met = self._met_for_positions(training_dataset, scaler_positions)
        train_truth = train_truth[train_joint_support]
        train_met = train_met[train_joint_support]
        train_state = self.invisible_coordinates.encode(train_truth, train_met)
        self.invisible_scaler = InvisibleScaler.fit(
            train_state,
            self.invisible_coordinates.token_names,
            representation=(
                f"invisible_coordinate_state_"
                f"{self.invisible_coordinates.coordinate_type}_v1"
            ),
            active_mask=self.invisible_coordinates.active_mask,
        )
        from joint_event_reconstruction.m0.data.splits import index_hash

        self.invisible_scaler_fit_metadata = {
            "population": "selected_train_subset" if store.synthetic else "frozen_train_split",
            "event_id_hash": index_hash(np.sort(np.asarray(scaler_ids, dtype=np.int64))),
            "event_count": len(scaler_ids),
            "supported_event_count_by_role": train_role_support.sum(dim=0).tolist(),
            "joint_supported_event_count": int(train_joint_support.sum().item()),
        }
        if self.point_enabled:
            point_truth, point_support = store.lookup_with_support(scaler_ids)
            self.point_scaler = InvisibleScaler.fit(
                point_truth,
                role_names,
                representation="direct_point_cartesian_v1",
                support_mask=point_support,
            )
        else:
            self.point_scaler = None
        self.structured_collator.invisible_store = store
        self.structured_collator.invisible_scaler = self.invisible_scaler
        self.structured_collator.point_scaler = self.point_scaler

    def _met_for_positions(self, dataset, positions) -> torch.Tensor:
        positions = np.asarray(positions, dtype=np.int64)
        payload = getattr(dataset, "payload", None)
        if payload is None:
            payload = getattr(dataset, "cache_payload", None)
        dataset_positions = getattr(dataset, "positions", None)
        if payload is not None and dataset_positions is not None:
            rows = np.asarray(dataset_positions, dtype=np.int64)[positions]
            vectors = payload["physics_vectors"][torch.as_tensor(rows)]
            type_ids = payload["type_ids"][torch.as_tensor(rows)]
            padding = payload["padding_mask"][torch.as_tensor(rows)]
            return met_xy_from_batch(
                {
                    "physics_vectors": vectors,
                    "type_ids": type_ids,
                    "padding_mask": padding,
                },
                self.schema,
            )

        # The non-cached HDF5 dataset is used for the canonical real samples.
        # Reading one structured row at a time makes train-only coordinate
        # scaler fitting needlessly expensive at multi-million-event scale.
        # Read only the MET rows in one batch while retaining the exact dataset
        # ordering and the same physical (px, py) convention as the collator.
        event_indices = getattr(dataset, "indices", None)
        h5_path = getattr(dataset, "h5_path", None)
        if event_indices is not None and h5_path is not None and "MET" in self.schema.sources:
            event_indices = np.asarray(event_indices, dtype=np.int64)[positions]
            with h5py.File(h5_path, "r") as source:
                path = "/INPUTS/MET"
                if path not in source:
                    raise KeyError(f"MET source is missing from {h5_path}: {path}")
                if event_indices.size:
                    # h5py requires increasing, unique fancy indices.  Split
                    # positions are normally ordered, but subset selection is
                    # deliberately free to permute them, so restore the exact
                    # requested order after one sorted read.
                    order = np.argsort(event_indices, kind="stable")
                    sorted_indices = event_indices[order]
                    unique_indices, inverse = np.unique(
                        sorted_indices, return_inverse=True
                    )
                    rows_sorted = source[path][unique_indices]
                    rows_in_sorted_order = rows_sorted[inverse]
                    rows = np.empty_like(rows_in_sorted_order)
                    rows[order] = rows_in_sorted_order
                else:
                    rows = np.empty((0,), dtype=source[path].dtype)
            if rows.dtype.names is None:
                raise TypeError(f"MET source must have named fields: {h5_path}:{path}")
            names = set(rows.dtype.names)
            if {"px", "py"}.issubset(names):
                met = np.stack(
                    (np.asarray(rows["px"]).reshape(-1),
                     np.asarray(rows["py"]).reshape(-1)),
                    axis=-1,
                )
            else:
                pt_name = "pt" if "pt" in names else "met" if "met" in names else None
                if pt_name is None or "phi" not in names:
                    raise ValueError(
                        f"MET source lacks a physical (px,py) or (pt,phi) pair: "
                        f"{h5_path}:{path} fields={sorted(names)}"
                    )
                pt = np.asarray(rows[pt_name], dtype=np.float64).reshape(-1)
                phi = np.asarray(rows["phi"], dtype=np.float64).reshape(-1)
                met = np.stack((pt * np.cos(phi), pt * np.sin(phi)), axis=-1)
            met = np.asarray(met, dtype=np.float32)
            if met.shape != (len(positions), 2) or not np.isfinite(met).all():
                raise ValueError(
                    f"MET source contains invalid physical px/py values: "
                    f"{h5_path}:{path} rows={len(positions)}"
                )
            return torch.from_numpy(met)

        values = []
        for position in positions:
            item = dataset[int(position)]
            values.append(
                met_xy_from_batch(
                    {
                        "physics_vectors": item["physics_vectors"].unsqueeze(0),
                        "type_ids": item["type_ids"].unsqueeze(0),
                        "padding_mask": item["padding_mask"].unsqueeze(0),
                    },
                    self.schema,
                )[0]
            )
        return torch.stack(values)

    def _loader(self, dataset, split: str, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.config[f"{split}_batch_size"]),
            shuffle=shuffle,
            collate_fn=self.structured_collator,
            **self._dataloader_kwargs(split),
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, "train", True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, "val", False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, "test", False)
