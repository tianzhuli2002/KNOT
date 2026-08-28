"""Add topology-derived hierarchical candidate tensors to visible-event batches."""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import (
    TransformerDataModule,
    _collate_transformer,
)
from joint_event_reconstruction.data.invisible import (
    InvisibleScaler,
    InvisibleTruthStore,
)
from joint_event_reconstruction.topology.assignments import build_assignment_template
from joint_event_reconstruction.topology.candidates import build_event_candidates
from joint_event_reconstruction.topology.compiler import load_topology


class StructuredCollator:
    def __init__(
        self,
        topology_path: str | Path,
        source_type_ids: dict[str, int],
        *,
        build_assignments: bool = False,
        max_assignment_states: int | None = None,
    ):
        self.topology_path = str(Path(topology_path).expanduser().resolve())
        self.source_type_ids = dict(source_type_ids)
        self.build_assignments = bool(build_assignments)
        self.max_assignment_states = (
            None if max_assignment_states is None else int(max_assignment_states)
        )
        if self.max_assignment_states is not None and self.max_assignment_states <= 0:
            raise ValueError("max_assignment_states must be positive or null")
        self._spec = None
        self._templates = None
        self._assignment_template = None
        self._source_positions = None
        self.invisible_store = None
        self.invisible_scaler = None

    @property
    def spec(self):
        if self._spec is None:
            self._spec = load_topology(self.topology_path)
        return self._spec

    def _initialise_static_layout(
        self, type_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        source_positions = {}
        for source in {role.source for role in self.spec.visible_leaf_roles}:
            if source not in self.source_type_ids:
                raise KeyError(
                    f"Visible topology source {source} has no token type ID"
                )
            expected = type_ids == self.source_type_ids[source]
            counts = expected.sum(dim=1)
            if not torch.equal(counts, counts[:1].expand_as(counts)):
                raise ValueError(
                    f"Token layout for source {source} varies within a batch"
                )
            positions = expected[0].nonzero(as_tuple=False).flatten()
            if not positions.numel():
                raise ValueError(
                    f"Token layout contains no slots for visible source {source}"
                )
            source_positions[source] = positions
        self._source_positions = source_positions
        self._templates, self._assignment_template = self._build_templates(
            source_positions, type_ids.shape[1]
        )
        return source_positions

    def __call__(self, items: list[dict]) -> dict:
        batch = _collate_transformer(items)
        if "reco_raw_labels" not in batch:
            raise ValueError(
                "Structured reconstruction requires raw visible reconstruction labels"
            )
        type_ids = batch["type_ids"]
        padding = batch["padding_mask"]
        raw = batch["reco_raw_labels"]
        if self._templates is None:
            self._initialise_static_layout(type_ids)

        batch_size = len(items)
        truth_by_node = {}
        pools = {}
        for node_name in self.spec.dag_order:
            template = self._templates[node_name]
            ownership = template["leaf_ownership"]
            candidate_mask = ~(
                padding.unsqueeze(1) & ownership.unsqueeze(0)
            ).any(dim=2)
            candidate_truth = torch.ones(
                (batch_size, len(ownership)), dtype=torch.bool
            )
            for role in self.spec.nodes[node_name].leaf_roles:
                if role.invisible:
                    continue
                indices = template["leaf_indices"][role.key]
                values = raw[:, indices.reshape(-1)].reshape(
                    batch_size, len(indices), role.multiplicity
                )
                expected = template["truth_values"][role.key].to(
                    dtype=values.dtype
                )
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
                "leaf_indices": template["leaf_indices"],
                "child_indices": template["child_indices"],
                "truth_index": truth_index,
                "reconstructable": reconstructable,
            }
        structured = {"pools": pools}
        if self._assignment_template is not None:
            structured["assignments"] = self._assignment_template.batch(pools)
        batch["structured"] = structured
        if self.invisible_store is not None:
            invisible = self.invisible_store.lookup(batch["event_index"])
            batch["invisible_truth"] = invisible
            batch["invisible_truth_scaled"] = self.invisible_scaler.transform(
                invisible
            )
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
            ownership = torch.zeros(
                (len(candidates), token_count), dtype=torch.bool
            )
            leaf_indices = {
                role.key: torch.full(
                    (len(candidates), role.multiplicity),
                    -1,
                    dtype=torch.long,
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
                role.key: torch.full(
                    (len(candidates),), -1, dtype=torch.long
                )
                for role in self.spec.nodes[node_name].composite_roles
            }
            for candidate in candidates:
                for role_key, local_indices in candidate.constituents:
                    source = self.spec.leaf_roles[role_key].source
                    leaf_indices[role_key][candidate.index] = source_positions[
                        source
                    ][torch.tensor(local_indices, dtype=torch.long)]
                for role_key, child_index in candidate.children:
                    child_indices[role_key][candidate.index] = child_index
                for source, local_index in candidate.leaf_ownership:
                    ownership[
                        candidate.index, source_positions[source][local_index]
                    ] = True
            templates[node_name] = {
                "leaf_ownership": ownership,
                "leaf_indices": leaf_indices,
                "truth_values": truth_values,
                "child_indices": child_indices,
            }

        assignment_template = None
        if self.build_assignments:
            assignment_template = build_assignment_template(self.spec, records)
            if (
                self.max_assignment_states is not None
                and assignment_template.state_count > self.max_assignment_states
            ):
                root_sizes = {
                    root: len(records[root]) for root in self.spec.roots
                }
                raise RuntimeError(
                    "Complete assignment state space is too large for configured "
                    "DFM/joint flow: "
                    f"states={assignment_template.state_count}, "
                    f"max_states={self.max_assignment_states}, roots={root_sizes}, "
                    f"topology={self.topology_path}. Increase "
                    "model.reconstruction.flow_matching.max_states only after "
                    "checking memory scaling."
                )
        return templates, assignment_template


class StructuredCachedDataset(Dataset):
    """Read the validated visible cache without constructing obsolete M0 pools."""

    def __init__(self, payload: dict, positions):
        import numpy as np

        self.payload = payload
        self.positions = np.asarray(positions, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = int(self.positions[item])
        return {
            "tokens": self.payload["tokens"][position],
            "type_ids": self.payload["type_ids"][position],
            "padding_mask": self.payload["padding_mask"][position],
            "labels": self.payload["labels"][position],
            "event_index": self.payload["indices"][position],
            "reco_raw_labels": self.payload["reco_raw_labels"][position],
            "reco_source_ids": self.payload["type_ids"][position].to(torch.int8),
        }


class StructuredDataModule(TransformerDataModule):
    def __init__(
        self,
        config,
        topology_path: str | Path,
        *,
        tensor_cache: str | Path | None = None,
        invisible_path: str | Path | None = None,
        allow_synthetic: bool = False,
        max_events: dict | None = None,
        subset_seed: int = 42,
        build_assignments: bool = False,
        max_assignment_states: int | None = None,
    ):
        self.tensor_cache = (
            None
            if tensor_cache is None
            else Path(tensor_cache).expanduser().resolve()
        )
        config.dataset.cache.enabled = False
        super().__init__(config)
        self.structured_collator = StructuredCollator(
            topology_path,
            dict(self.schema.source_type_ids),
            build_assignments=build_assignments,
            max_assignment_states=max_assignment_states,
        )
        self.invisible_path = invisible_path
        self.allow_synthetic = allow_synthetic
        self.max_events = max_events
        self.subset_seed = subset_seed
        self.invisible_scaler = None
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
        import json
        import numpy as np

        manifest = json.loads(manifest_path.read_text())
        expected = {
            "runtime_input_h5_path": str(
                Path(self.h5_path).expanduser().resolve()
            ),
            "token_feature_dim": int(self.schema.dim),
            "token_features": list(self.schema.names),
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
        payload_indices = torch.as_tensor(
            payload.get("indices"), dtype=torch.long
        ).numpy()
        selected = __import__("numpy").asarray(
            self.split_indices["selected"], dtype=__import__("numpy").int64
        )
        if not __import__("numpy").array_equal(payload_indices, selected):
            raise ValueError(
                "Explicit tensor cache event indices do not equal the stable "
                "selected split: "
                f"cache={len(payload_indices)}, selected={len(selected)}"
            )

        def selected_positions(indices):
            indices = __import__("numpy").asarray(
                indices, dtype=__import__("numpy").int64
            )
            positions = __import__("numpy").searchsorted(selected, indices)
            if (
                (positions >= len(selected)).any()
                or not __import__("numpy").array_equal(
                    selected[positions], indices
                )
            ):
                raise ValueError(
                    "Stable split contains an event absent from the explicit "
                    "tensor cache"
                )
            return positions

        self.cache_payload = payload
        self.train_dataset = StructuredCachedDataset(
            payload,
            selected_positions(self.split_indices["train"]),
        )
        self.val_dataset = StructuredCachedDataset(
            payload,
            selected_positions(self.split_indices["validation"]),
        )
        self.test_dataset = StructuredCachedDataset(
            payload,
            selected_positions(self.split_indices["test"]),
        )
        self._apply_event_limits_and_invisible_truth()
        self._joint_setup_done = True

    def _selected_ids(self, split_name: str, positions):
        import numpy as np

        rows = np.asarray(
            self.split_indices[split_name], dtype=np.int64
        )[positions]
        with h5py.File(self.h5_path, "r") as source:
            return source["EVENT_ID"][rows] if "EVENT_ID" in source else rows

    def _apply_event_limits_and_invisible_truth(self):
        import numpy as np
        from torch.utils.data import Subset

        split_names = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        selected_positions = {}
        for offset, (loader_name, split_name) in enumerate(split_names.items()):
            dataset = getattr(self, f"{loader_name}_dataset")
            count = (
                None
                if self.max_events is None
                else self.max_events.get(loader_name)
            )
            positions = np.arange(len(dataset), dtype=np.int64)
            if count is not None and int(count) < len(positions):
                rng = np.random.default_rng(self.subset_seed + offset)
                positions = np.sort(
                    rng.choice(positions, size=int(count), replace=False)
                )
                setattr(
                    self,
                    f"{loader_name}_dataset",
                    Subset(dataset, positions.tolist()),
                )
            selected_positions[loader_name] = positions

        if self.invisible_path is None:
            return

        role_names = tuple(
            role.key for role in self.structured_collator.spec.invisible_roles
        )
        store = InvisibleTruthStore(
            self.invisible_path,
            role_names,
            allow_synthetic=self.allow_synthetic,
        )
        if store.synthetic:
            scaler_ids = self._selected_ids(
                "train", selected_positions["train"]
            )
        else:
            full_train = np.arange(
                len(self.split_indices["train"]), dtype=np.int64
            )
            scaler_ids = self._selected_ids("train", full_train)
        train_truth = store.lookup(scaler_ids)
        self.invisible_scaler = InvisibleScaler.fit(
            train_truth,
            role_names,
        )
        self.structured_collator.invisible_store = store
        self.structured_collator.invisible_scaler = self.invisible_scaler

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
