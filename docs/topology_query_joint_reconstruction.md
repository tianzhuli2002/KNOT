# Topology-query joint reconstruction

This extension keeps the maintained L-GATr visible encoder, structured
assignment head, and unconditioned MET-residual CFM generator.  When enabled,
`TopologyQueryBlock` consumes the encoder's scalar event context and scalar
object tokens:

```text
pooled [B,D], tokens [B,N,D], padding [B,N]
        -> query states [B,Q,D], query context [B,D]
        -> existing assignment/CFM context and optional task projections
```

Queries are declared by name, compiled-node metadata, kind, and related query
names.  `full`, `topology`, and `topology_plus_global` interaction policies are
available.  The block is deliberately scalar: it is not claimed to be a
Lorentz-equivariant query transformer.  The surrounding L-GATr representation
and its geometric channels are not mixed with arbitrary query operations.

The partonic head selects configured query roles and predicts scaled
`(px,py,pz,log1p(m))` features.  The train-fitted scaler reconstructs a
physical `(E,px,py,pz)` vector with event-dependent positive invariant mass;
no nominal mass or reconstructed constituent is used.  Its input is measured
data and learned query state only.  It does not accept truth assignments,
truth neutrinos, or truth parent vectors.

The sidecar contract is an HDF5 file with `EVENT_ID`,
`PARTONIC/<configured role>` float32 physical four-vectors,
`PARTONIC/SUPPORT` boolean masks, and `partonic_roles` plus
`partonic_truth_bindings` attributes.  The current 30_4 production data has no
such hard-process truth and therefore keeps this head disabled.  The existing
`hybrid_visible_reconstructed_plus_truth_invisible_v1` parent target remains a
separate optional head and is not reinterpreted here.

The versioned synthetic contract fixture and release checks are run with:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  python scripts/data/build_topology_query_synthetic.py \
  --output results/experiments/topology_query_joint_20260923/synthetic_data/measured_v2.h5 \
  --partonic-output results/experiments/topology_query_joint_20260923/synthetic_data/partonic_truth_v2.h5

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  python scripts/validate_topology_query_contract.py \
  --experiment results/experiments/topology_query_joint_20260923/configs/topology_query_full.public.yaml \
  --output results/experiments/topology_query_joint_20260923/validation/topology_query_contract_cpu.json
```

Use the public configuration in
`results/experiments/topology_query_joint_20260923/configs/` for the exact
development fixture and resolved paths.  Disable both `model.topology_queries`
and `model.heads.partonic` to recover the existing architecture; disabled
partonic identity fields are omitted so old point/parent checkpoints retain
their historical architecture signatures.

For deployment, keep the partonic query head enabled but remove its training
sidecar and loss from the resolved public configuration. The checkpoint
restores the fitted target scaler; no target tensor is required by the model
input or prediction path:

```bash
python -m joint_event_reconstruction.cli.run predict \
  --experiment configs/experiments/topology_query_synthetic_full.public.yaml \
  --checkpoint <partonic-checkpoint> --output <prediction.h5> \
  --set model.heads.partonic.target_path=null \
  --set 'training.trainable=[visible,assignment,invisible,queries]' \
  --set training.loss_weights.partonic=0
```

The partonic role names and truth-binding semantics remain in the checkpoint
identity; the sidecar filesystem path does not. Training with the partonic
loss still fails early when the explicit sidecar is absent.
