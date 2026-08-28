# Topology format

A topology file contains one `topology` mapping. Each decay node contains named roles.
The format is intentionally small: visible leaves, repeated exchangeable leaves,
composite references and invisible leaves are enough to compile the reconstruction graph.

```yaml
topology:
  Whad:
    q:
      - [JET, 3]
      - [JET, 4]
  thad:
    W: Whad
    b: [JET, 2]
  tlep:
    b: [JET, 1]
    lepton: [LEPTON, 1]
    nu: [NEUTRINO, 0]
```

- `[SOURCE, TRUTH_INDEX]` is one leaf.
- A list of leaves under the same named role is one exchangeable group. Thus
  `Whad.q[0] <-> Whad.q[1]`; roles with different names are never exchangeable.
- A string is a reference to another decay node. References infer the hierarchy;
  declaration order has no meaning.
- `NEUTRINO` is invisible truth. MET is observed detector context and is never a
  decay leaf.

The compiler rejects malformed leaves, duplicate truth ownership, undefined references
and cycles. It derives:

- DAG order, roots, parents, children and descendants;
- visible and invisible semantic roles;
- hierarchy-preserving candidate pools;
- exact repeated-role exchange symmetries;
- leaf ownership used for global assignment feasibility;
- recurrent topology tokens;
- topology signatures used by checkpoints and prediction files.

## Complete assignment space

Discrete Flow Matching and joint flow additionally compile a complete assignment space
from the root candidate pools. A clean state is one globally feasible tuple of root
choices. Each root may select one candidate or its physical `NULL` state. States that
reuse the same visible object across incompatible roots are removed exactly.

The transient DFM `MASK` state is separate from physical `NULL`:

```text
NULL  = this physical branch is not reconstructable from the selected input objects
MASK  = the discrete generative process has not yet resolved the assignment
```

Repeated exchangeable daughters are already quotient-ed in candidate construction, so
the complete state space does not reintroduce two labels for the same physical solution.

Complete assignment tensors are materialised only for `reconstruction.formulation =
flow_matching` or `model.mode = joint_flow`. Ordinary discriminative/EDM/CFM baselines
therefore do not pay the combinatorial state-space cost. The experiment setting
`model.reconstruction.flow_matching.max_states` is a memory/safety guard and is not part
of checkpoint architecture identity.

For a new topology, inspect the resulting candidate/state counts before raising that
limit. Exact complete-state DFM is deliberately retained while it is practical because
it gives a particularly clean globally constrained baseline.

## Adding a topology

A production topology should require only:

1. one topology YAML;
2. a visible dataset configuration that exposes every named visible source;
3. raw truth labels with the declared semantic indices;
4. invisible truth roles in the declared order when invisible inference is enabled.

No model file should branch on the topology name. `configs/topology/synthetic_contract.yaml`
is kept specifically as a non-ttbar CI fixture to exercise hierarchy, repeated-role
symmetry, an invisible role and multiple roots through the same maintained code path.

Raw visible label zero means a real unassigned object. NaN is permitted only for a
padded/nonexistent object. Truth beyond the selected object cap is physically present
but makes the affected branch unreconstructable from the model input.
