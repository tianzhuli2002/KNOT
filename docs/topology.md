# Topology format

Topology YAML is the authoritative description of decay structure. Model code should not
contain process-name branches: a new process should normally require a topology file, a
visible-data configuration, and truth in the declared semantic order.

A topology contains one required `topology` mapping and may contain one optional
`physics` mapping. Each decay node contains named roles:

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

## Grammar

- `[SOURCE, TRUTH_INDEX]` is one leaf.
- Several leaves under the same named role are exchangeable. In the example,
  `Whad.q[0] <-> Whad.q[1]` exactly; differently named roles are never exchangeable.
- A string value is a reference to another decay node. References define the hierarchy;
  declaration order is irrelevant.
- `NEUTRINO` denotes invisible truth. The invisible role order is compiled from the
  topology and is preserved in `[B,Rnu,3] = [px,py,pz]` targets.
- MET is observed detector context, not a decay daughter.

The compiler rejects malformed leaves, duplicate semantic truth ownership, undefined
references and cycles. It derives:

- DAG order, roots, parents, children and descendants;
- visible and invisible semantic roles;
- hierarchy-preserving candidate pools;
- exact exchange symmetry for repeated roles;
- visible-object ownership for global compatibility;
- recurrent topology tokens;
- stable topology signatures stored in checkpoints and prediction files.

The optional physics block is deliberately small. It declares named four-vector sums and
compatibility observables, without introducing an expression language:

```yaml
physics:
  composites:
    W_e: {sum: [top_e.lepton, top_e.nu]}
    W_mu: {sum: [top_mu.lepton, top_mu.nu]}
    ttbar: {sum: [top_e, top_mu]}
  observables:
    W_e_mass: {type: invariant_mass, object: W_e, reference: 80.377}
    top_e_mass: {type: invariant_mass, object: top_e, reference: 172.5}
    top_mass_balance:
      {type: mass_difference, objects: [top_e, top_mu], reference: 0.0}
    met_closure:
      {type: transverse_sum_residual, objects: [top_e.nu, top_mu.nu], measurement: MET}
```

The supported observable types are `invariant_mass`, `mass_difference`, and
`transverse_sum_residual`. Four-vector addition and configured invisible mass shells are
physical kinematics; W/top proximity and MET closure are detector-level compatibility
residuals and are never imposed as exact constraints. The references above are the
explicit nominal ttbar2L convention used here; they are not part of the historical MC
topology identity.

The compiler keeps the historical `TopologySpec.signature` restricted to topology and
candidate semantics. An optional `PhysicsSpec.signature` records the physics block
separately, so old checkpoints and topology-only files retain their identity.

For the first reconstruction paper, one stored unconditioned invisible posterior is
broadcast over the visible top-K hypotheses and evaluated with the compiled topology
kinematics. The resulting residual summaries can drive an interpretable analytic
compatibility score or a small learned residual correction; this score is an
unnormalised compatibility/evidence-like quantity, not an event likelihood. Physics
feature scales are fitted on the training split using the correct assignment and truth
invisible state, then held fixed for validation studies. Alpha scans and scorer ablations
are offline validation operations and do not regenerate the posterior.

The paper-critical order is generator objective/inference budget, invisible coordinates,
invisible generator backbone, then the invisible-to-assignment coupling study. The
validation-only CFM inference scan selected the eight-NFE-equivalent Heun-4 point as the
current low-cost plateau setting; this is an inference budget choice, not a new objective
gate. A reverse
assignment-conditioned invisible pass is a later, slim study and is run only if the
forward coupling has measurable oracle headroom. The older topology-token feedback,
native-joint-flow, and recurrent co-refinement studies remain deferred follow-up work.

Hierarchical candidates reuse compiled child candidates. A parent top containing a W uses
the W candidate representation rather than rebuilding an independent flat leaf tuple.

## Reconstructability and NULL

A node is reconstructable when its declared truth candidate can be represented by the
selected model inputs. Physical `NULL` means that a branch is not reconstructed from a
physical candidate in that event.

Raw visible label zero means a real unassigned object. NaN is allowed only for a padded or
non-existent object. Truth beyond a configured object cap remains physically present but
makes the affected branch unreconstructable from the model input.

`NULL` and the discrete-flow `MASK` state have different meanings:

```text
NULL  physical clean assignment choice / unreconstructed branch
MASK  transient unresolved state used only during discrete Flow Matching
```

`MASK` is never part of the final physical assignment support.

## Assignment support semantics

The repository deliberately records two different complete-assignment supports.

### Discriminative factorized decoding

The local node classifiers contain a NULL class, but the exact global top-K decoder uses

```text
complete_non_null
```

support: every root selects a physical candidate and the chosen roots are globally
compatible. NULL remains meaningful locally but is excluded from the decoded complete
prediction support.

### DFM and joint DFM+CFM

The clean categorical state space uses

```text
complete_null_including
```

support: every globally feasible clean tuple is represented, including physical NULL
choices. The extra MASK state exists only during the generative trajectory.

These supports are not interchangeable denominators. Prediction HDF5 files persist the
support definition, evaluators reproduce it in `metrics.json`, and comparison plotting
separates assignment accuracies by support.

## Complete DFM assignment space

Assignment-only DFM and joint flow compile the clean categorical state space from root
candidate pools. Globally incompatible visible-object reuse is removed exactly.
Exchangeable repeated daughters have already been quotiented in candidate construction,
so the state space does not create duplicate labels for physically identical permutations.

The complete state tensor is materialised only when

```yaml
model:
  assignment:
    formulation: flow_matching
```

or when

```yaml
model:
  mode: joint_flow
```

Ordinary discriminative M1/M2/M3, diffusion and factorized CFM runs therefore do not pay
this complete-state tensor cost.

The safety limit is

```yaml
data:
  max_assignment_states: 100000
```

It is a memory guard, not a learned architecture parameter and not part of checkpoint
identity. If the compiled topology exceeds it, inspect candidate counts before increasing
the limit. Exact complete-state DFM is retained while practical because it provides a
clean globally constrained baseline.

## Invisible-role genericity

The model is generic in the number of ordered invisible roles `Rnu`.

Production examples currently include one-neutrino semileptonic and two-neutrino
dileptonic topologies. CI additionally uses `configs/topology/synthetic_multinu.yaml`,
which contains three distinct invisible roles (`Rnu=3`) together with a three-member
exchangeable visible role and multiple roots. This fixture exists specifically to catch
hard-coded assumptions about one or two neutrinos.

The `Rnu=3` fixture is a software/genericity contract, not a physics benchmark.

## Adding a topology

A production topology should require only:

1. one topology YAML;
2. a visible dataset configuration exposing every named visible source;
3. raw truth labels with the declared semantic indices;
4. invisible truth roles in the declared order when invisible inference is enabled.

Useful checks before training are:

```bash
python scripts/inspect_topology.py --topology configs/topology/<name>.yaml
```

and, for maintained experiment products,

```bash
python -m joint_event_reconstruction.cli.run validate-data \
  --experiment configs/experiments/<name>.yaml
```

`configs/topology/synthetic_contract.yaml` and `synthetic_multinu.yaml` are CI fixtures
that prove hierarchy, repeated-role symmetry, multi-root assignment construction and
invisible-role genericity through the same production code paths.
