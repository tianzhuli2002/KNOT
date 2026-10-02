# KNOT scientific closure: source and semantic audit

This audit describes the implementation at `57a078b` and its follow-up branch.
It is source-level evidence only; synthetic closure and real-data checks remain
separate gates.

## Source lineage

The KNOT baseline is based on `ec54f6b`. The following relevant commits are
ancestors of that baseline and therefore already supply the maintained path:

| Commit | Classification | KNOT relevance |
| --- | --- | --- |
| `f009ad6`, `87859cf` | Scientific implementation | Shared topology queries, structured assignment, and parent conditional-flow support used by M/C/J. |
| `dcd98e8` | Runtime implementation | Avoids repeated measured-event work in the maintained flow path. KNOT separately encodes once before posterior sampling. |
| `eaeb872`, `cb2b604` | Training/scaler correctness | Train-fitted scaler identity and initialization provenance; KNOT receives train-fitted invisible and parent scalers from the data module. |
| `0527c78`, `875d449` | Evaluation/training contract | Explicit parent-context and joint-context conventions. |
| `ec54f6b` | Evaluation-only audit | Conditional-skill and source-rotation diagnostics; no KNOT model change. |

The sibling snapshot at `4a4e0ce` adds an optional geometric parent CFM and
H_geom transport. This changes the parent-flow architecture and belongs to the
parent-geometry comparison, so it is not folded into the conventional P0/P2
KNOT comparison. Its solver-context preparation mechanism is already present
in the KNOT baseline. `e064fad` changes the RNG device used by a separate PCFM
validation path and its Slurm wrapper; it does not change model equations or
KNOT's CPU path. The VyPER recovery branch changes production launch/cache
plumbing and descends from a separate stable line; no physics correction from
it applies to KNOT. No later correction to KNOT's measured encoding, topology
query, invisible coordinate, parent feature, or sample-identity contract was
found that could be applied without changing the intended M/C/J models.

No later branch was merged or cherry-picked.

## Implemented probability semantics

`A` is a feasible complete structured assignment state sampled from the
maintained discriminative assignment distribution. `N` is the configured
invisible-role state in the maintained MET-residual coordinates. `P` contains
only configured parent-root states in scaled
`(px, py, pz, log1p(m))` coordinates; the parent-pair observables are derived
from those roots after sampling.

| Variant | Assignment conditioning | Invisible conditioning | Parent conditioning | Continuous latent state | Training conditioning | Prediction conditioning | Shared components | Variant-specific components |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | `q(A|x)` | `q(N|x)` | `q(P|x)` (P0) | N and P are separately generated; no joint continuous state | `training_conditioning` is stored but does not affect marginal heads; assignment loss uses its ordinary supervised label | Draw A, N, and P separately from event-level marginals | Measured L-GATr, topology queries, structured assignment, invisible codec/scaler | Invisible marginal CFM and P0 parent CFM |
| C | `q(A|x)` | `q(N|A,x)` | `q(P|A,N,x)` (P2) | N then one joint top/antitop parent state P | Default `truth`: N loss uses `A_truth`; P loss uses `A_truth,N_truth`. Optional `predicted` samples A and then N before the P loss; no gradients pass through either sample | For each sample index k: draw `A[k]`, draw `N[k]|A[k],x`, then draw `P[k]|A[k],N[k],x` | Measured L-GATr, topology queries, structured assignment, invisible codec/scaler | Hard assignment contexts for N and P, P2 parent CFM |
| J | `q(A|x)` | Joint with P | Joint with N under `q(N,P|A,x)` | One CFM state formed by active scaled N coordinates concatenated with scaled P-root coordinates | Default `truth`: one joint CFM loss on `(N_truth,P_truth)` under `A_truth`; no truth N is passed as a conditioner. Optional `predicted` samples A for the joint loss | Draw A, then draw one joint `(N,P)` sample under that same A | Measured L-GATr, topology queries, structured assignment, invisible codec/scaler | One shared-attention composite CFM over all active N and P role tokens |

For M, equal output sample indices are export slots only and do not encode
learned dependence. M's N/P outputs are independent marginal draws. For C,
the same event-local sample index is used throughout the ancestral chain. For
J, one denoiser performs cross-role self-attention on the packed joint state;
it is not a concatenation of separately evaluated invisible and parent
denoisers. The MET residual and mass-shell codec restores physical invisible
four-vectors after J sampling. Parent roots are not duplicated with a derived
pair target.

The prediction method encodes each measured batch once before expanding the
event/sample axis. Its ODE calls receive cached measured tokens and query
contexts; no L-GATr call occurs inside a solver step.

## Closure gates

The completed CPU known-answer toy uses an ambiguous binary assignment, an
assignment-dependent invisible state, and a parent state with strong
invisible-parent dependence. M recovers useful marginals but has near-zero
sample-wise N-P covariance. C and J recover the toy's covariance. Shuffling P
within each event leaves N/P means and variances unchanged to floating-point
precision and removes the C/J covariance, validating the joint evaluator.
Truth-conditioned versus predicted-context C endpoints execute; the toy shows
large upstream context degradation, so production diagnostics should later
separate assignment and neutrino error propagation.

The bounded real contract checked 144 events (96 train, 48 validation), with
48 each in emu, ee, and mumu. Producer-declared neutrino slots and
after-FSR top/antitop branches match the charge-side convention. The existing
full train/validation measured and authoritative supervision files have the
same source SHA and exact train/validation EVENT_ID memberships as the checked
2M source. No reserved test rows are present in those inputs.

The model-native assignment support scan used the maintained
`StructuredCollator` candidate state over 1,900,095 train/validation events:
1,899,839 are supported (99.9865%). All 256 unsupported events are caused by
candidate truncation at the configured eight-jet limit. All other recorded
causes (missing role, topology/feasibility exclusion, jet multiplicity, other)
are zero.

Bounded CPU M/C/J real-data smoke jobs each completed two steps with finite
losses and nonzero intended gradients, saved and reloaded a checkpoint,
produced four posterior samples, round-tripped the prediction file, and ran
the joint evaluator. C also ran the truth/predicted oracle endpoints. These
144-event runs are execution checks, not comparative physics results.

Matched full-population production configs use the same measured cache,
supervision files, scalers, L-GATr, topology queries, assignment model,
optimizer, scheduler, batch size, precision, 20-epoch fixed stopping rule,
and 8-step Heun flow budget. Shared parameter tensors hash identically under
seed 42011. Variant-specific flows are deterministically initialized with the
same recorded seed. J has a different parameter count; capacity matching is
deferred unless production shows a scientifically meaningful difference.

No GPU or Slurm production job was submitted. The reserved test population
was neither materialized nor read for truth or model evaluation.
