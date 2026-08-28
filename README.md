# JointEventReconstruction

Topology-aware particle-physics event reconstruction with discrete visible assignments
and probabilistic invisible-particle inference. The repository is deliberately small:
known decay structure lives in topology data, mathematical transport lives in small flow
modules, and neural networks learn the event-dependent relationships.

There is deliberately **no neutrino-regression mode**.

## Model in one picture

The maintained code supports two primary scientific factorizations.

**Factorized**

```text
observed event X -> structured p(A|X) -> conditional p(nu|X,A)
```

**Joint flow**

```text
observed event X -> (A_t, nu_t [, H_t]) -> ... -> joint samples (A, nu)
```

`A` is a topology-constrained visible assignment and `nu` contains one or more ordered
invisible roles with shape `[Rnu, 3] = [px, py, pz]`. `H_t`, when enabled, is a
persistent topology-shaped latent reasoning state.

The topology compiler is authoritative. Neural modules do not contain process branches
for ttbar1L, ttbar2L or ttH1L.

## Architecture

### Visible event encoder

Observed jets, leptons, MET and configured global information are encoded once by a
Transformer. The fixed visible representation is then reused by candidate reconstruction,
recurrent reasoning and generative inference. Iterative flow/reasoning therefore does
not repeatedly re-encode the raw detector event.

### Topology compiler and hierarchical candidates

`configs/topology/*.yaml` declares visible leaves, repeated exchangeable roles,
composite references and invisible roles. The compiler derives the decay DAG, roots,
parents/children, candidate pools, visible ownership, invisible completion, topology
reasoning tokens and a stable topology signature.

Composite reconstruction is genuinely hierarchical: a learned child candidate such as a
W is reused inside its parent top rather than independently relearning the same leaf set.
Repeated leaves under one role are represented with exact exchange symmetry.

### M1 / M2 / M3 visible baselines

- **M1**: one-pass structured candidate scoring.
- **M2**: shared recurrent topology reasoning; the same reasoning block can be evaluated
  at different recurrent depths.
- **M3**: M2 plus differentiable soft reconstruction feedback into later reasoning steps.

These remain first-class discriminative baselines for all later flow studies.

### Invisible inference

All invisible formulations use the same role-aware Transformer and semantic
`[B, Rnu, 3]` Cartesian target representation.

- **standard**: VP Gaussian diffusion with epsilon prediction and deterministic DDIM;
- **edm**: EDM preconditioning, log-normal training noise and Heun sampling;
- **flow_matching**: linear conditional Flow Matching with Euler or Heun inference;
- **OT-CFM**: the same CFM model with exact minibatch Hungarian source-noise coupling.

OT only reassigns independent Gaussian source samples. Event/truth pairs are never
shuffled.

### C1, C2 and compatibility controls

Factorized discriminative reconstruction can condition invisible inference on assignment
information using `none`, `soft`, `hard` or `truth` modes. `truth` is explicitly an
oracle diagnostic. Detached C2 posterior feedback lets generated invisible information
feed back into recurrent reconstruction without differentiating through the generator.
Compatibility scoring remains a learned reranking diagnostic and is never called an
exact likelihood.

### Discrete Flow Matching

Assignment-only DFM evolves from a transient `MASK` state towards complete clean
assignments. Its state space is compiled from globally feasible root assignments, so
incompatible object reuse has zero support by construction. A physical root `NULL` is
distinct from generative `MASK`.

Factorized DFM is intentionally assignment-only. Conditioning an invisible generator on
an intermediate corrupted DFM state would have ambiguous semantics; use discriminative
factorized reconstruction for `p(nu|X,A)` or joint flow for genuine coupled inference.

### Joint DFM + CFM

Joint flow evolves

```text
A_t   complete discrete assignment state
nu_t  continuous invisible momentum state
H_t   optional persistent topology state
```

One shared step implements both coupling directions:

```text
nu_t -> topology reasoner -> complete assignment posterior
complete assignment posterior -> expected parent context -> neutrino velocity
```

The discrete side currently uses a masked absorbing categorical path and clean-state
prediction. The continuous side uses linear CFM. Both share flow time `t`.

Joint inference currently uses one Euler update per shared network evaluation. There is
no nested `flow_steps x recurrent_steps` loop. With
`model.joint_flow.recurrent_state: true`, only topology tokens persist between flow
states; the observed-event encoding remains fixed.

## Maintained configuration matrix

| mode | assignment | invisible | meaning |
|---|---|---|---|
| `factorized` | `discriminative` | disabled | M1/M2/M3 visible reconstruction |
| `factorized` | `discriminative` | `standard` | standard diffusion posterior |
| `factorized` | `discriminative` | `edm` | EDM posterior |
| `factorized` | `discriminative` | `flow_matching` | continuous FM posterior |
| `factorized` | `flow_matching` | disabled | assignment-only DFM |
| `joint_flow` | `flow_matching` | `flow_matching` | joint sampled `(A,nu)` posterior |
| `joint_flow` | `flow_matching` | `flow_matching` + recurrent state | joint flow with persistent topology state |

Invalid combinations fail during configuration validation rather than being silently
ignored.

## Complete assignment state space and scaling

Complete assignment tensors are needed only by DFM and joint flow. Ordinary
M1/M2/M3/standard/EDM/CFM runs therefore build candidate templates but **do not
materialise the complete state space**.

For DFM/joint flow:

```yaml
model:
  reconstruction:
    flow_matching:
      source: mask
      max_states: 100000
```

`max_states` is a memory/safety guard, not a trained architecture choice. If a topology
exceeds it, the code reports the actual state count and per-root candidate counts rather
than unexpectedly allocating a very large `[B,C]` tensor. Exact complete-state DFM is
kept while practical because it gives an unusually clean globally constrained baseline.
A scalable role-wise DFM is intentionally deferred until a topology actually requires it.

## Data and invisible scaling

Authoritative real products are canonical event-aligned VyPER HDF5 files. Semileptonic
data have one invisible role, while dilepton data keep two ordered and distinct roles:

```text
(top_e.nu, top_mu.nu) = (nu_e, nu_mu)
```

The electron/muon branches are never symmetrised.

```bash
python scripts/data/convert_vyper_joint.py --topology ttbar1L \
  --output data/real/ttbar1L_canonical.h5
python scripts/data/convert_vyper_joint.py --topology ttbar2L \
  --output data/real/ttbar2L_canonical.h5
```

`EVENT_ID = source_partition * 2^32 + source_row`, with partition zero from the even
source and partition one from the odd source. Even is deterministically split 95% train /
5% validation and odd is the independent test partition.

For real data, invisible mean/std statistics are fitted from the **full stable TRAIN
split before `max_events` subsampling**. Bounded studies therefore use the same target
coordinates. The scaler is saved in every checkpoint and prediction restores the
checkpoint scaler rather than silently recomputing a different one. Synthetic fixtures
fit only their fixture training events.

Static candidate and assignment templates depend only on topology plus fixed token layout
and are constructed once per collator rather than per event.

## Configuration and study overrides

Primary maintained experiment files are:

```text
configs/experiments/ttbar1L.yaml
configs/experiments/ttbar2L.yaml
configs/experiments/ttH1L.yaml
```

Obsolete pre-unified stage configs are not maintained. Scientific variations should use
the same experiment plus repeatable dot-list overrides:

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml \
  --set model.invisible.enabled=true \
  --set model.invisible.formulation=flow_matching \
  --set model.invisible.conditioning.assignment=none \
  --set training.output=results/ttbar1L_cfm
```

Assignment-only DFM:

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml \
  --set model.reconstruction.formulation=flow_matching \
  --set model.invisible.enabled=false \
  --set training.output=results/ttbar1L_dfm
```

Joint flow:

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml \
  --set model.mode=joint_flow \
  --set model.reconstruction.formulation=flow_matching \
  --set model.invisible.enabled=true \
  --set model.invisible.formulation=flow_matching \
  --set model.invisible.conditioning.assignment=none \
  --set training.output=results/ttbar1L_joint_flow
```

Add:

```bash
--set model.joint_flow.recurrent_state=true
```

for persistent joint topology recurrence.

This avoids a Cartesian product of near-identical YAML files.

## Training, prediction and evaluation

One CLI is maintained:

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml

python -m joint_event_reconstruction.cli.run predict \
  --experiment configs/experiments/ttbar2L.yaml \
  --checkpoint results/run/checkpoints/best-loss.ckpt \
  --split test --output results/run/test_predictions.h5 --device cuda

python -m joint_event_reconstruction.cli.run evaluate \
  --experiment configs/experiments/ttbar2L.yaml \
  --predictions results/run/test_predictions.h5
```

Prediction uses the scientifically natural representation for each model family:

1. **Discriminative factorized**: exact constrained top-K complete assignments followed by
   `K x M` invisible samples. Assignment probabilities/logZ retain their exact constrained
   semantics.
2. **Factorized DFM**: `S` sampled clean assignment endpoints. Evaluation reports empirical
   mode accuracy, truth sample frequency/coverage, per-node results and Njet dependence.
   Sample frequencies are Monte-Carlo quantities, not exact likelihoods.
3. **Joint flow**: `S` complete sampled `(A,nu)` hypotheses. Physical NULL branches are
   represented as unsupported/NaN composites instead of invented four-vectors, so support
   and conditional kinematic quality can be evaluated separately.

DFM uses:

```yaml
inference:
  posterior_samples: 16
  assignment_flow: {steps: 8}
```

Joint flow uses:

```yaml
inference:
  posterior_samples: 16
  joint_flow: {steps: 8, solver: euler}
```

Prediction files record topology/checkpoint provenance and inference cost. DFM and joint
outputs explicitly distinguish NFE per posterior trajectory from total network calls per
event.

## Topology interface

A production topology should require only:

1. a topology YAML;
2. a visible dataset configuration exposing the named sources;
3. raw visible truth labels with the declared semantic indices;
4. invisible truth roles in declared order when invisible inference is enabled.

No model code should branch on the topology name. See `docs/topology.md` for the grammar.
`configs/topology/synthetic_contract.yaml` is intentionally non-ttbar and exists solely to
prove hierarchy, repeated-role symmetry, invisible roles and multi-root assignment
construction through the same maintained path.

## Validation

Software smoke results are mechanism checks only; their losses are not physics results.

### Portable synthetic CPU smoke

No Manchester files are required:

```bash
python scripts/smoke_synthetic.py --topology ttbar1L --device cpu --events 4
python scripts/smoke_synthetic.py --topology ttbar2L --device cpu --events 4
python scripts/smoke_synthetic.py --topology synthetic_contract --device cpu --events 4
```

The ttbar jobs exercise all maintained old/new modes, including exact pair symmetry,
recurrent depths, M3 feedback, standard/EDM generation, C1/C2, top-K, compatibility,
CFM/OT-CFM, DFM, joint flow, bidirectional joint gradients and recurrent joint state.
The synthetic contract additionally proves that these mechanisms are not tied to ttbar
node names.

All three run as a CPU GitHub Actions matrix on relevant source/config changes.

### Manchester cache/real-data smoke

```bash
python scripts/smoke_all.py --topology ttbar1L --device cpu --events 8
python scripts/smoke_all.py --topology ttbar2L --real --device cpu --events 8

python scripts/smoke_flows.py \
  --experiment configs/experiments/ttbar1L.yaml --device cpu --events 8
python scripts/smoke_flows.py \
  --experiment configs/experiments/ttbar2L.yaml --device cpu --events 8
```

Repeat the relevant real-data matrices on CUDA before launching a campaign.

### Scientific validation

Only trained held-out real-data checkpoints support physics conclusions. Common external
metrics should be used across generator formulations: energy score, coverage, component
residuals, composite masses/kinematics, MET closure and actual inference cost.

## Planned controlled study order

1. M1 vs M2 vs M3 visible reconstruction.
2. Recurrent vs parameter-matched and compute-matched non-recurrent controls.
3. Standard diffusion vs EDM vs CFM, then OT-CFM.
4. C1 none vs soft vs truth oracle; hard only as a diagnostic.
5. Detached C2 invisible-to-reconstruction feedback.
6. Discriminative assignment vs assignment-only DFM.
7. Best factorized `p(A|X)p(nu|X,A)` vs joint DFM+CFM.
8. Stateless vs persistent-recurrent joint flow.
9. Only then consider explicit topology-relation attention/equivariance extensions.

Aggregate results should always be supplemented by Njet and event-ambiguity studies;
recurrence should additionally track wrong→correct and correct→wrong transitions across
reasoning depth.

## Speed and scaling knobs

The main quantities to report alongside accuracy are:

```text
B   batch size
C   complete DFM assignment-state count
K   factorized top-K assignments
M   posterior samples
R   recurrent depth
NFE EDM/CFM/joint-flow network evaluations
```

Important implementation choices:

- discriminative baselines do not build complete DFM state tensors;
- exact minibatch OT is optional and not the default;
- static topology/candidate templates are reused;
- training and iterative flow paths remain tensorised;
- bounded Python loops are acceptable in detailed offline physics composition/evaluation;
- no fake `denoiser_chunk_size` option exists until chunked execution is actually
  implemented.

Large posterior sampling is currently sampled trajectory-by-trajectory for DFM/joint
flow. Vectorised/chunked posterior inference is a legitimate future speed optimisation,
but should be added only with identical prediction semantics and measured memory benefit.

## Current intentional boundaries

The following are **not** silently implemented:

- scalable role-wise DFM beyond exact complete assignment states;
- discrete paths other than the MASK absorbing path;
- a higher-order coupled discrete/continuous joint solver;
- explicit topology-DAG attention masks/relation biases inside the joint reasoner;
- symmetry-aware OT beyond the existing exact assignment-symmetry quotient;
- cylindrical or Lorentz-equivariant visible encoders;
- adaptive halting/test-time compute;
- physics guidance or preference optimisation;
- MeanFlow/flow-map one-step generation.

These are future research hypotheses, not compatibility gaps. The current code is meant
to provide a clean baseline on which each can be tested independently.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

Missing real data, split files or canonical sidecars are fatal. Synthetic invisible truth
requires explicit `allow_synthetic: true`.

## M0 reference validation

The frozen M0 semantic-candidate Transformer from ReconstructionAndSB remains under
`joint_event_reconstruction.m0` solely for controlled equivalence/reference checks.

```bash
python scripts/validate_m0_equivalence.py \
  --reference-root /mnt/iusers01/fatpou01/phy01/w00238tl/scratch/ReconstructionAndSB_ref \
  --checkpoint /path/to/best_reconstruction.ckpt --events 32
```

See `PROVENANCE.md` for source revisions and local-state decisions.
