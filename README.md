# JointEventReconstruction

Topology-aware particle-physics event reconstruction with discrete visible assignments
and probabilistic invisible-particle inference.  The repository is deliberately small:
known decay structure lives in topology data, mathematical transport lives in small flow
modules, and the neural networks learn only the event-dependent relationships.

There is deliberately **no neutrino-regression mode**.

## Model in one picture

The maintained code supports two scientific factorizations.

**Factorized**

```text
observed event X -> structured p(A|X) -> conditional p(nu|X,A)
```

**Joint flow**

```text
observed event X -> (A_t, nu_t [, H_t]) -> ... -> joint samples (A, nu)
```

`A` is a complete topology-constrained visible assignment and `nu` contains one or more
ordered invisible roles with shape `[Rnu, 3] = [px, py, pz]`.  `H_t`, when enabled, is
a persistent topology-shaped latent reasoning state.

The topology compiler remains authoritative.  Neural modules do not contain process
branches for ttbar1L, ttbar2L or ttH1L.

## Maintained scientific configurations

| `model.mode` | reconstruction | invisible | meaning |
|---|---|---|---|
| `factorized` | `discriminative` | disabled | structured visible baseline (M1/M2/M3 switches remain available) |
| `factorized` | `discriminative` | `standard` | VP Gaussian diffusion / DDIM |
| `factorized` | `discriminative` | `edm` | EDM posterior inference |
| `factorized` | `discriminative` | `flow_matching` | continuous Flow Matching neutrino posterior |
| `factorized` | `flow_matching` | disabled | assignment-only masked Discrete Flow Matching baseline |
| `joint_flow` | `flow_matching` | `flow_matching` | coupled DFM+CFM posterior over complete `(A, nu)` hypotheses |
| `joint_flow` | `flow_matching` | `flow_matching` + recurrent state | coupled flow with persistent topology reasoning state |

Factorized C1 soft/hard/truth-oracle assignment conditioning, detached C2 posterior
feedback, exact top-K assignment-conditioned generation, and compatibility reranking are
retained as controls for **discriminative factorized reconstruction**.  They are not
silently mixed into joint flow.

Factorized DFM is intentionally assignment-only at present.  Conditioning an invisible
generator on an intermediate corrupted DFM state would have ambiguous scientific
semantics; use the discriminative factorized model for `p(nu|X,A)` or joint flow for
bidirectional assignment-invisible inference.

Invalid combinations fail during configuration validation.

## Structured assignments and symmetry

`configs/topology/*.yaml` uses four constructs: visible leaves `[SOURCE, truth_label]`,
repeated exchangeable roles, composite references by node name, and invisible
`[NEUTRINO, index]` leaves. References compile into a DAG; YAML declaration order is
irrelevant. MET remains observed context and is never treated as a decay daughter.

The maintained reconstruction:

- constructs hierarchy-preserving candidates, reusing the actual child embedding inside
  its parent;
- represents repeated physical roles with exact exchange symmetry;
- compiles a tensorised `AssignmentSpace` of globally feasible complete root states;
- enforces leaf ownership exactly, so invalid complete assignments have zero support;
- distinguishes a physical per-root `NULL` state from the transient DFM `MASK` source;
- normalises joint-flow assignment beliefs over the complete feasible state space.

The CPU constrained decoder remains a reference implementation.  Training and new-flow
prediction use the tensorised assignment space.

## Invisible inference

All invisible formulations use the same role-aware Transformer network and the same
semantic `[B, Rnu, 3]` Cartesian target representation.

- **standard**: cosine VP Gaussian diffusion, epsilon prediction and deterministic DDIM;
- **edm**: Karras-style preconditioning, log-normal training noise and Heun sampling;
- **flow_matching**: straight conditional path
  `x_t = (1-t) x_0 + t x_1`, direct velocity prediction, Euler or Heun sampling;
- **OT-CFM**: the same CFM model with exact minibatch Hungarian source-noise coupling.

OT pairing only permutes independent Gaussian source samples; event/truth pairs are not
shuffled.

Sampling budgets and numerical solvers are inference choices, not checkpoint
architecture.  A trained EDM checkpoint can therefore be used for NFE scans and a CFM
checkpoint for Euler/Heun or step-count scans without changing the active architecture
signature.

## Joint discrete-continuous Flow Matching

Joint flow evolves one physical state

```text
A_t   complete discrete assignment state
nu_t  continuous invisible momentum state
H_t   optional persistent topology state
```

One shared step performs both coupling directions:

```text
nu_t -> topology reasoner -> complete assignment posterior
complete assignment posterior -> expected parent context -> neutrino velocity
```

The discrete side currently uses a masked absorbing categorical path and clean-state
prediction.  The continuous side uses linear CFM.  Both use the same flow time `t`.
Joint inference currently uses one Euler update per shared network evaluation; there is
no nested `flow_steps x recurrent_steps` loop.

With `model.joint_flow.recurrent_state: true`, training uses a short coherent
teacher-forced trajectory and carries only topology tokens between flow times.  The main
loss is evaluated at the final unrolled state.  The fixed detector/event encoding is
reused at every step.

## Data and invisible scaling

Authoritative real products are the canonical event-aligned VyPER Run-3 HDF5 files.
Semileptonic data have one invisible role, `tlep.nu`.  Dilepton data have two distinct,
ordered roles

```text
(top_e.nu, top_mu.nu) = (nu_e, nu_mu)
```

and the electron/muon branches are never symmetrised.

```bash
python scripts/data/convert_vyper_joint.py --topology ttbar1L \
  --output data/real/ttbar1L_canonical.h5
python scripts/data/convert_vyper_joint.py --topology ttbar2L \
  --output data/real/ttbar2L_canonical.h5
```

`EVENT_ID = source_partition * 2^32 + source_row`, with partition zero from the even
source file and partition one from the odd source file.  Even is deterministically split
95% train / 5% validation and odd is the independent test partition.

For real data, invisible mean/std statistics are fitted **once from the full stable TRAIN
split before `max_events` subsampling**.  Thus 100k, 500k and larger experiments use the
same target coordinates.  The scaler is stored in every checkpoint and prediction loads
the checkpoint scaler rather than recomputing a potentially different coordinate system.
Synthetic software fixtures fit only the synthetic fixture events.

## Configuration

Primary experiment files are:

```text
configs/experiments/ttbar1L.yaml
configs/experiments/ttbar2L.yaml
configs/experiments/ttH1L.yaml
```

The YAML contains only implemented choices.  Fixed implementation facts such as standard
diffusion's cosine/epsilon/DDIM definition or EDM's Heun sampler are not presented as
fake switches.

Typical continuous-FM change:

```yaml
model:
  mode: factorized
  reconstruction:
    formulation: discriminative
  invisible:
    enabled: true
    formulation: flow_matching
    conditioning: {assignment: none, allow_truth_oracle: false}
    flow_matching:
      path: linear
      coupling: independent       # or minibatch_ot
      sampler: heun
      inference_steps: 8
```

Joint flow:

```yaml
model:
  mode: joint_flow
  reasoning: {enabled: false, ...}
  reconstruction:
    formulation: flow_matching
    feedback: {enabled: false, ...}
  invisible:
    enabled: true
    formulation: flow_matching
    conditioning: {assignment: none, allow_truth_oracle: false}
  joint_flow:
    recurrent_state: false        # true for persistent topology state
    block_layers: 2
    train_steps: [1, 2, 3, 4]
  coupling:
    rounds: 1
    invisible_to_reconstruction: {enabled: false, ...}
  compatibility: {enabled: false, ...}

inference:
  posterior_samples: 16
  joint_flow: {steps: 8, solver: euler}
```

## Training, prediction and evaluation

One CLI is maintained for every formulation:

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

Factorized prediction retains exact top-K complete assignments followed by `K x M`
invisible samples. Joint-flow prediction instead stores the natural posterior object:
sampled complete hypotheses `(A^(s), nu^(s))`.  Empirical joint-flow assignment frequency
is a Monte-Carlo posterior quantity and is **not** labelled an exact likelihood.

## Validation

There are three distinct validation levels.  Do not use software-smoke outputs as physics
results.

### 1. File-free synthetic model smoke

This is portable and requires no Manchester data/cache files.  It constructs random
visible tensors with exact topology truth labels in memory, runs the real
`StructuredCollator`, attaches synthetic invisible truth, then executes both the old and
new maintained mode assertions.

```bash
python scripts/smoke_synthetic.py --topology ttbar1L --device cpu --events 4
python scripts/smoke_synthetic.py --topology ttbar2L --device cpu --events 4
```

It checks, among other things:

- exact repeated-role exchange symmetry;
- recurrent depths 1/2/4/8;
- differentiable M3 feedback;
- standard diffusion and EDM generation;
- soft/hard/truth-oracle C1, detached C2, top-K and compatibility paths;
- CFM and OT-CFM finite loss/sampling/NFE;
- DFM complete-state validity and MASK resolution;
- joint DFM+CFM finite complete hypotheses;
- non-zero `A -> nu` and `nu -> A` gradient paths;
- persistent joint topology recurrence.

The same two synthetic CPU jobs run in GitHub Actions on changes to source/configuration.

### 2. Manchester cache/real-data smoke

The historical aggregate matrix remains available against the actual CSF3/VyPER data
environment:

```bash
python scripts/smoke_all.py --topology ttbar1L --device cpu --events 8
python scripts/smoke_all.py --topology ttbar2L --real --device cpu --events 8

python scripts/smoke_flows.py --experiment configs/experiments/ttbar1L.yaml \
  --device cpu --events 8
python scripts/smoke_flows.py --experiment configs/experiments/ttbar2L.yaml \
  --device cpu --events 8
```

Repeat the relevant real-data aggregate on CUDA before starting a campaign.

### 3. Scientific validation

Only trained real-data checkpoints support physics conclusions.  Generator comparisons
must use fixed held-out events and common conditioning/scaling/capacity, with energy
score, coverage, component residuals, composite masses, MET closure and actual NFE/cost.
Joint-vs-factorized conclusions should additionally be studied versus event ambiguity.

## Current intentional boundaries

The following are **not** silently implemented:

- scalable role-wise DFM beyond the exact complete assignment state space;
- discrete paths other than the MASK absorbing path;
- a higher-order coupled discrete/continuous joint solver;
- explicit topology-DAG attention masks/relation biases inside the joint reasoner;
- symmetry-aware OT beyond the existing exact assignment symmetry quotient;
- cylindrical or Lorentz-equivariant visible encoders;
- adaptive halting/test-time compute, physics guidance or preference optimisation;
- MeanFlow/flow-map one-step generation.

These are research extensions, not missing compatibility fallbacks.  The present code is
intended to provide a clean baseline on which each can be tested independently.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

Missing real data, split files or canonical sidecars are fatal.  Synthetic invisible
truth requires `allow_synthetic: true` explicitly.

## M0 reference validation

The frozen M0 semantic-candidate Transformer from ReconstructionAndSB remains under
`joint_event_reconstruction.m0` solely for controlled equivalence/reference checks.

```bash
python scripts/validate_m0_equivalence.py \
  --reference-root /mnt/iusers01/fatpou01/phy01/w00238tl/scratch/ReconstructionAndSB_ref \
  --checkpoint /path/to/best_reconstruction.ckpt --events 32
```

See `PROVENANCE.md` for source revisions and the local-state decision.
