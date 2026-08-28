# JointEventReconstruction

Topology-aware particle-physics event reconstruction with discrete visible assignments and
probabilistic invisible-particle inference.

The repository is intentionally organised around a small number of scientific concepts:

```text
observed event X
    |
    v
visible Transformer
    |
    +--> topology-aware assignment A
    |
    +--> invisible posterior nu
```

The topology YAML describes the decay graph. Neural code does not branch on process names.
There is no separate neutrino-regression mode and no compatibility layer for obsolete
experiment settings.

## Scientific model

Two model families are maintained.

### Factorized

```text
p(A, nu | X) = p(A | X) p(nu | X, A)
```

The assignment term can be a discriminative structured model or assignment-only Discrete
Flow Matching (DFM). Invisible inference can use standard diffusion, EDM, or continuous
Flow Matching (CFM) when the assignment model is discriminative.

### Joint flow

```text
(A_t, nu_t [, H_t]) -> ... -> (A, nu)
```

A shared discrete-continuous Flow-Matching step updates complete assignment state `A_t`
and invisible momentum state `nu_t`. `H_t` is an optional persistent topology-shaped
latent state. The visible detector event is encoded once and reused throughout inference.

For every topology,

```text
nu shape = [B, Rnu, 3] = [px, py, pz]
```

where `Rnu` is the number of ordered invisible roles compiled from the topology.

## Architecture

### Visible event encoder

Jets, leptons, MET and configured global features are encoded by one Transformer. The
resulting event/token representation is reused by structured assignment, recurrent
reasoning and generative invisible inference rather than re-encoding detector inputs at
each iterative step.

### Hierarchical structured assignment

`configs/topology/*.yaml` defines visible leaves, repeated exchangeable roles, composite
references and invisible roles. The compiler derives the decay DAG, candidate pools,
visible-object ownership, globally feasible complete assignments, recurrent topology
tokens and a stable topology signature.

Composite candidates are hierarchical: a learned child such as a W is reused inside its
parent top. Repeated daughters within one role have exact exchange symmetry.

### M1 / M2 / M3 discriminative baselines

- **M1**: one-pass structured assignment.
- **M2**: M1 plus shared recurrent topology reasoning.
- **M3**: M2 plus differentiable soft assignment feedback into later recurrent steps.

M2/M3 train with recurrent depths sampled from
`model.assignment.reasoning.train_steps`; evaluation depth is independently selectable.

### Invisible generators

All invisible formulations share the same role-aware Transformer denoiser/velocity model.

- `standard`: VP Gaussian diffusion with deterministic DDIM inference.
- `edm`: EDM preconditioning with log-normal training noise and Heun sampling.
- `flow_matching`: linear CFM with Euler or Heun inference.
- `flow_matching` + `coupling: minibatch_ot`: minibatch Hungarian source-noise coupling.

OT changes source-noise pairing only; event/truth pairs are never shuffled.

### Assignment-only DFM

Factorized DFM evolves from a transient MASK state to a clean complete assignment. The
clean categorical state space is compiled from globally feasible root tuples, so
incompatible visible-object reuse has zero support by construction.

Factorized DFM is deliberately assignment-only. Use discriminative factorized assignment
for `p(nu|X,A)` or joint flow for coupled discrete-continuous inference.

### Joint DFM + CFM

One joint step implements both directions of coupling:

```text
nu_t -> topology reasoner -> assignment posterior
assignment posterior -> expected parent context -> invisible velocity
```

The discrete side uses the fixed MASK absorbing path; the continuous side uses linear
CFM. Joint inference uses one Euler update per shared network evaluation. With
`model.joint_flow.recurrent_state=true`, topology state persists between flow steps while
the observed-event encoding stays fixed.

## Maintained experiment configuration

The production experiment files are:

```text
configs/experiments/ttbar1L.yaml
configs/experiments/ttbar2L.yaml
configs/experiments/ttH1L.yaml
```

The public structure is:

```yaml
experiment: ...
topology_config: ...
visible_config: ...

data:
  tensor_cache: ...
  max_events: {train: ..., val: ..., test: ...}
  max_assignment_states: 100000
  invisible: {path: ..., allow_synthetic: false}

model:
  mode: factorized
  visible: ...
  assignment:
    formulation: discriminative
    reasoning: ...
    feedback: ...
  invisible:
    enabled: false
    formulation: edm
    conditioning: {assignment: none, allow_truth_oracle: false}
    denoiser: ...
    standard: ...
    edm: ...
    flow_matching: ...
  joint_flow: ...
  coupling:
    invisible_to_assignment: ...
  compatibility: ...

inference:
  top_k_assignments: 1
  posterior_samples: 1
  invisible: ...
  assignment_flow: ...
  joint_flow: ...
  compatibility_weight: 1.0

training:
  learning_rate: ...
  weight_decay: ...
  epochs: ...
  precision: ...
  gradient_clip: ...
  accumulate_grad_batches: ...
  loss_weights: {assignment: 1.0, invisible: 1.0, compatibility: 1.0}
  freeze: {visible: false, assignment: false, invisible: false, compatibility: false}
  scheduler: ...
  early_stopping: ...
  output: ...
```

Validation uses exact key sets. Misspelled, obsolete or duplicate settings fail instead of
being ignored or translated.

Fixed implementation facts are not exposed as fake choices. In particular, MASK-source
DFM, the linear CFM path, detached C2 feedback and joint-flow Euler integration are fixed
by the maintained implementation.

## Assignment conditioning for invisible inference

`model.invisible.conditioning.assignment` has five explicit meanings.

| value | training semantics | prediction semantics | status |
|---|---|---|---|
| `none` | no assignment summary | one event-level posterior | baseline |
| `soft` | probability-weighted candidate summary | one posterior conditioned on predicted soft assignments | differentiable C1 |
| `hard` | local argmax candidate summary | one posterior conditioned on predicted hard local assignments | diagnostic |
| `hypothesis` | truth assignment is used to train `p(nu|X,A)` on reconstructable events | each decoded **predicted** complete assignment gets its own hard conditioning branch | production factorization |
| `truth` | truth assignment | truth assignment | explicit oracle only |

### `hypothesis` is not an oracle at inference

This distinction is important.

During training, `hypothesis` uses the truth candidate for each invisible parent so that
the generator learns the conditional distribution

```text
p(nu | X, A)
```

for physically reconstructable truth assignments. Events for which the required truth
assignment is not reconstructable are masked out of the invisible loss.

At prediction time, truth assignment information is not used. The assignment model first
decodes the top `K` predicted complete hypotheses. The invisible generator then receives
each predicted hypothesis separately and draws `M` samples:

```text
[B, K, M, Rnu, 3]
```

Evaluation can therefore form the truncated assignment-weighted mixture using the retained
`p(A|X)` mass.

By contrast, `truth` uses truth assignment information at inference and requires
`allow_truth_oracle=true`. It is an oracle diagnostic and must not be reported as an
end-to-end reconstruction result.

## Assignment support semantics

Complete assignment accuracy is only meaningful when its support is stated.

### Discriminative complete decoder

```text
assignment_support = complete_non_null
```

Every decoded root selects a physical candidate. Local node models still contain NULL,
but physical NULL is excluded from the exact complete top-K decoder.

### DFM and joint flow

```text
assignment_support = complete_null_including
```

The clean categorical state space includes physical NULL choices. The additional MASK
state is transient and is not part of the final physical support.

Prediction HDF5 files store:

```text
assignment_support
assignment_support_includes_null
assignment_null_semantics
assignment_mask_semantics
assignment_support_description
```

Evaluators reproduce this information in `metrics.json`. Multi-model assignment plots are
grouped by support, so `complete_non_null` and `complete_null_including` accuracies are
never drawn on one comparison axis.

Old prediction files without explicit support metadata must be regenerated rather than
silently assigned guessed semantics.

## One maintained CLI

Use:

```bash
python -m joint_event_reconstruction.cli.run <command> ...
```

The commands are `show-config`, `train`, `predict`, `evaluate`, `plot`, `smoke`, and
`validate-data`.

### Inspect the exact resolved experiment

```bash
python -m joint_event_reconstruction.cli.run show-config \
  --experiment configs/experiments/ttbar1L.yaml \
  --set model.invisible.enabled=true \
  --set model.invisible.formulation=edm
```

Use this before long campaigns when several overrides are applied.

### Train

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml \
  --device auto
```

`--device` is `auto`, `cpu` or `cuda`.

### Predict

```bash
python -m joint_event_reconstruction.cli.run predict \
  --experiment configs/experiments/ttbar1L.yaml \
  --checkpoint results/ttbar1L/checkpoints/best-loss.ckpt \
  --split test \
  --output results/ttbar1L/test_predictions.h5 \
  --device cuda
```

For discriminative recurrent assignment, `--recurrent-steps R` overrides the configured
evaluation depth without changing checkpoint identity.

### Evaluate

Prediction HDF5 is self-describing, so evaluation does not require an experiment YAML:

```bash
python -m joint_event_reconstruction.cli.run evaluate \
  --predictions results/ttbar1L/test_predictions.h5
```

### Plot

One prediction:

```bash
python -m joint_event_reconstruction.cli.run plot \
  --predictions results/ttbar1L/test_predictions.h5 \
  --output results/ttbar1L/plots
```

Several predictions:

```bash
python -m joint_event_reconstruction.cli.run plot \
  --predictions results/m1/test.h5 results/edm/test.h5 results/joint/test.h5 \
  --labels M1 EDM Joint \
  --output results/comparison \
  --formats png pdf
```

Comparison plotting requires one topology signature. Assignment accuracy is split into
separate figures for each assignment support. Energy score and inference-cost summaries
can still be compared across compatible model families.

Individual plotting writes `metrics.json` plus assignment, invisible, calibration/PIT,
composite-mass and MET diagnostics when available. Efficiency/coverage plots include
finite-event statistical errors; residual plots include zero references and compact
bias/RMSE summaries.

## Common study overrides

The repository deliberately avoids a Cartesian product of nearly identical YAML files.
Use repeatable `--set KEY=VALUE` overrides.

### M2 recurrent assignment

```bash
--set model.assignment.reasoning.enabled=true
```

### M3 recurrent + soft feedback

```bash
--set model.assignment.reasoning.enabled=true \
--set model.assignment.feedback.enabled=true
```

### EDM with no assignment conditioning

```bash
--set model.invisible.enabled=true \
--set model.invisible.formulation=edm \
--set model.invisible.conditioning.assignment=none
```

### Conditional EDM with predicted assignment hypotheses

```bash
--set model.invisible.enabled=true \
--set model.invisible.formulation=edm \
--set model.invisible.conditioning.assignment=hypothesis \
--set inference.top_k_assignments=5 \
--set inference.posterior_samples=16
```

### Continuous Flow Matching

```bash
--set model.invisible.enabled=true \
--set model.invisible.formulation=flow_matching \
--set model.invisible.conditioning.assignment=none
```

For minibatch OT training add:

```bash
--set model.invisible.flow_matching.coupling=minibatch_ot
```

### Assignment-only DFM

```bash
--set model.assignment.formulation=flow_matching \
--set model.invisible.enabled=false
```

### Joint DFM + CFM

```bash
--set model.mode=joint_flow \
--set model.assignment.formulation=flow_matching \
--set model.invisible.enabled=true \
--set model.invisible.formulation=flow_matching \
--set model.invisible.conditioning.assignment=none
```

Persistent topology state:

```bash
--set model.joint_flow.recurrent_state=true
```

## C1, C2 and compatibility

C1 is the assignment-to-invisible conditioning path. `none`, `soft`, `hard`,
`hypothesis`, and truth-oracle controls are described above.

Detached C2 feeds sampled invisible-posterior information back into recurrent assignment:

```bash
--set model.assignment.reasoning.enabled=true \
--set model.invisible.enabled=true \
--set model.coupling.invisible_to_assignment.enabled=true
```

C2 is deliberately detached from the generator. There is no `detach` setting because the
opposite behaviour is not maintained.

Compatibility is an optional factorized reranker:

```bash
--set model.invisible.enabled=true \
--set model.compatibility.enabled=true \
--set inference.compatibility_weight=1.0
```

Its score is not a normalised likelihood and prediction metadata says so explicitly.

## Freeze semantics and optimiser

The only freeze keys are:

```yaml
training:
  freeze:
    visible: false
    assignment: false
    invisible: false
    compatibility: false
```

Frozen parameters are excluded from AdamW entirely. CUDA training uses fused AdamW when
available through PyTorch. Joint flow has shared assignment/invisible dynamics, so only
`training.freeze.visible` is supported in joint-flow mode.

## Inference budgets

Training/model choices and numerical inference budgets are intentionally separated.

```yaml
inference:
  top_k_assignments: 1
  posterior_samples: 1
  invisible:
    standard: {steps: 50}
    edm: {sigma_min: 0.004, sigma_max: 160.0, rho: 7.0, nfe: 17}
    flow_matching: {steps: 8, solver: heun}
  assignment_flow: {steps: 8}
  joint_flow: {steps: 8, sample_chunk_size: 8}
```

Changing recurrent evaluation depth, NFE, solver, top-K, posterior sample count,
compatibility weight or joint sampling chunk size does not redefine the trained checkpoint.

### Joint posterior chunking

`inference.joint_flow.sample_chunk_size` controls how many posterior trajectories are
vectorised together. For `S` posterior samples and chunk size `C`, joint flow executes
approximately

```text
ceil(S / C)
```

sample chunks. The physical posterior semantics are unchanged; the setting is purely a
memory/throughput tradeoff.

Prediction records both:

```text
batched_network_evaluations_per_event = steps * ceil(S / C)
sample_equivalent_nfe_per_event       = steps * S
```

so throughput comparisons do not confuse batching with a different numerical budget.

## Data and scaling

Canonical real-data products are event-aligned HDF5 files. Current production examples
use one invisible role for semileptonic events and two ordered roles for dileptonic events:

```text
(top_e.nu, top_mu.nu) = (nu_e, nu_mu)
```

Electron and muon branches remain distinct and are never exchange-symmetrised.

For real data, invisible mean/std statistics are fitted from the full stable TRAIN split
before `max_events` subsampling. Bounded studies therefore use one stable coordinate
system. The scaler is saved in checkpoints and restored for prediction rather than being
silently refit.

Canonical conversion:

```bash
python scripts/data/convert_vyper_joint.py --topology ttbar1L \
  --output data/real/ttbar1L_canonical.h5

python scripts/data/convert_vyper_joint.py --topology ttbar2L \
  --output data/real/ttbar2L_canonical.h5
```

Validate the configured product with:

```bash
python -m joint_event_reconstruction.cli.run validate-data \
  --experiment configs/experiments/ttbar2L.yaml
```

Synthetic invisible truth requires explicit `allow_synthetic: true` and is never silently
substituted for missing real data.

## Topology and `Rnu` genericity

The model should depend on topology data, not on names such as `ttbar1L` or `ttbar2L`.
See `docs/topology.md` for the grammar.

Two synthetic CI fixtures enforce genericity:

- `synthetic_contract`: hierarchy, multiple roots, repeated-role exchange symmetry and an
  invisible role;
- `synthetic_multinu`: three ordered invisible roles (`Rnu=3`), a three-member
  exchangeable visible role, hierarchy and multiple roots.

The `Rnu=3` test exists specifically to catch hidden assumptions that invisible inference
has one or two neutrinos. It is a software contract, not a physics result.

## Portable synthetic validation

The complete synthetic validation is file-free and runs on CPU or CUDA. It exercises
forward/backward, gradients, all maintained formulations, HDF5 prediction, evaluation and
plot rendering.

Run the same four topologies used by CI:

```bash
python scripts/smoke_synthetic.py --topology ttbar1L --device cpu --events 4
python scripts/smoke_synthetic.py --topology ttbar2L --device cpu --events 4
python scripts/smoke_synthetic.py --topology synthetic_contract --device cpu --events 4
python scripts/smoke_synthetic.py --topology synthetic_multinu --device cpu --events 4
```

For production topologies the suite covers, among other things:

- M1/M2/M3 assignment paths;
- recurrent-depth execution;
- C1 and detached C2;
- standard diffusion and EDM;
- factorized CFM with `none` and `hypothesis` prediction/evaluation round trips;
- OT-CFM;
- assignment-only DFM;
- joint DFM+CFM and persistent joint state;
- deterministic bidirectional `A <-> nu` gradient probes;
- assignment-support metadata;
- prediction HDF5 validation;
- actual PNG diagnostic rendering and comparison plotting.

All synthetic losses and metrics are mechanism checks only, not physics-performance
results.

## Synthetic throughput/memory benchmark

Use:

```bash
python scripts/benchmark_synthetic.py \
  --experiment configs/experiments/ttbar1L.yaml \
  --modes visible cfm dfm joint \
  --device cuda \
  --events 32 \
  --warmup 2 \
  --iterations 5 \
  --posterior-samples 8 \
  --flow-steps 8 \
  --joint-sample-chunk-size 8 \
  --output results/synthetic_benchmark.json
```

The benchmark reports training and inference events/s, step/call timing, CUDA peak
allocated memory, trainable/total parameter counts, assignment-state count, `Rnu`, AdamW
fusion status and sample-equivalent inference work.

It uses a fixed in-memory synthetic batch and excludes HDF5 I/O. It is a relative software
profile, not a production throughput measurement or physics benchmark.

## Real-data/cache smoke

Where the Manchester data/cache products are available:

```bash
python scripts/smoke_all.py --topology ttbar1L --device cpu --events 8
python scripts/smoke_all.py --topology ttbar2L --real --device cpu --events 8

python scripts/smoke_flows.py \
  --experiment configs/experiments/ttbar1L.yaml --device cpu --events 8
```

Repeat relevant campaign configurations on CUDA before expensive training.

## Evaluation principles

Prediction products distinguish exact probabilities from Monte-Carlo frequencies.

- Discriminative factorized top-K assignment probabilities are exact constrained
  probabilities **within `complete_non_null` support**.
- DFM and joint assignment sample frequencies are Monte-Carlo estimates, not exact
  likelihoods.
- `hypothesis` factorized invisible evaluation reports the truncated
  assignment-weighted mixture and its retained assignment probability mass.
- Truth-conditioned and truth-selected diagnostics are explicitly labelled as oracle or
  selection-biased where appropriate.
- Physical NULL composites remain unsupported/NaN rather than being assigned invented
  four-vectors.

Physics conclusions should come from trained held-out real-data checkpoints using common
external observables such as assignment efficiency, energy score, coverage/PIT,
component residuals, composite kinematics, MET closure and measured inference cost.

## Controlled study order

A clean comparison programme is:

1. M1 vs M2 vs M3 assignment.
2. Recurrent vs parameter-matched and compute-matched non-recurrent controls.
3. Standard diffusion vs EDM vs CFM; then OT-CFM.
4. C1 `none` vs `soft` vs `hypothesis`; hard and truth only as diagnostics/controls.
5. Detached C2.
6. Discriminative assignment vs assignment-only DFM, respecting support semantics.
7. Best factorized `p(A|X)p(nu|X,A)` vs joint DFM+CFM.
8. Stateless vs persistent-recurrent joint flow.
9. Only after those baselines are stable, test larger architectural hypotheses one at a
   time.

Recurrence studies should include wrong->correct and correct->wrong transitions as a
function of depth rather than only aggregate final accuracy.

## Repository maintenance principles

The maintained code follows a few deliberate rules:

- topology data are authoritative;
- public names map to scientific concepts (`assignment`, `invisible`, `compatibility`);
- no process-name model branches;
- no silent compatibility aliases for deleted config names;
- no fake options for fixed implementation details;
- no fallback to synthetic data when real data were requested;
- prediction schemas record scientific support and provenance explicitly;
- frozen parameters do not enter the optimiser;
- bounded Python loops are acceptable for detailed offline physics composition/evaluation,
  while training and iterative model paths stay tensorised;
- Git history is the archive for removed entry points rather than a live `legacy/` tree.

The frozen M0 implementation under `joint_event_reconstruction.m0` is retained solely for
controlled reference/equivalence checks. Historical standalone M0 evaluation and Hydra
split-preparation commands are not part of the maintained interface.

## M0 reference equivalence

```bash
python scripts/validate_m0_equivalence.py \
  --reference-root /path/to/ReconstructionAndSB_ref \
  --checkpoint /path/to/best_reconstruction.ckpt \
  --events 32
```

See `PROVENANCE.md` for source revisions and reference-state decisions.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

For a minimal CI-like check after installation:

```bash
python -m compileall -q src scripts
python scripts/smoke_synthetic.py --topology synthetic_multinu --device cpu --events 4
```
