# JointEventReconstruction

Topology-aware event reconstruction with structured visible assignments and probabilistic invisible-particle inference.

The repository is organised around three explicit inputs:

```text
observed event X
    |
    v
visible encoder
    |
    +--> topology-aware assignment A
    |
    +--> invisible posterior nu
```

The decay graph lives in `configs/topology/*.yaml`. Model code is topology-driven and does not branch on process names.

The v2 framework keeps the shared JER latent contract separate from the visible
input backbone.  Set `model.latent` for downstream topology reasoning and choose
one `model.encoder.type`: `transformer` (the maintained baseline),
`physics_transformer`, `lgatr`, or `lgatr_slim`.  Physics-aware backbones consume
the token-aligned `(E,px,py,pz)` `physics_vectors` field; padded rows are zero and
the physical MET token is the sole MET source.  L-GATr is optional:

```bash
pip install -e '.[lgatr]'
```

Invisible DDIM/EDM/CFM states are coordinate-agnostic.  `cartesian` preserves
the original state, `met_residual` is an exactly invertible MET-aware basis, and
diagnostic `met_exact` projects truth transversely to measured MET.  Prediction
products always store Cartesian neutrino momenta and, for `met_exact`, the
projected target and residual are recorded separately.  CFM coupling can be
`independent`, `minibatch_ot`, or detached condition-aware `conditional_ot`.

The full v2 seams, identity rules, calibration separation and human-gated study
graph are in [docs/FRAMEWORK_V2.md](docs/FRAMEWORK_V2.md).  Study 12 and Study 13
are defined templates and are not submitted automatically.

## Maintained scientific modes

### Factorized reconstruction

```text
p(A, nu | X) = p(A | X) p(nu | X, A)
```

The assignment model can be:

- **M1**: one-pass discriminative structured assignment;
- **M2**: M1 plus recurrent topology reasoning;
- **M3**: M2 plus differentiable soft assignment feedback;
- **DFM**: assignment-only Discrete Flow Matching on globally feasible complete assignments.

For discriminative assignment, invisible inference can use DDIM, EDM or continuous Flow Matching (CFM). CFM can optionally use minibatch OT source-noise coupling.

### Joint flow

```text
(A_t, nu_t [, H_t]) -> ... -> (A, nu)
```

Joint flow couples complete-assignment DFM and invisible CFM in one model. `H_t` is an optional persistent topology-shaped latent state. The visible detector event is encoded once and reused throughout inference.

For every topology:

```text
nu shape = [B, Rnu, 3] = [px, py, pz]
```

where `Rnu` is compiled from the ordered invisible roles in the topology.

## Configuration philosophy

Files edited by users are deliberately compact. Fixed implementation details and inactive model families do not appear in public experiment YAMLs.

A public config is resolved once into the explicit internal contract consumed by model and training code:

```text
compact experiment YAML
        |
        v
public_config.py
        |
        v
strict resolved config
        |
        v
training / prediction / model code
```

Use `show-config` when the full resolved representation is useful for debugging or provenance.

## Public experiment config

Maintained experiments are:

```text
configs/experiments/ttbar1L.yaml
configs/experiments/ttbar2L.yaml
configs/experiments/ttH1L.yaml
```

A normal assignment-only experiment is intentionally small:

```yaml
experiment:
  name: ttbar2L
  seed: 42

topology: configs/topology/ttbar2L.yaml

data:
  config: configs/data/ttbar2L.yaml
  subset_seed: 42
  events: {train: 100000, validation: 50000, test: 50000}
  max_assignment_states: 100000

model:
  encoder:
    d_model: 128
    n_heads: 4
    layers: 4
    d_feedforward: 384
    dropout: 0.05
  assignment:
    method: discriminative

prediction: {}

training:
  learning_rate: 3.61545e-4
  weight_decay: 6.0735e-5
  epochs: 20
  precision: 16-mixed
  output: results/ttbar2L
```

Inactive EDM/CFM/DFM/C2/joint-flow/compatibility settings are absent. Hidden defaults are in `configs/defaults.yaml` and are expanded by the resolver.

## JER data configs

Input definitions live separately in:

```text
configs/data/ttbar1L.yaml
configs/data/ttbar2L.yaml
configs/data/ttH1L.yaml
```

A data config owns only data semantics and loading choices:

```yaml
source: data/real/ttbar2L_canonical.h5
invisible_truth: data/real/ttbar2L_canonical.h5
split: data/real/splits/ttbar2L/split_indices.npz
tensor_cache: null

loader:
  batch_size: 512
  workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

objects:
  JET:
    count: 8
    variables: [e, eta, phi, pt, btag, charge]
    scalers: [Log, "minmax:min=-2.5,max=2.5", SinCos, Log, "minmax:min=0,max=6", "minmax:min=-1,max=1"]
  ELECTRON: ...
  MUON: ...
  MET: ...
  GLOBAL: ...
```

The mature fixed-token visible-loader configuration is generated in memory from this file and the topology. There is no maintained public `visible_config` YAML.

## Compact model choices

### M1

```yaml
model:
  assignment:
    method: discriminative
```

### M2

```yaml
model:
  assignment:
    method: discriminative
    recurrence:
      train_steps: [1, 2, 3, 4]
      eval_steps: 4
```

### M3

```yaml
model:
  assignment:
    method: discriminative
    recurrence:
      train_steps: [1, 2, 3, 4]
      eval_steps: 4
      feedback: soft
```

`feedback: soft` is a public shorthand. The resolver expands it into the complete internal feedback configuration.

### Invisible DDIM / EDM / CFM

```yaml
model:
  assignment:
    method: discriminative
  invisible:
    method: edm       # ddim | edm | cfm
    conditioning: hypothesis
```

For OT-CFM:

```yaml
model:
  invisible:
    method: cfm
    conditioning: hypothesis
    coupling: minibatch_ot
```

### Assignment-only DFM

```yaml
model:
  assignment:
    method: dfm
```

### Joint DFM + CFM

```yaml
model:
  joint_flow:
    recurrent: false
```

Persistent topology state is enabled with:

```yaml
model:
  joint_flow:
    recurrent: true
```

`joint_flow` is one scientific model choice. Users do not separately enable DFM, CFM and their coupling.

## C1, C2 and compatibility

Assignment conditioning is configured through `model.invisible.conditioning`:

| value | meaning |
|---|---|
| `none` | no assignment context |
| `soft` | probability-weighted assignment context |
| `hard` | predicted local argmax context; diagnostic |
| `hypothesis` | train `p(nu|X,A)` with truth assignment, predict from each decoded predicted assignment |
| `truth_oracle` | truth assignment at training and inference; oracle diagnostic |

`hypothesis` is not an oracle at prediction time. The assignment model first decodes predicted complete hypotheses and the generator samples each hypothesis separately.

C2 feeds sampled invisible-posterior information back into recurrent assignment:

```yaml
model:
  assignment:
    method: discriminative
    recurrence:
      train_steps: [1, 2, 3, 4]
      eval_steps: 4
      feedback: soft
  invisible:
    method: cfm
    conditioning: soft
  c2:
    posterior_samples: 8
    training_sampler: {steps: 17, solver: euler}  # CFM training semantics
```

Compatibility is an optional factorized reranker:

```yaml
model:
  compatibility: {}
```

Its score is not interpreted as a normalised likelihood.

## Positive trainability semantics

Training specifies what is trainable, not what must be frozen:

```yaml
training:
  trainable: [visible, assignment]
```

or, when fitting only a generator from a frozen assignment checkpoint:

```yaml
training:
  trainable: [invisible]
```

Available semantic components are:

```text
visible
assignment
invisible
c2
compatibility
```

The complement is frozen and excluded from AdamW. Loss weights default consistently with the trainable components. Joint-flow assignment and invisible dynamics are tied because they share the joint topology-state machinery.

## Prediction config and reproducibility

Prediction settings have their own compact public section:

```yaml
prediction:
  seed: 12345
  top_k: 5
  samples: 16
  steps: 16
  solver: heun
```

The three seeds have different meanings:

```text
experiment.seed -> model/training stochasticity
data.subset_seed -> stable EVENT_ID-based train subset identity
prediction.seed -> posterior Monte-Carlo stochasticity
```

The prediction seed is stored in every prediction HDF5. Numerical inference budgets do not redefine checkpoint identity.

## Assignment support semantics

Complete assignment accuracy is only meaningful together with its support.

Discriminative complete decoding uses:

```text
assignment_support = complete_non_null
```

Every decoded root selects a physical candidate. Local node models can still predict NULL, but NULL is not part of the final exact complete top-K support.

DFM and joint flow use:

```text
assignment_support = complete_null_including
```

Their clean categorical state space includes physical NULL choices. The transient MASK state is not a final physical state.

Prediction HDF5 and evaluation output record support metadata explicitly. Cross-family studies also report a common truth-non-NULL event subset so discriminative and flow assignment results are not compared on silently different supports.

## Data scaling

Invisible scaling is fitted on the full stable TRAIN split before bounded `data.events.train` subsampling. A 10k and 500k training study therefore uses the same invisible coordinate system.

The scaler is saved in checkpoints and restored for prediction rather than silently refitted.

Canonical conversion examples:

```bash
python scripts/data/convert_vyper_joint.py --topology ttbar1L \
  --output data/real/ttbar1L_canonical.h5

python scripts/data/convert_vyper_joint.py --topology ttbar2L \
  --output data/real/ttbar2L_canonical.h5
```

Validate a configured canonical product with:

```bash
python -m joint_event_reconstruction.cli.run validate-data \
  --experiment configs/experiments/ttbar2L.yaml
```

## One maintained CLI

```bash
python -m joint_event_reconstruction.cli.run <command> ...
```

Commands are `show-config`, `train`, `predict`, `evaluate`, `plot`, `smoke`, and `validate-data`.

Inspect a resolved experiment:

```bash
python -m joint_event_reconstruction.cli.run show-config \
  --experiment configs/experiments/ttbar2L.yaml
```

Train:

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar2L.yaml \
  --device cuda
```

Predict:

```bash
python -m joint_event_reconstruction.cli.run predict \
  --experiment configs/experiments/ttbar2L.yaml \
  --checkpoint results/ttbar2L/checkpoints/best-loss.ckpt \
  --split test \
  --output results/ttbar2L/test.h5 \
  --device cuda
```

For discriminative recurrence, `--recurrent-steps R` changes evaluation depth without changing checkpoint identity.

Evaluate and plot directly from self-describing HDF5:

```bash
python -m joint_event_reconstruction.cli.run evaluate \
  --predictions results/ttbar2L/test.h5

python -m joint_event_reconstruction.cli.run plot \
  --predictions results/ttbar2L/test.h5 \
  --output results/ttbar2L/plots
```

## Study manifests

Controlled studies live in `configs/studies/`. Variants use structured YAML patches rather than lists of dot-notation command-line overrides.

Example:

```yaml
variants:
  c2:
    label: detached C2 feedback
    reporting: primary
    patch:
      model:
        assignment:
          method: discriminative
          recurrence:
            train_steps: [1, 2, 3, 4]
            eval_steps: 4
            feedback: soft
        invisible:
          method: cfm
          conditioning: soft
        c2:
          posterior_samples: 8
      training:
        trainable: [assignment, c2]
    initialise_from: c1
    initialise_components: [visible, assignment, invisible]
    predictions: ...
```

Each run writes both the compact merged experiment and the fully resolved internal YAML beside its outputs.

### Dependency closure

Requesting a downstream variant automatically includes its training prerequisites. For example:

```bash
python scripts/run_study.py configs/studies/04_c2_feedback.yaml train \
  --variants c2 --device cuda
```

can schedule:

```text
assignment_reference -> c1 -> c2
```

without requiring the caller to list those precursor checkpoints manually. Predictions and summaries still report only the requested variant.

Plan a study without running it:

```bash
python scripts/run_study.py configs/studies/04_c2_feedback.yaml plan \
  --variants c2 --dry-run
```

Submit the same filtered dependency graph to Slurm:

```bash
python scripts/submit_study_slurm.py configs/studies/04_c2_feedback.yaml \
  --variants c2
```

Slurm workers run one exact variant per dependency level, avoiding recursive resubmission of precursor variants.

Each real submission gets an immutable campaign directory under
`results/studies/<study>/_slurm/<campaign-id>/`. Its `submission.json` records
the source commit, manifest, output root, variants, seeds and dependency graph;
task files and logs from a previous campaign are never reused. Training and
prediction artifacts also carry identity sidecars. Matching identities may be
resumed, while a stale or incomplete artifact is a hard error unless the user
explicitly requests `--force` (which invalidates dependent products).

Studies 02--11 are human decision-gated templates. They may be dry-planned
while their upstream choices are pending, but real submission is blocked until
the corresponding entry in `configs/studies/UPSTREAM_SELECTIONS.yaml` is
locked. Study 01 is the first submission-ready gate and uses validation for
selection; the locked finalists are evaluated on test only after the decision
gates are recorded.

## Maintained study sequence

The initial controlled campaign is intentionally ttbar2L-first:

```text
01_visible_recurrence
02_generator_formulations
03_assignment_conditioning
04_c2_feedback
05_discrete_assignment
06_joint_vs_factorized
07_joint_recurrence
08_generator_inference
09_posterior_size
10_compatibility_reranking
11_data_scaling
```

This keeps topology fixed while individual modelling effects are isolated. A `ttH2L` topology is not currently defined and is deliberately not fabricated; it can be added to the same study infrastructure once its real topology and canonical data mapping exist.

## Data-scaling study

`configs/studies/11_data_scaling.yaml` scans:

```text
10k, 25k, 50k, 100k, 250k, 500k training events
```

with fixed validation/test samples and nested EVENT_ID-based training subsets.

## Recurrence controls

Generate both a parameter-matched and a training-compute-matched non-recurrent control for M2/M3:

```bash
python scripts/match_recurrence_controls.py \
  --experiment configs/experiments/ttbar2L.yaml \
  --target m3 \
  --recurrent-steps 4 \
  --device cuda \
  --output results/recurrence_controls_m3.json
```

The script first searches non-recurrent encoder depth/feed-forward width by parameter count, then benchmarks shortlisted candidates and the recurrent target on the same synthetic batch. It records separate parameter- and training-compute-matched controls; the latter is not automatically inference-compute-equivalent. These measurements are engineering provenance, not physics results.

## Validation

The synthetic smoke suite is explicitly a mechanism/software test, not a physics benchmark. CI checks:

- Python compilation and fatal Ruff errors;
- compact experiment resolution and generated visible-loader configs;
- every maintained structured study patch;
- filtered dependency closure and Slurm planning;
- topology-generic synthetic smokes for 1L, 2L and synthetic stress topologies;
- synthetic throughput smoke;
- recurrence-control generation.

Run locally with:

```bash
python scripts/smoke_synthetic.py --topology ttbar2L --device cpu --events 4
```

## Development rule

The maintained interface is strict. Misspelled or obsolete public settings fail rather than being silently ignored. Fixed implementation facts are not exposed as fake switches. New scientific mechanisms should only be added after the existing controlled study gates have established which current components are useful.
