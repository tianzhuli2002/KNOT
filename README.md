# JointEventReconstruction

JER is a focused framework for topology-aware visible assignment and invisible
momentum reconstruction. The maintained workflow has one canonical HDF5 input,
one deterministic tensor cache, one DataModule path, and one input declaration
shared by every model family.

The public configuration has four scientific parts:

- `data`: input HDF5 and cache destination;
- `inputs`: retained collections, maximum counts, features and transforms;
- `topology`: semantic physics roles and decays;
- `model`: encoder, assignment, invisible and joint choices.

Studies apply controlled patches to a maintained experiment.

## Normal configuration

### Current Paper-1 selection status

The maintained Paper-1 path is selected in `configs/studies/UPSTREAM_SELECTIONS.yaml`:
L-GATr visible representation, one-pass M1/discriminative assignment, and CFM
invisible generation. Invisible coordinates, invisible-generator backbone and
physics coupling remain validation gates. The example below is deliberately a
feature-rich configuration example; its MET-residual coordinates, hypothesis
conditioning and conditional-OT coupling are not implied to be selected.

This excerpt uses the maintained ttbar2L feature ranges. `count` is the maximum
number of retained objects from that collection.

```yaml
experiment: {name: ttbar2L, seed: 42}
topology: configs/topology/ttbar2L.yaml

data:
  input: data/real/ttbar2L_canonical.h5
  cache: /scratch/w00238tl/jer_cache
  subset_seed: 42
  events: {train: 100000, validation: 50000, test: 50000}
  max_assignment_states: 100000

inputs:
  transforms:
    e: log
    eta: [minmax, -2.5, 2.5]
    phi: sincos
    pt: log
    btag: [minmax, 0, 6]
    charge: [minmax, -1, 1]
    njet: [minmax, 0, 8]
    nelectron: [minmax, 0, 2]
    nmuon: [minmax, 0, 2]
    nBJets_77: [minmax, 0, 8]
  JET: {count: 8, features: [e, eta, phi, pt, btag, charge]}
  ELECTRON: {count: 1, features: [e, eta, phi, pt, charge]}
  MUON: {count: 1, features: [e, eta, phi, pt, charge]}
  MET: {count: 1, features: [phi, pt]}
  GLOBAL: {features: [njet, nelectron, nmuon, nBJets_77]}

model:
  mode: factorized
  latent: {d_model: 128, n_heads: 4, d_feedforward: 384, dropout: 0.05}
  encoder: {type: transformer, representation: four_vector, relations: physics}
  assignment:
    formulation: discriminative
  invisible:
    coordinates: {type: met_residual}
    conditioning: {assignment: hypothesis}
    generator:
      type: cfm
      coupling: {type: conditional_ot, condition_weight: 1.0}

inference:
  seed: 12345
  assignment: {top_k: 5}
  invisible: {samples_per_assignment: 32, flow: {steps: 8}}

training:
  batch_size: 512
  workers: 4
  precision: 32-true
  optimizer: {type: adamw, learning_rate: 0.000361545, weight_decay: 0.00006073529387441413}
```

Only `none`, `log`, `sincos`, `standard`, `pid`, and
`[minmax, MIN, MAX]` are accepted transforms. The canonical adapter constructs
physical vectors as `(E, px, py, pz)` in GeV with metric `(+---)`. MET remains a
measured transverse quantity and is represented internally as `(ET,px,py,0)`.

## Cache and data path

If `data.cache` is a directory, JER uses `<directory>/<input_stem>.pt`; an
explicit `.pt` path is used exactly. The file is a plain dictionary of CPU
tensors and primitive metadata. It stores transformed inputs, physical vectors,
type IDs, padding masks, GLOBAL inputs, EVENT_ID, canonical split identity,
visible truth and invisible truth. It never stores learned state.

The cache is built once with large contiguous HDF5 reads, validated, written
atomically, and mmap-loaded thereafter:

```text
canonical HDF5 -> tensor cache -> thin DataModule -> non-blocking GPU transfer
```

A missing cache builds automatically. A compatible cache is reused. An
incompatible cache is a hard error; rebuilding requires `--force` or
`--force-rebuild-cache`. A lock prevents concurrent builders.

## Encoder study

Study 12 compares five views of identical selected event information:

1. Transformer with transformed scalar features;
2. Transformer with four-vectors;
3. Transformer with four-vectors and physics pairwise relations;
4. L-GATr;
5. L-GATr-Slim.

This separates information, representation, and inductive bias. The ttbar2L
dataset remains deliberately electron-muon only, with electron-side and
muon-side semantic truth. Charge is not treated as proven top/antitop identity.
Flavour is not semantic role; same-flavour invisible particles remain distinct
when their parent roles differ.

## Commands

```bash
jer config validate configs/experiments/ttbar2L.yaml
jer config show configs/experiments/ttbar2L.yaml
jer inputs show configs/experiments/ttbar2L.yaml
jer inputs compare configs/studies/12_encoder_backbone.yaml
jer topology show configs/topology/ttbar2L.yaml
jer cache show configs/experiments/ttbar2L.yaml
jer cache build configs/experiments/ttbar2L.yaml
jer cache build configs/experiments/ttbar2L.yaml --force
jer checkpoint show checkpoint.ckpt
jer study plan configs/studies/12_encoder_backbone.yaml
```

The historical Study-01 reanalysis is under
`results/studies/visible_recurrence/_reanalysis/2026-09-02-4e33113`. It did not
retrain models and does not select a winner. Ordinary downstream studies inherit
the human-selected one-pass M1 assignment; recurrence is retained only in the
explicit Study-01 and Study-11 diagnostics. Check the current campaign with
`python scripts/submit_campaign.py --status` and release only an intentional
Paper-1 study, for example
`python scripts/submit_campaign.py --ready --studies invisible_coordinates`.
