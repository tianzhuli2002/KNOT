# KNOT implementation surface

Source: `ec54f6b13817009b5ec0472e084a034f044980a4`, the clean
`research/realnu-mass-response-audit-v1` worktree. Isolated branch:
`research/knot-implementation-v1`. Existing production files/results/jobs were
not changed. No sealed test data is used by the development commands.

## Model and posterior

`KnotModule` extends the maintained Lightning module and keeps its L-GATr
encoder, compiled topology, structured candidate encoders, feasible complete
assignment states, query block, invisible coordinate codec/scaler, parent
feature/scaler convention, component/checkpoint metadata and optimizer.

- **M:** existing assignment objective and invisible CFM with assignment
  conditioning `none`; maintained conventional P0 parent CFM. Exported sample
  slots pair independent marginal draws. They are explicitly labelled
  `independent_marginals`; they do not establish posterior correlations.
- **C:** sample feasible complete assignment states with `torch.multinomial`;
  gather candidate-or-NULL embeddings for that exact state; condition the
  invisible CFM on those role contexts. The maintained P2 parent velocity model
  receives the same hard assignment and physical invisible draw. Flattening
  `[event,sample]` happens once, in the same order, for every conditioner.
- **J:** discrete assignment remains separate. A single CFM transports the
  active scaled invisible state and all configured parent states. The reused
  attention denoiser allows interaction among every continuous state token.
  No invisible or parent sampler is executed separately in J.

`EventPosterior` contains EVENT_ID, explicit sample indices, complete assignment
state and node candidate indices, physical invisible/parent p4s, topology/role
names and support masks. Continuous outputs are `[event,sample,role,4]`, in
`(E,px,py,pz)` GeV. Assignment support denotes non-NULL visible support;
continuous support denotes the existence of the configured latent role, not
availability of truth labels or successful detector reconstruction. Candidate
indices refer to the maintained deterministic topology candidate pools for the
measured event. Store the measured inputs/config with exported posteriors.
Pair mass and pT are derived from the sampled parents when writing HDF5.

The invisible MET residual/contrast coordinates are **not** physical role p4s.
Their deterministic/inactive entries are omitted from J's transported vector,
then restored to zero before using the existing inverse scaler and codec.
Parents use the existing scaled `(px,py,pz,log1p(m))` coordinates. There is no
independent parent-pair target. Both codecs/scalers remain in checkpoint
metadata. Nonfinite physical posterior outputs fail validation.

The research YAML sets `training_conditioning: truth` or `predicted` explicitly.
Truth teacher conditioning applies only in training. Predicted training uses
sampled complete assignments and (in C) generated invisible momenta; it does
not backpropagate through the discrete draw/ODE. Prediction is always ancestral
and never reads truth assignment, momenta or support. All variants encode the
measured batch once before posterior expansion and solver calls.

## Existing P1/P2 audit

Historical P1 marginalizes candidate embeddings over feasible complete
assignment probabilities. P2 conditions on those soft summaries and **one**
predicted invisible sample per event, reused across its parent sample slots.
It exposes invisible index zero for every parent draw. This is conditional
parent generation, but is not KNOT-C's sample-specific `A -> N -> P` ancestry.
Those production paths are retained. The only parent-generator extension adds
an optional explicit pair of role names; its existing default and parameter
shapes are preserved.

## Configuration and bounded commands

Use the existing public JER YAML for encoder/topology/data/optimizer settings
and the small `configs/research/knot_{m,c,j}.yaml` for factorisation, conditioning,
solver and objective weights. Their corresponding `*_jer.yaml` files point only
to this worktree's synthetic development experiment. No new configuration
migration or checkpoint format is introduced.

From this worktree:

```bash
PYTHONPATH=src:. python scripts/validate_knot.py
PYTHONPATH=src python scripts/run_knot.py train \
  --config configs/research/knot_c_jer.yaml \
  --research configs/research/knot_c.yaml \
  --output results/experiments/knot/c_train --steps 2
PYTHONPATH=src python scripts/run_knot.py predict \
  --config configs/research/knot_c_jer.yaml \
  --research configs/research/knot_c.yaml \
  --checkpoint results/experiments/knot/c_train/model.ckpt \
  --output results/experiments/knot/c_predict --samples 3 --batches 1
```

The bounded runner requires a fresh project-specific output directory and an
explicit train/validation-only HDF5. It reloads the checkpoint scalers and
builds a measured-only prediction adapter. Validation creates 32 synthetic
events (24 train, 8 validation), fits scalers on the selected training population
and takes two optimizer steps per variant. Artifacts and JSON evidence live in
`results/experiments/knot/development/`. They demonstrate plumbing, not physics
performance. Re-running the validation script refreshes its own development
checkpoints and reports; use a copied isolated config/output for distinct runs.

## Current limits

One-pass discriminative assignment only: recurrent feedback, assignment flows,
and geometric parent backbones are rejected by this surface. Parent targets
must be precisely the compiled roots. M/C reuse the maintained two-parent
conventional CFM; C also requires exactly one invisible role per parent and
orders those roles explicitly and requires fixed invisible masses. Other parent multiplicities are rejected.
KNOT currently requires shared encoder/query gradient policies set to `both`. J derives its state
from the configured scaler/compiled roles. Only the dilepton synthetic topology
has been exercised. Scheduled mixtures of teacher/predicted conditioning and
real-data scientific comparisons remain subsequent studies.
