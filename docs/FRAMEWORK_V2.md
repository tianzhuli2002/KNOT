# JER framework v2 contract

This document records the maintained v2 seams.  It is an implementation and
provenance contract, not evidence for any physics improvement.

## Shared latent and visible backbones

`model.latent` defines the width and default topology-reasoner capacity used by
assignment, feedback and joint-flow modules.  `model.encoder` selects exactly
one visible backbone: `transformer`, `physics_transformer`, `lgatr`, or
`lgatr_slim`.  Every backend returns the same `{pooled, tokens, padding_mask}`
interface with width `model.latent.d_model`.  The standard Transformer remains
the baseline.  L-GATr is optional (`pip install -e '.[lgatr]'`) and is validated
with the installed package version recorded in checkpoints.

Physics-aware backbones consume the token-aligned `physics_vectors` batch field,
whose representation is `(E, px, py, pz)` in GeV and whose padded rows are zero.
The scalar inputs are the explicitly configured physical scalars (`btag` and/or
`charge`).  MET is read from the one unpadded MET token; it is never recovered
from scaled embeddings.

## Invisible coordinates

Diffusion and CFM operate only on a coordinate state `[B,T,3]` and an explicit
active mask.  Prediction always decodes to Cartesian `(px, py, pz)` neutrino
momenta.  The maintained transforms are:

* `cartesian`: `T=R`, one Cartesian momentum token per invisible role;
* `met_residual`: role `pz` tokens, `R-1` Helmert transverse contrasts, and a
  transverse MET residual token;
* `met_exact`: role `pz` tokens and `R-1` contrasts, with transverse momentum
  projected to the measured MET by an equal shift across roles.

`met_residual` is exactly invertible for arbitrary truth and MET.  `met_exact`
is diagnostic: the original truth is retained as the primary target, while the
projected target and per-role projection residuals are written to prediction
products and reported separately.  Inactive state components remain zero in
targets, noise, velocities, integration and losses.

## Flow coupling

CFM training supports `independent`, `minibatch_ot`, and `conditional_ot`.
Coupling changes only source/target pairing; posterior sampling remains Gaussian
and does not perform OT.  Conditional OT uses the detached normalized condition
`[visible pooled, mean parent context]` and the configured non-negative weight.
The condition-aware method is labelled C2OT-style rather than as an exact claim
about an external implementation.

## Evaluation and calibration

Evaluation reports energy score, weighted finite-sample CRPS, discrete rank/PIT
diagnostics (not continuous PIT), coverage curves, composite observables,
conditional slices and offline reliability/selective-risk curves.  Adaptive
prediction-set calibration is separate from prediction and uses validation only;
it never changes model training or base posterior probabilities.  A finite top-K
candidate list reports truth-outside-support explicitly.

## Campaign gates

`configs/studies/UPSTREAM_SELECTIONS.yaml` records human-reviewed decisions.  No
winner is selected automatically.  The intended dependency is:

```text
Study 01 → assignment_reasoning → Study 12
Study 12 → visible_assignment → Study 02 / Study 05 / Study 11 / Analysis 14
Study 02 → invisible_generator → Study 13
Study 13 → invisible_coordinates → generator-dependent studies
Study 06 → joint_model → Study 07
```

Study 12 (encoder backbones) and Study 13 (invisible coordinates) are templates
only in this branch; they are not submitted automatically.

## Identity and execution

Training artifact identity includes the active architecture, training semantics,
model/data-subset identity, trainable components, C2 training sampler and any
initial checkpoint/component identity.  Prediction identity adds the checkpoint,
prediction seed, top-K/sample count, solver/NFE, chunking and reranking settings.
Runtime compilation and activation checkpointing are not architecture identity.

Slurm attempts snapshot the study manifest and record the submitted Git SHA and
base experiment checksums.  Workers hard-fail if the checkout, manifest snapshot,
or any base experiment changes before execution.
