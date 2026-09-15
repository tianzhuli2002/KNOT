# JER framework contract

This document records the maintained framework seams. It is an implementation and
provenance contract, not evidence for any physics improvement.

## Shared latent and visible backbones

`model.latent` defines the width and default topology-reasoner capacity used by
assignment, feedback and joint-flow modules.  `model.encoder` selects exactly
one visible backbone. Transformer representation and relations are orthogonal:
`features/none`, `four_vector/none`, or `four_vector/physics`. L-GATr and
L-GATr-Slim require four-vectors implicitly. Every backend returns the same `{pooled, tokens, padding_mask}`
interface with width `model.latent.d_model`.  The standard Transformer remains
the baseline.  L-GATr is optional (`pip install -e '.[lgatr]'`) and is validated
with the installed package version recorded in checkpoints.

Physics-aware backbones consume the token-aligned `physics_vectors` batch field,
whose representation is `(E, px, py, pz)` in GeV and whose padded rows are zero.
The scalar inputs are the explicitly configured physical scalars (`btag` and/or
`charge`).  MET is read from the one unpadded MET token; it is never recovered
from scaled embeddings.

## Input and cache boundary

Public configuration has four physics-facing parts: `data` selects one canonical
HDF5 and one optional cache destination; `inputs` selects collections, counts,
features and compact transforms; `topology` names semantic roles and decays; and
`model` selects reconstruction mechanisms. Storage label indices are owned by
the maintained dataset adapter.

The deterministic path is `HDF5 -> tensor cache -> thin DataModule -> GPU`. One
plain PyTorch `.pt` cache contains transformed scalar inputs, physical
`(E,px,py,pz)` vectors in GeV with metric `(+---)`, type IDs, masks, GLOBAL
features, EVENT_ID, canonical split identity, visible truth and invisible truth.
It stops before learned computation and is shared by every maintained encoder and
reconstruction formulation. A fingerprint in metadata validates its contents;
the filename remains `<input_stem>.pt`. Missing caches build automatically,
incompatible caches fail, and replacement requires an explicit force option.

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
Study 12 → visible_assignment → Study 02 / Study 05 / Study 11
Study 02 + Study 08 → invisible_generator + inference budget
Study 13 → invisible_coordinates
Study 15 → invisible_backbone
Study 10 → assignment_coupling
optional slim Study 03 → one reverse assignment-conditioned pass
Study 12 predictions → Analysis 14 (offline; no encoder selection)
```

The historical branches `assignment_conditioning → Study 04`, the joint-model
comparison in Study 06, and recurrent Study 07 remain available as follow-up
research. They are not Paper-1 dependencies.

configs/studies/UPSTREAM_SELECTIONS.yaml is the small human gate registry.
Locked gates contain an explicit selected patch and, where weights are reused,
an exact per-topology/per-seed checkpoint identity. Pending gates hard-block
real submission; not_applicable is reported explicitly. Configuration
inheritance and checkpoint initialisation are separate manifest fields.

The Paper-1 critical path is intentionally shorter:

```text
Study 02 + Study 08
    ↓ generator objective + CFM inference budget
Study 13
    ↓ invisible_coordinates
Study 15
    ↓ invisible_backbone
Study 10
    ↓ assignment_coupling
optional slim Study 03
    ↓ one possible refined-assignment → invisible pass
freeze the model, then evaluate the sealed test split once
```

Study 10 compares the visible baseline, a matched visible-only control, analytic
physics compatibility, a small learned correction, and a truth-neutrino oracle.
Studies 04, 06 and 07 remain documented follow-up work and are not released by
the Paper-1 campaign command. `python scripts/submit_campaign.py --ready`
supports an explicit study allowlist for this reason. The Study-10 physics score
is an interpretable unnormalised compatibility/evidence-like score, not a
normalised likelihood; its feature scaler is fit on the training split only.

The optional `HYPOTHESES` prediction group stores padded top-K support and
physics summaries without invalidating historical prediction files that do not
contain it. For an unconditioned invisible posterior, one generated posterior
is broadcast over all assignment hypotheses; generator cost therefore scales
with posterior samples rather than `K × samples`.

`python scripts/submit_campaign.py --status` reports every maintained study,
while `--ready --studies study_name,...` releases only the named eligible
studies. Scientific winner selection remains a human-gated validation decision.
scripts/run_study.py runs Analysis 14 on paired Study-12 validation predictions;
it is a review diagnostic, never an encoder-selection rule.

## Identity and execution

Training artifact identity includes the active architecture, training semantics,
model/data-subset identity, trainable components, C2 training sampler and any
initial checkpoint/component identity.  Prediction identity adds the checkpoint,
prediction seed, top-K/sample count, solver/NFE, chunking and reranking settings.
Runtime compilation and activation checkpointing are not architecture identity.

Slurm attempts snapshot the study manifest and record the submitted Git SHA and
base experiment checksums.  Workers hard-fail if the checkout, manifest snapshot,
or any base experiment changes before execution.
