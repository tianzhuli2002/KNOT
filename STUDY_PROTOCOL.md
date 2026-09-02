# Controlled study protocol

This document is the operational protocol for the maintained JER campaign. It
separates what is implemented and mechanically validated from what is established
only by a controlled physics comparison.

## Common rules

- `experiment.seed` controls model/training randomness.
- `data.subset_seed` fixes the EVENT_ID-based, nested train subset.
- `prediction.seed` controls posterior sampling and is inference-only.
- Studies 01--11 select configurations on `validation`. The held-out `test`
  split is evaluated once, after the relevant model and inference choices are
  locked.
- Compared products must contain the same ordered EVENT_ID sequence. Report
  native assignment support and, when supports differ, the common truth
  non-NULL subset separately.
- Downstream manifests are explicit templates. Their assignment/generator
  reference patches remain provisional until the human-reviewed gate in
  `configs/studies/UPSTREAM_SELECTIONS.yaml` is locked and the matching patch is
  updated.
- A mechanism being implemented or passing a smoke test is not evidence that it
  improves physics performance.

## Study 01 — assignment reasoning

**Question:** Does recurrent topology reasoning improve complete structured
assignment beyond a one-pass baseline?

**Prerequisite:** validated canonical ttbar2L data; no upstream scientific gate.

**Primary variants:** one-pass structured assignment; recurrent assignment
refinement; recurrent assignment refinement plus soft feedback. The fixed
parameter-matched and training-compute-matched non-recurrent controls are
required diagnostics, not optional conveniences.

**Selection and metrics:** validation complete-assignment top-1 accuracy on the
intended common support, with top-k coverage, truth rank, retained mass, per-role
accuracy, Njet/ambiguity stratification, paired transitions and bootstrap
intervals. Report parameters, training/prediction time and memory.

**Depths:** R=1,2,3,4 are primary. R=8 is an explicitly labelled diagnostic
extrapolation/overthinking test; it is not a trained checkpoint or a primary
depth choice.

**Decision:** adopt recurrence only when the gain survives the capacity and
training-compute controls. Confirm only decisive finalists with seeds 42, 123
and 456 on the same fixed event subsets. Propagate the human-selected assignment
reference to Study 02 and later studies. This study cannot establish that C2,
generator formulation, jointness or compatibility helps.

## Study 02 — invisible generator formulation

**Question:** Which invisible posterior formulation gives the best calibrated
reconstruction for a reasonable inference cost?

**Prerequisite:** locked Study 01 assignment reference.

**Primary variants:** DDIM/standard diffusion, EDM, CFM and OT-CFM, trained and
evaluated with the same frozen assignment reference and validation event IDs.

**Diagnostics and metrics:** energy score, component CRPS, finite-sample
rank/PIT diagnostics, coverage curves, composite top-object posterior metrics,
physics residuals and effective network evaluations/NFE. Report quality and
cost together.

**Decision:** select a generator only on validation, subject to calibration and
cost checks. Confirm the selected finalist with additional seeds before locking
it for Study 03. This study does not compare assignment architectures.

## Study 03 — assignment to invisible conditioning

**Question:** Does assignment information improve invisible reconstruction?

**Prerequisite:** locked assignment and generator references.

**Primary variants:** none; soft; predicted-hypothesis K=1, M=32 (controlled
comparison); predicted-hypothesis K=5, M=32 (practical factorised posterior).

**Diagnostics:** hard local assignment conditioning and truth-assignment oracle.
The oracle is headroom only and is never a deployable result. Report hypothesis
training valid fraction and common-support dependence.

**Decision:** use the paired validation comparison and the oracle gaps to decide
whether conditioning is useful. Propagate the selected conditioning policy;
do not conflate predicted hypotheses with truth oracle.

## Study 04 — invisible posterior to assignment feedback

**Question:** Can invisible-posterior information resolve visible assignment
ambiguity?

**Prerequisite:** locked assignment, generator and conditioning references.

**Primary comparison:** continued assignment training without posterior feedback
versus continued assignment training with detached C2 feedback. Both branches
start from the same C1 checkpoint and receive the same epochs, optimiser and LR
schedule. The pre-continuation C1 product is a diagnostic baseline only.

**C2 contract:** its training sampler, posterior sample count and formulation
are explicit training semantics; prediction sampler settings are inference-only.
Report paired wrong-to-correct/correct-to-wrong transitions and calibration;
do not claim a C2 gain from the unpaired C1-to-C2 change.

## Study 05 — discriminative versus DFM assignment

**Question:** Is complete-state generative assignment useful relative to the
strongest discriminative structured assignment?

**Prerequisite:** locked assignment reference; the selected assignment model is
updated explicitly after Study 01.

**Primary variants:** selected discriminative assignment versus complete-state
DFM. One-pass is a diagnostic where useful.

**Metrics and rule:** report native-support metrics and a common truth-non-NULL
subset; never place incompatible support denominators on one misleading axis.
Check DFM posterior-mode stability versus sample count before interpretation.
This study determines whether a factorised DFM-to-conditional-invisible
baseline is needed in Study 06. It cannot by itself establish a jointness gain.

## Study 06 — factorised versus joint inference

**Question:** Does native joint discrete-continuous inference improve over a
strong controlled factorisation?

**Prerequisite:** locked assignment and generator references, plus the Study 05
decision about whether a factorised DFM baseline is required.

**Primary variants:** factorised and native joint DFM+CFM from a common visible
reference state, with comparable training exposure. A warm-started joint model
is diagnostic only.

**Controls:** match loss budget, data IDs, inference settings and report the
assignment-formulation difference. Do not claim a jointness effect if the result
could be only a DFM-versus-discriminative assignment effect.

Before interpreting this comparison, record representative assignment and
invisible loss scales together with shared gradient norms on fixed batches.
Retain the simple 1:1 loss weighting unless there is an order-of-magnitude
imbalance; any loss-weight scan is validation-only.

## Study 07 — persistent joint recurrence

**Question:** Does a persistent topology-shaped state improve a plain joint flow?

**Prerequisite:** a validation decision that the joint model is worth studying.

**Variants:** stateless joint flow versus persistent-state joint flow, with
matched initialisation, training budget and inference settings. Report added
parameters, network calls, runtime and memory. This study cannot rescue a joint
model that Study 06 does not justify.

## Study 08 — generator inference budget

**Question:** What quality-cost point is sufficient at inference without
retraining?

**Prerequisite:** locked assignment, generator and conditioning policies.

**Variants:** formulation-appropriate NFE/step scans; CFM Euler/Heun comparisons
use actual network-call accounting. Select the validation Pareto point and lock
it before test evaluation. This is an inference scan, not a training comparison.

## Study 09 — posterior size and convergence

**Question:** How many assignment hypotheses and invisible samples are needed?

**Prerequisite:** locked factorised/joint model policy and generator.

**Design:** generate one factorised K=10, M=64 product and one joint S=64
product. Evaluate deterministic nested prefixes offline: K=1,2,3,5,10;
M=1,4,8,16,32,64; S=1,4,8,16,32,64. Renormalise retained branch
probabilities for each K and use stable sample prefixes. Report energy score,
CRPS, coverage/rank diagnostics, retained assignment mass and uncertainty.

**Decision:** lock the smallest validation point with adequate quality and
stability. Never compare scans generated with different random posterior draws.

## Study 10 — compatibility reranking

**Question:** Does learned visible/invisible compatibility improve selection
among already-generated hypotheses?

**Prerequisite:** locked upstream assignment, generator and posterior size.

**Design:** generate one posterior at compatibility weight zero and scan weights
offline. Report assignment-only and reranked top-1, paired transitions and
bootstrap intervals. The compatibility score is a learned compatibility logit,
not a normalized likelihood. Select the weight on validation and evaluate the
locked choice on test only afterward.

## Study 11 — data scaling

**Question:** Is final performance architecture-limited or data-limited?

**Prerequisite:** all relevant upstream architecture and inference decisions
locked.

**Design:** train 10k, 25k, 50k, 100k, 250k and 500k nested EVENT_ID subsets;
validation and test sets remain fixed. The primary curve uses a sufficiently
high max-epoch bound, best validation checkpoint and real early stopping. A
fixed-compute curve is optional and must be labelled separately.

**Decision:** describe scaling only after checking convergence and subset
identity. Do not attribute a data-size trend to architecture without holding
the selected model and inference policy fixed.

## Final held-out evaluation

After the human decision gates are locked, update the explicit downstream
reference patches, record the validated source commit in
`UPSTREAM_SELECTIONS.yaml`, and run the selected models on `test` once. Test
metrics are final held-out evaluation, not a further selection loop.

External ReconstructionAndSB/VyPER/conventional kinematic baselines and a
later ttbar1L/ttH1L compact generalisation check are planned before publication;
they are not part of the current mechanism-discovery campaigns.

## v2 framework extensions

Study 12 (`configs/studies/12_encoder_backbone.yaml`) is the controlled visible
backbone template: Transformer, physics Transformer, L-GATr and L-GATr-Slim all
share the resolved `model.latent` and topology-reasoner contract. It requires
the human-reviewed `assignment_reasoning` gate and does not select a winner by
argmax in code.

Study 13 (`configs/studies/13_invisible_coordinates.yaml`) compares Cartesian,
MET-residual and diagnostic MET-exact states after the upstream assignment and
generator decisions. MET-residual preserves the physical truth exactly;
MET-exact keeps the original truth as the primary metric target and reports the
equal-shift projected target and residual separately. It is a template only.

Analysis 14 is the offline validation-split APS assignment-set calibration in
`src/joint_event_reconstruction/calibration/`; calibration never changes model
training or prediction probabilities and test data is rejected as a fit source.

The maintained human gates are `assignment_reasoning`, `visible_assignment`,
`invisible_generator`, `invisible_coordinates`, and `joint_model` in
`configs/studies/UPSTREAM_SELECTIONS.yaml`. The scientific order is Study 01 →
assignment reasoning → Study 12 → visible assignment; Study 02 and Study 05/11
then become available as their individual prerequisites are reviewed. No gate
is locked automatically and no v2 study is submitted by this framework pass.

Study 01 historical products may use architecture signature v4. Study 12/v2
uses signature v5 and trains from scratch under its selected visible-backbone
configuration; a human decision transfers the reasoning choice rather than
blindly loading an incompatible full checkpoint.
