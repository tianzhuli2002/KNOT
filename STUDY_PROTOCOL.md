# Controlled study protocol

This document is the operational protocol for the maintained JER campaign. It
separates what is implemented and mechanically validated from what is established
only by a controlled physics comparison.

## Common rules

- `experiment.seed` controls model/training randomness.
- `data.subset_seed` fixes the EVENT_ID-based, nested train subset.
- `prediction.seed` controls posterior sampling and is inference-only.
- All scientific gates and maintained studies select configurations on
  `validation`. The held-out `test`
  split is evaluated once, after the relevant model and inference choices are
  locked.
- Compared products must contain the same ordered EVENT_ID sequence. Report
  native assignment support and, when supports differ, the common truth
  non-NULL subset separately.
- Downstream manifests resolve the explicit human gates in
  `configs/studies/UPSTREAM_SELECTIONS.yaml`. Configuration inheritance and
  checkpoint initialisation are separate; a pending gate hard-blocks submission.
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
training-compute controls. The completed validation campaign locks
`assignment_reasoning=m1`; this does not establish that recurrence is useless.
Propagate the human-selected assignment reference to Study 02 and later studies.
This study cannot establish that C2, generator formulation, jointness or
compatibility helps.

## Study 02 — invisible generator formulation

**Question:** Which invisible posterior formulation gives the best calibrated
reconstruction for a reasonable inference cost?

**Prerequisite:** locked `visible_assignment` gate and its exact per-seed
visible/assignment checkpoints.

**Primary variants:** DDIM/standard diffusion, EDM, CFM, OT-CFM and
condition-aware OT-CFM, trained for three model seeds with the same frozen M1
visible/assignment system and validation event IDs.

**Diagnostics and metrics:** energy score, component CRPS, finite-sample
rank/PIT diagnostics, coverage curves, composite top-object posterior metrics,
physics residuals and effective network evaluations/NFE. Report quality and
cost together.

**Decision:** select a generator only on validation, subject to calibration and
cost checks. Confirm the selected finalist with additional seeds before locking
it for Study 03. This study does not compare assignment architectures.

## Study 03 — assignment to invisible conditioning

**Question:** Does assignment information improve invisible reconstruction?

**Prerequisite:** locked `visible_assignment`, `assignment_formulation`,
`invisible_generator` and `invisible_coordinates` gates.

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

**Prerequisite:** locked `visible_assignment`, `assignment_formulation`,
`invisible_generator`, `invisible_coordinates` and `assignment_conditioning`
gates.

**Primary comparison:** continued assignment training without posterior feedback
versus continued assignment training with detached C2 feedback. Both branches
start from the same C1 checkpoint and receive the same epochs, optimiser and LR
schedule. The pre-continuation C1 product is a diagnostic baseline only.

**C2 contract:** the selected one-pass assignment is continued with either no
posterior information or one explicitly defined detached posterior correction.
Its training sampler, posterior sample count and formulation are explicit
training semantics; prediction sampler settings are inference-only. Report
paired wrong-to-correct/correct-to-wrong transitions and calibration; do not
claim a C2 gain from the unpaired C1-to-C2 change.

## Study 05 — discriminative versus DFM assignment

**Question:** Is complete-state generative assignment useful relative to the
strongest discriminative structured assignment?

**Prerequisite:** locked `visible_assignment`; the formulation comparison is
the selected-encoder M1 baseline versus complete-state DFM.

**Primary variants:** selected one-pass discriminative assignment versus
complete-state DFM. The old M3 recurrent-feedback baseline is not a primary
variant.

**Metrics and rule:** report native-support metrics and a common truth-non-NULL
subset; never place incompatible support denominators on one misleading axis.
Check DFM posterior-mode stability versus sample count before interpretation.
This study determines whether a factorised DFM-to-conditional-invisible
baseline is needed in Study 06. It cannot by itself establish a jointness gain.

## Study 06 — factorised versus joint inference

**Question:** Does native joint discrete-continuous inference improve over a
strong controlled factorisation?

**Prerequisite:** locked `visible_assignment`, `assignment_formulation`,
`invisible_generator`, `invisible_coordinates`, `assignment_conditioning` and
`invisible_feedback` gates.

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

**Prerequisite:** a locked `joint_model` gate. If it selects `factorized`, this
study is explicitly `NOT APPLICABLE`, not an invented joint comparison.

**Variants:** stateless joint flow versus persistent-state joint flow, with
matched initialisation, training budget and inference settings. Report added
parameters, network calls, runtime and memory. This study cannot rescue a joint
model that Study 06 does not justify.

## Study 08 — generator inference budget

**Question:** What quality-cost point is sufficient at inference without
retraining?

**Prerequisite:** locked visible assignment for the candidate Study-02
checkpoints. It is an inference-only scan and may be submitted with `afterok`
dependencies on known Study-02 candidate training jobs before the generator
winner gate is locked.

**Variants:** formulation-appropriate NFE/step scans; CFM Euler/Heun comparisons
use actual network-call accounting. Select the validation Pareto point and lock
it before test evaluation. This is an inference scan, not a training comparison.

## Study 09 — posterior size and convergence

**Question:** How many assignment hypotheses and invisible samples are needed?

**Prerequisite:** locked factorised/joint model policy and selected generator;
the maximum posterior is generated once and all reported prefixes are offline.

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

**Prerequisite:** locked upstream assignment, generator, conditioning and
coordinate policies.

**Design:** generate one posterior at compatibility weight zero and scan weights
offline. Report assignment-only and reranked top-1, paired transitions and
bootstrap intervals. The compatibility score is a learned compatibility logit,
not a normalized likelihood. Select the weight on validation and evaluate the
locked choice on test only afterward.

## Study 11 — data scaling

**Question:** Is final performance architecture-limited or data-limited?

**Prerequisite:** locked visible assignment for the primary M1 scaling curve.

**Design:** train M1 at 10k, 25k, 50k, 100k, 250k and 500k nested EVENT_ID
subsets; validation and test sets remain fixed. Retain only the limited,
clearly labelled M3 diagnostic at 100k, 250k and 500k to test data hunger; it
does not reopen Study 01. The primary curve uses a sufficiently high max-epoch
bound, best validation checkpoint and real early stopping.

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

## Final framework extensions

Study 12 (`configs/studies/12_encoder_backbone.yaml`) is the next eligible
controlled visible-backbone campaign: transformed-feature Transformer,
four-vector Transformer, relational Transformer, L-GATr and L-GATr-Slim all
share the resolved `model.latent`, physical inputs and one-pass M1 assignment.
It requires the human-reviewed `assignment_reasoning` gate and does not select
a winner by argmax in code.

Study 13 (`configs/studies/13_invisible_coordinates.yaml`) compares Cartesian,
MET-residual and diagnostic MET-exact states after the upstream assignment and
generator decisions. MET-residual preserves the physical truth exactly;
MET-exact keeps the original truth as the primary metric target and reports the
equal-shift projected target and residual separately. It is held behind the
generator gate.

Analysis 14 is the offline validation-split APS assignment-set calibration in
`src/joint_event_reconstruction/calibration/`; calibration never changes model
training or prediction probabilities and test data is rejected as a fit source.

The maintained human gates are `assignment_reasoning`, `visible_assignment`,
`assignment_formulation`, `assignment_conditioning`, `invisible_feedback`,
`invisible_generator`, `invisible_coordinates`, and `joint_model` in
`configs/studies/UPSTREAM_SELECTIONS.yaml`. The scientific order is Study 01 →
assignment reasoning → Study 12 → visible assignment; Study 02, Study 05 and
Study 11 then become available in parallel. No gate is locked automatically.
`scripts/submit_campaign.py --ready` submits every currently eligible study
once, records exact source identities and scheduler dependencies, and leaves
scientific winner selection to the human gate.

Study 01 historical products may use architecture signature v4. Study 12
uses signature v5 and trains from scratch under its selected visible-backbone
configuration; a human decision transfers the reasoning choice rather than
blindly loading an incompatible full checkpoint.
