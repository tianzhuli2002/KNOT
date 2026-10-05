# KNOT matched-production physics evaluation

## Sample and method

This report summarizes the completed matched KNOT-M/C/J production inference,
using the best-validation checkpoints from the frozen training campaign. The
evaluator read 10,000 common validation events and 32 posterior samples per
event. The ordered EVENT_ID SHA-256 is
`2f5256303663568bf2eacb64ed6b9375d95798ecda6ab9d728daea811428d647`; the
selection was drawn from the 100,342-event validation population using
`splitmix64_event_id_v1`, seed 42211. M, C and J files have identical ordered
IDs. Each posterior HDF5 also stores the pre-truncation physical jet count as
`diagnostics/physical_njet`, joined from the train+validation measured input by
exact EVENT_ID. Reserved test use is false. Event-resampling intervals below use 500
paired validation-event bootstrap replicas (seed 42561) and do not include
training-seed uncertainty.

Truth roles follow positive-lepton/top and negative-lepton/antitop sides.
Neutrino slots are top-side then antitop-side; parent targets are partonic
after-FSR. M sample slots are independent marginals. C and J slots are coherent
ancestral and joint event hypotheses, respectively.

## Reconstruction and posterior coherence

| Metric | M | C | J |
|---|---:|---:|---:|
| Complete assignment accuracy (supported) | 0.8438 | 0.8409 | 0.8461 |
| Assignment NLL (supported) | 0.3901 | 0.3919 | 0.3888 |
| Top / antitop neutrino pT RMSE [GeV] | 38.88 / 39.60 | 38.72 / 38.58 | 38.53 / 39.42 |
| Top / antitop neutrino pz RMSE [GeV] | 90.12 / 90.31 | 89.46 / 90.34 | 89.87 / 90.06 |
| Top / antitop parent pT RMSE [GeV] | 41.82 / 42.45 | 42.20 / 42.27 | 41.42 / 41.98 |
| (m_{t\bar t}) bias / RMSE [GeV] | +6.66 / 80.17 | -4.42 / 80.48 | +4.56 / 79.19 |
| (p_{T,t\bar t}) bias / RMSE [GeV] | +1.63 / 25.54 | -0.31 / 26.92 | +6.42 / 26.51 |
| (m_{t\bar t}) central 68% coverage | 0.689 | 0.660 | 0.628 |
| Coherent joint energy score [GeV] | 110.83 | 108.88 | 105.75 |
| Mean top-side N-P correlation | -0.0005 | 0.431 | 0.828 |
| Decay-chain ΔpT RMSE [GeV] | 73.90 | 63.71 | 60.30 |

The joint energy score evaluates the paired top/antitop neutrino and parent
longitudinal components; lower is better. Its paired 95% difference intervals
are C-M `[-2.314, -2.115]`, J-M `[-3.707, -3.462]`, and J-C
`[-3.731, -2.523]` GeV. Shuffling parent samples across neutrino sample index
changes the score by `-0.064` GeV for M (95% `[-0.123, 0.002]`), `+2.153`
GeV for C (`[2.067, 2.234]`), and `+3.526` GeV for J (`[3.415, 3.631]`). The
one-dimensional means and variances are preserved by the shuffle. The separate
assignment-to-continuous pairing shuffle increases decay-chain ΔpT RMSE by
`1.925` GeV for C and `2.374` GeV for J; the M change is consistent with zero.
These tests confirm learned sample-wise dependence in C/J and its absence in M.

Point reconstruction is broadly similar, not uniformly improved by
conditioning: assignment metrics are nearly identical, C's (m_{t\bar t})
RMSE is slightly above M, and J's is slightly below M with the paired J-M 95%
interval `[-1.975, 0.064]` GeV. The joint/coherence conclusion is separate
from these marginal metrics. J-C 95% event-bootstrap intervals exclude zero
for both joint energy score (`[-3.731,-2.523]` GeV) and (m_{t\bar t}) RMSE
(`[-2.287,-0.218]` GeV); they support J_CLEARLY_BETTER_THAN_C for this
checkpoint, without accounting for training-seed uncertainty.

The v10 report also records neutrino eta/phi and top/antitop rapidity/phi
residuals. Azimuth point estimates are circular posterior means and residuals
are wrapped to [-pi,pi). The (nu-top eta, nu-antitop eta, top y, antitop y)
RMSE values are `0.978, 0.968, 0.277, 0.274` for M;
`0.973, 0.966, 0.278, 0.277` for C; and
`0.978, 0.966, 0.273, 0.272` for J. Neutrino phi RMSE is
`1.279/1.265`, `1.279/1.268`, and `1.283/1.271` for M/C/J, respectively.

## Assignment ambiguity, calibration and tails

Events are divided into four common bins of mean M/C/J assignment entropy,
with 2,499, 2,499, 2,499 and 2,500 events. Joint energy score (M/C/J, GeV) is:

| Common-entropy bin | M | C | J |
|---|---:|---:|---:|
| Lowest | 100.93 | 99.72 | 95.06 |
| Low-mid | 102.40 | 100.58 | 97.20 |
| High-mid | 114.94 | 112.38 | 109.59 |
| Highest | 125.06 | 122.86 | 121.14 |

C/J improve this joint metric in all bins, but their advantage does not grow
monotonically with ambiguity; it narrows in the highest-entropy bin. The
(m_{t\bar t}) RMSE differences across bins are not consistently resolved by
paired bootstrap intervals.

For (m_{t\bar t}), 68% coverage is 0.689/0.660/0.628 for M/C/J; the top-side
neutrino pz coverage is 0.661/0.623/0.627. Thus M is near nominal at the
system level while C/J under-cover. Posterior width increases with absolute
error across the (m_{t\bar t}) quintiles, but rank-PIT summaries show
remaining calibration structure; this is not a claim of fully calibrated
posteriors.

The truth-decile tail starts at (m_{t\bar t}=763.97\) GeV (1,000 events).
All models show negative mean residuals there: M `-66.3` GeV, C `-87.4` GeV,
J `-60.5` GeV. RMSE is 167.3/175.2/164.6 GeV and 68% coverage is
0.622/0.559/0.546. The differential residuals transition from positive at
low mass to negative at high mass, consistent with high-mass shrinkage.
Diagonal retention in eight truth-quantile migration bins is 0.379/0.399/0.393
for M/C/J. The three dilepton channels have similar (m_{t\bar t}) RMSE
(M: 80.4, 82.1, 78.7; C: 79.2, 83.3, 79.2; J: 76.7, 81.5, 78.7 GeV for
ee, mumu, emu).

## KNOT-C context oracle ladder

Parent pT RMSE for the three endpoints is:

| Endpoint | Top [GeV] | Antitop [GeV] |
|---|---:|---:|
| (P\mid A_{truth},N_{truth}) | 36.25 | 36.32 |
| (P\mid A_{pred},N_{truth}) | 40.60 | 40.89 |
| (P\mid A_{pred},N_{pred}) | 42.20 | 42.27 |

The assignment-context step adds 4.35/4.57 GeV in top/antitop pT RMSE; the
predicted-neutrino step adds a further 1.60/1.38 GeV. Assignment context is
the larger measured degradation. Parent-mass RMSE remains near 6 GeV across
these endpoints. No conditioning remedy is implemented here.

## Hybrid spin closure

The spin check uses the agreed hybrid benchmark: after-FSR truth parents with
measured charge-ordered leptons, because generator-level charged-lepton truth
is unavailable. The evaluator boosts lab → (t\bar t) rest → corresponding
parent rest with rotation-free boosts; each helicity axis is its own parent's
direction in the pair rest frame, including the antitop without an extra sign.
The opening angle uses the two lepton directions in the common pair rest frame
before the parent boosts. Δφ is the absolute wrapped lab azimuth difference.

For M/C/J respectively, hybrid-truth means are Δφ `1.874` for all;
(\cos\theta_{\ell^+}) `0.090`, (\cos\theta_{\ell^-}) `0.099`, helicity
product `-0.015`, and opening cosine `-0.371`. Sample-slot means are below;
only C/J have coherent event hypotheses.

| Observable | M | C | J |
|---|---:|---:|---:|
| (\Delta\phi_{\ell\ell}) | 1.874 | 1.874 | 1.874 |
| (\cos\theta_{\ell^+}) | -0.341 | -0.268 | -0.338 |
| (\cos\theta_{\ell^-}) | -0.344 | -0.268 | -0.330 |
| Helicity product | 0.184 | 0.133 | 0.176 |
| Opening cosine | -0.312 | -0.321 | -0.309 |

Shuffling sample pairing leaves lab Δφ unchanged, as expected. It shifts
the reconstructed helicity, product and opening-angle distributions by many
event-sampling standard errors, so these are sensitive to coherent pairing in
C/J. M's same-index draws remain marginal and are not interpreted as coherent
hypotheses.
Their means remain biased relative to the hybrid truth benchmark; this verifies
pairing sensitivity and does not establish calibrated spin physics.

## Computation and decision

| Variant | Stored / trainable parameters | GPU training [s] | Training presentations/s | Posterior inference [s] | Peak CUDA allocation [GB] |
|---|---:|---:|---:|---:|---:|
| M | 2,775,317 / 2,741,309 | 10,043 | 3,584 | 71.94 | 1.250 |
| C | 2,775,317 / 2,758,339 | 11,342 | 3,174 | 91.11 | 1.255 |
| J | 2,954,891 / 1,550,262 | 8,240 | 4,368 | 29.69 | 1.242 |

The total J parameter count is 6.47% above C. For this fixed seed, the
posterior evidence supports **KNOT_MECHANISM_CONFIRMED** and the event-bootstrap
joint-score ordering is **J_CLEARLY_BETTER_THAN_C**. This is a measured
single-checkpoint difference, not a claim that J is more expressive than C:
finite capacity, optimization, inductive bias and the observed sequential
error propagation remain possible causes. Training-seed uncertainty is not
covered by the event bootstrap. A matched multi-seed replication is justified
before treating the J-C ordering as stable. The production recommendation is
**KNOT_FULL_GPU_CAMPAIGN_WITH_EVALUATOR_UPDATE_ONLY**; the campaign itself is
complete, and no new jobs were submitted during this evaluation.

The measured J-C ordering and the larger truth-assignment-to-prediction parent
oracle degradation justify **RUN_MATCHED_SEEDS_NEXT** before treating the
ordering as stable. No additional campaign is submitted here. A harmonized
historical JER retraining remains a useful later control, but it is less direct
than replication of the observed matched M/C/J difference.

Historical JER source semantics are documented in
[`HISTORICAL_JER_VS_KNOT_M.md`](HISTORICAL_JER_VS_KNOT_M.md). Given the C oracle
gap, a harmonized retrained old-JER baseline could isolate deterministic
visible-plus-neutrino composition from learned parent reconstruction. It is a
useful later control, but the matched-seed replication is the more direct next
KNOT action.

## Machine outputs

Full metrics, paired bootstrap intervals, migration arrays, spin metadata and
provenance are in
`results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/physics-analysis-v11/physics_summary.json`.
The oracle endpoints are in the neighboring `oracle_summary.json`; the figures
and the frozen validation selection are in that same `physics-analysis-v11/`
directory. Production posterior HDF5 files are under the sibling `M/`, `C/`
and `J/` directories.
