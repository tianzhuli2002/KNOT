# Real ttbar CPU pilot

This follow-up starts at `92b42e8b995158a61046ee8e168261e226b763dc`.
The scientific-closure production configurations remain unchanged. The pilot
uses their full measured L-GATr, topology queries, assignment network and flows.
CPU changes are batch size 128, float32, unfused AdamW and zero loader workers.
The driver fixes training to 600 optimizer steps, independently of the inherited
20-epoch production setting. No early stopping, seed scan or capacity matching
is used. The public pilot configs describe these overlays; the resolved config,
research variant, budget and training history record the actual execution.

Fifty-step timings for each full architecture selected a matched 600-step
budget. The same deterministic EVENT_ID population contains 25,000 train and
5,000 validation events. The measured preprocessing and target scalers remain
fitted on the frozen full training membership. All three common initial states
match the production initialization hash. Batch order and per-step RNG seeds
are matched. Six validation checks use the same fixed 1,024-event prefix and
fixed flow-loss RNG; training always continues to step 600.

The best checkpoint criterion is assignment loss plus continuous MSE averaged
over all active dimensions: six invisible and eight parent dimensions for M/C,
or the same fourteen-dimensional joint MSE for J. Final and best checkpoints
are retained; posterior physics below uses final checkpoints. Their training
loss totals are not interchangeable architecture-ranking metrics.

Posterior evaluation uses a separate deterministic selection of 1,024 events
from the pilot validation membership, with 16 draws each and eight Heun steps.
The measured-only prediction batch is stripped of truth. Evaluation joins
truth afterwards; explicit C oracle calls alone receive truth conditioning.
No reserved test rows are loaded. The spin benchmark is explicitly hybrid:
authoritative truth parents with measured charge-ordered leptons, as approved
by the user. It is not a generator-level charged-lepton benchmark.

Outputs live under `results/experiments/knot/cpu-physics-pilot-v1`: membership,
benchmark/budget, training histories, best/final checkpoints, common posterior
HDF5 files, compact reconstruction/joint/spin metrics, oracle ladder, spin
definitions and a small distribution plot set. The common posterior and shuffle
semantics are documented in `KNOT_POSTERIOR_PILOT_CONTRACT.md`; the historical
source comparison is in `HISTORICAL_JER_VS_KNOT_M.md`.

Preflight found a correctness defect: unsupported visible assignments had been
supervised as NULL rather than masked. The authoritative sidecar now takes
precedence over cached labels, and assignment CE masks unsupported events.
Three already-recorded truncation failures give exactly zero assignment loss
with finite continuous losses. No new population scan or max-jets change was
performed. Parent tensor order is top, antitop; scaler/query ordered equality
and explicit name-based candidate mappings protect it, while the compiler's
alphabetically sorted root list is used only for name coverage.

This pilot diagnoses mechanisms before convergence. Small differences do not
establish statistical superiority, and a J advantage would first concern
optimization, finite capacity or sequential error propagation, not greater
representational expressivity. The decision record contains the interpretation
of the actual metrics and the single production recommendation.

## Observed result

Outcome K1: C/J exhibit coherent N-P dependence while marginal metrics remain
broadly comparable. Mean conditional correlations (coherent -> shuffled) are
M -0.00002 -> 0.01643, C 0.16570 -> 0.00488, J 0.54512 -> -0.00827.
The truth-dependent bivariate energy-score shuffle penalties are respectively
0.068 +/- 0.124, 0.468 +/- 0.141, and 1.877 +/- 0.142 GeV. Marginals remain
unchanged to float precision. A stronger J shuffle response is not proof of
better joint calibration or greater expressivity.

C's parent-pT RMSEs for truth-A/truth-N, predicted-A/truth-N, and deployed
contexts are (72.17,72.85), (72.62,73.28), and (72.60,73.24) GeV. The corresponding
mttbar RMSEs are 139.82,142.29,142.20 GeV. There is no large upstream degradation
at this training length; no scheduled sampling or context-training change is
justified. The dominant error source at convergence is not established.

The measured lab dilepton azimuth is exactly shuffle-invariant. Helicity and
common-frame opening angles respond to top-antitop pairing, including in M.
All models have biased hybrid helicity distributions at 600 steps: this pilot
validates sensitivity of the evaluator, not a calibrated spin reconstruction.

Stored total parameters are M/C 2,775,317 and J 2,954,891. Trainable parameters
are M/C 2,775,317 and J 1,567,240: J retains a frozen unused invisible flow.
Thus the approximately 6.5% total-count difference is not an active capacity
comparison. No capacity matching is performed.

Recommendation: KNOT_FULL_GPU_CAMPAIGN_WITH_EVALUATOR_UPDATE_ONLY. Use the
common evaluator and validated correctness fixes with the frozen production
architectures and configs. A harmonized historical physical-composition control
would be useful later, but is not required before this campaign. No job was
submitted.
