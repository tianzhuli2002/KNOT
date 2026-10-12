# KNOT/RISE filesystem migration and posterior-usage audit

This is the final human-readable record for the canonical filesystem cleanup and KNOT posterior evaluation. The full file inventory, hashes, manifests, worktree list, and per-observable diagnostics are in `filesystem_migration_20261012.json`.

## Canonical state

- KNOT: `/net/scratch/w00238tl/knot`, branch `research/knot-production-physics-evaluation-v1`.
- RISE: `/net/scratch/w00238tl/rise`, branch `research/rise-r0-neural-diagnosis-v1`, HEAD `8fa8e0fa41f6801a14e9c77be21416adc8de817e`.
- The KNOT source/runtime commit captured by the manifest is `0f73ab8838295846e22fed1d6ba40b2885e52c1b`. The final documentation-only commit follows that capture; the manifest cannot contain its own commit hash.
- Both source trees were clean at capture. The final worktree paths and captured HEADs are listed in the JSON manifest.
- The obsolete top-level `JointEventReconstruction_KNOT`, `_evaluator`, `_evaldev`, and `JointEventReconstruction_RISE` project roots are absent. Other shared-repository worktrees and retained backup worktrees remain; they are not duplicate canonical KNOT/RISE roots.

## Preservation and provenance

The KNOT merge still accounts for 629 result paths: 385 were already identical in the evaluator tree and 244 unique paths were migrated. There were no same-path content conflicts. All 50 log paths were checked and 40 unique logs copied. RISE retains its 165 inventoried result files, including the R0 qualification decision and diagnostics.

The six earlier checksum differences are the expected mutable path/provenance manifests. Their original and current hashes are recorded individually in the JSON manifest; every current hash matches the resealed canonical hash. The J/M evaluation manifests were normalized to ordinary JSON when terminal padding was removed. No immutable scientific payload mismatch was found.

The bounded topology-query audit compared 11 relevant files under `partonic_ttbar2L_2M_v1` and 29 under `partonic_ttbar2L_2M_realnu_v1`, with relative path, size, SHA256, role, and classification recorded. The unique 2M development inputs were recovered into KNOT: source and development manifests, the development event list, and the max-jets-8/20 measured and supervision HDF5 files and join views. The split-membership file was already identical. Both join views now link to canonical KNOT HDF5 paths. Their 384 events split into 256 training and 128 validation events, with no sealed-test events; neutrino supervision remains disabled because its side association is unverified. All scoped real-neutrino source, split, support, and adapted-input candidates are present in KNOT. No additional useful file was missing. The backup remains untouched.

## Runtime dependency

The 195,505,429-byte external package bundle is now at `knot/.runtime/pythonpath/` and is ignored by Git. The four canonical KNOT Slurm wrappers use this path. A scan of the current and retained worktree scripts found no other consumers of the old top-level path. The pinned production interpreter, Python 3.13.15, successfully imported h5py, einops, lgatr, SciPy 1.18.1, scikit-learn, and PyTorch 2.14.0+cu130 from the relocated bundle. All four wrappers pass `bash -n`.

## K64 posterior and plotting products

The primary frozen KNOT-J posterior is:

`results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/J-K64/posterior.h5`

It contains 10,000 events and 64 coherent samples per event, from the existing frozen checkpoint `best-16-1.213281.ckpt`. CPU inference completed as Slurm job `22827672`. The evaluation manifest parses as JSON, validates the checkpoint and event population, and records `sealed_test_used: false`. No training or K128 inference was run.

The publication-oriented output is `results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/posterior-usage-plotting-v1/`. The separate K32 validation output remains at `posterior-usage-plotting-K32-validation-v1/`. M/C/J factorisation comparisons use K=32 for all three models.

The statistical definitions are:

- **One draw:** one deterministic SplitMix64 event-keyed sample, seed `42011`, reused coherently across parents, neutrinos, spin-capable quantities, and derived observables. The selection hash is `1c9d6b404fb87a282455c2b6bee63dfa79c2ef3c7dbae9f01ebec69e9720dd76`; replay and row-order invariance pass.
- **Posterior mean:** for nonlinear observable `O`, evaluate `O` on each coherent posterior sample and average the values, `E[O|x] = (1/K) Σk O(yᵏ)`. This is not `O(E[y|x])`.
- **Full posterior:** use all K samples with event-normalized weight `1/K`. Its total weight is one per event, and it is never entered into ordinary residual histograms.

For mttbar, the mean absolute eventwise difference between `E[mttbar|x]` and mttbar computed from posterior-mean parent four-vectors is 19.22 GeV (mean signed difference -19.22 GeV; RMS 34.24 GeV). The headline estimator uses `E[O]`.

The K64 output contains 115 files and 46 PNG/PDF figure pairs. Spectra show truth, one draw, posterior mean, and full posterior. Residuals show only one draw and posterior mean. Migration matrices use common bins and event-normalized weights. The existing PCFM-derived plotting style was retained; the core spectra and residuals, response, migration comparison, factorisation, coverage, neutrino-pz, convergence, and one-draw ensemble plots were visually inspected.

### mttbar point reconstruction

| Estimator | Bias [GeV] | RMSE [GeV] | MAE [GeV] |
| --- | ---: | ---: | ---: |
| One draw | 4.959 | 107.967 | 74.219 |
| Posterior mean | 4.670 | 78.615 | 53.641 |

### mttbar population reconstruction

| Representation | W1 [GeV] | KS | IQR ratio | 68% width ratio | q95 tail ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| One draw | 6.669 | 0.0260 | 1.0327 | 1.0198 | 1.0559 |
| Posterior mean | 12.386 | 0.0821 | 0.9311 | 0.9084 | 0.9142 |
| Full posterior | 5.639 | 0.0193 | 1.0445 | 1.0221 | 1.0619 |

The full posterior has `n/a` point RMSE in `tables/posterior_usage_scorecard.csv`. Point-estimator metrics and population-spectrum metrics are kept separate.

### Conditional posterior quality

K64 mttbar coverage is 0.479/0.662/0.890 at nominal 50/68/90%. The 65-bin rank/PIT summary has maximum absolute rank-fraction deviation 0.00532, RMS deviation 0.00186, and χ² p-value `1.87e-8` for 64 degrees of freedom; CRPS is 37.417 GeV. Mttbar coverage is slightly below its finite-draw expectation. Under-dispersion is stronger for spin helicity cosine posteriors, whose 68% coverages are 0.569 and 0.560. No posterior-width calibration was applied. Population-spectrum closure does not imply conditional calibration.

## K32 to K64 convergence

K32 and K64 use the same checkpoint and aligned event population, but the stored draws are non-nested: K32 was generated on CUDA and K64 on CPU. The comparison therefore measures differences between two empirical posterior ensembles, not paired sample trajectories. The full per-observable W1, KS, truth-metric, q90/q95/q99-tail, coverage, and migration changes are in `tables/k32_k64_convergence.csv` and `metadata/k32_k64_convergence.json`.

| Observable | W1(K32,K64) | KS(K32,K64) | Δ truth W1 | Δ truth KS | Δ IQR | Δ 68% width | Δ q90 | Δ q95 | Δ q99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mttbar | 0.3298 GeV | 0.002158 | +0.1909 GeV | -0.000361 | -0.000993 | +0.000870 | +0.000718 | +0.003244 | -0.005724 |
| ptttbar | 0.1729 GeV | 0.002005 | +0.0275 GeV | +0.000919 | -0.00232 | -0.005933 | +0.001452 | +0.002464 | -0.002166 |
| yttbar | 0.001036 | 0.001091 | +0.000534 | +0.000583 | -0.000243 | +0.000201 | -0.000062 | -0.010479 | +0.000155 |
| top pT | 0.1316 GeV | 0.000985 | +0.09884 GeV | +0.000289 | +0.000009 | +0.001618 | +0.002179 | +0.006337 | +0.003654 |
| neutrino pT | 0.06748 GeV | 0.000909 | +0.04563 GeV | +0.000238 | +0.002023 | +0.001729 | +0.000640 | +0.008492 | +0.009406 |
| neutrino pz | 0.2512 GeV | 0.001112 | -0.1882 GeV | -0.000917 | -0.003091 | -0.001023 | +0.004584 | +0.003684 | +0.008396 |

The mttbar migration matrix changes by at most 0.317 percentage points per cell, with mean truth-bin total variation 0.00338. For top pT these are 0.689 percentage points and 0.00294. Width, q95-tail, and 68%-coverage conclusions retain the same qualitative side of truth or finite-draw expectation across K32 and K64 for all six observables. K64 is the publication default for primary J; the reported differences are finite-sample shifts, not an exact-equality requirement.

## One-draw variability

The 128 existing one-draw pseudo-datasets use seeds `42012`–`42139`; the displayed draw uses seed `42011`. Percentiles use the empirical mid-rank among those pseudo-datasets.

| Metric | Mean | SD | 16–84% | 2.5–97.5% | Displayed draw | Percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 [GeV] | 5.775 | 0.603 | [5.228, 6.353] | [4.692, 7.197] | 6.669 | 95.3% |
| KS | 0.02106 | 0.00225 | [0.01903, 0.02347] | [0.01692, 0.02510] | 0.0260 | 99.2% |
| IQR ratio | 1.044 | 0.0103 | [1.033, 1.054] | [1.022, 1.063] | 1.033 | 11.7% |
| q95 tail ratio | 1.060 | 0.0267 | [1.029, 1.084] | [1.003, 1.105] | 1.056 | 40.2% |

The selected draw is unusually high in W1 and KS within this ensemble, so it is not representative on those two spectrum metrics. The complete seven-metric ensemble table includes mean, median, standard deviation, 16/84%, and 2.5/97.5% for W1, KS, IQR, central-68 width, and q90/q95/q99 tails.

## Principal output paths

- Scorecard: `results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/posterior-usage-plotting-v1/tables/posterior_usage_scorecard.csv`
- Concise summary: `results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/posterior-usage-plotting-v1/POSTERIOR_USAGE_SUMMARY.md`
- mttbar spectrum and residual: `.../plots/core/mttbar_shape.png`, `.../plots/core/mttbar_residual.png`
- K32/K64 convergence: `.../plots/convergence/mttbar_k32_k64_posterior.png`, `.../plots/convergence/top_pt_k32_k64_posterior.png`
- One-draw metric ensemble: `.../plots/core/mttbar_one_draw_metric_ensemble.png`
- Migration comparison: `.../plots/migration/mttbar_j_migration_comparison.png`

## Validation and remaining scientific limits

The final targeted programmatic checks passed: 47 assertions covered event alignment and uniqueness, supported parent/neutrino finiteness, K32/K64 dimensions and truth alignment, deterministic draw replay, event-normalized weights, residual representation policy, histogram integrals, migration normalization, finite convergence tables, output file integrity, and paired plots. `py_compile`, Ruff, `git diff --check`, and `bash -n` passed. The targeted pytest module could not run because `pytest` is not installed in either available Python environment.

RISE received filesystem and provenance validation only. Its R0 decision and qualification plots match their recorded hashes; no detector-level RISE work was started. The genuine remaining scientific limitation is conditional posterior under-dispersion, especially for the spin helicity cosines; no post-hoc calibration was applied.
