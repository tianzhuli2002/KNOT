# September 2026 framework finalisation audit

## Audit identity

- base SHA: `ce820dc1e39ade128ee86399e0a35761f54f379a`
- audit branch: `framework-final-audit`
- audit date: 2026-09-02 to 2026-09-03
- tracked files at audit start: 148
- final tracked files in the current tree: 147
- final CPU validation pass: 2026-09-13
- excluded artifacts: untracked/generated caches, HDF5 products, plots and Slurm logs

This report is the permanent file-by-file record for the last framework audit
before the encoder-representation campaign. Historical Study-01 products are
immutable inputs and are not counted as tracked source files.

## Findings

### P0

- A model-dependent, hash-named historical Transformer cache path remained in
  the public data interface and normal structured training could still fall
  through to per-event HDF5 preprocessing. Fixed with one model-agnostic,
  validated tensor cache and cache-backed DataModule path.
- Public topology roles and numeric HDF5 truth labels shared ownership. Fixed by
  keeping semantic truth in topology and moving numeric bindings to explicit
  maintained dataset adapters.

### P1

- Study-01 postprocessing could not import its sibling recurrent-depth module
  when executed by file path. The repaired entry point writes a distinct
  derived-analysis attempt with source and analysis provenance.
- Public configuration exposed storage paths, kinematic field maps, parallel
  variable/scaler lists and DataLoader engineering knobs. Fixed with one
  `data.input/data.cache` pair, one ordered `inputs` declaration and runtime
  loader defaults.
- An incompatible cache could be rebuilt implicitly. Fixed: mismatch is fatal;
  only an explicit force request replaces it.
- Canonical `EVENT_ID`, `SPLIT` and invisible truth were not all retained at the
  deterministic model-input boundary. Fixed and validated in cache format 9.
- Cached batches exposed storage row indices where invisible truth joins require
  canonical EVENT_ID (especially wrong for the odd/test partition). Fixed so
  cached and raw batches both expose canonical EVENT_ID.

### P2

- Study 12 exposed a permanent `relational_transformer` type and repeated input
  lists per encoder. Fixed with orthogonal Transformer `representation` and
  `relations` controls and one top-level input declaration.
- Public recurrence and invisible-generator blocks remained deeper than their
  scientific choices require. Simplified without changing resolved internal
  mechanics.
- User documentation still described a versioned replacement framework. The
  version label and duplicate public vocabulary were removed.

## Validation evidence

- all 13 maintained study variant sets resolve under the final public schema;
- compileall, selected Ruff correctness rules and `git diff --check` pass;
- a 256-event real ttbar2L fixture proved build, mmap reuse, mismatch failure
  and forced rebuild; the final 3,921,094-event cache is one model-agnostic
  mmap-loadable tensor file;
- raw/scalar versus vectorised cache equivalence passed on fixed train,
  validation and test rows, including canonical EVENT_ID joins and invisible
  truth;
- the corrected vectorised build completed in 114.49 s on a warm source cache
  (the first final vectorised build was 415.83 s), versus the earlier 890.93 s
  per-event baseline; peak RSS was 10,592,512 KiB;
- unrestricted scientific invariants, five shared-cache encoder smokes,
  baseline raw/cache regression, representative real-cache reconstruction
  paths, Study-01 controls, all study plans and all Slurm dry plans pass;
- the final L40S gate remains pending until a clean committed SHA is submitted.

## Final CPU evidence (2026-09-13)

- Cache equivalence artifact: `results/engineering/framework-final-audit/cache_equivalence.json`;
  maximum continuous difference `4.76837158203125e-07`; type IDs are stored as
  `int8` and promoted to `long` at the embedding boundary; EVENT_ID, split,
  visible truth and invisible truth are exact.
- Cache artifact: `results/engineering/framework-final-audit/cache-build.json`;
  input `/net/scratch/w00238tl/JointEventReconstruction/data/real/ttbar2L_canonical.h5`,
  cache `/net/scratch/w00238tl/jer_cache/ttbar2L_canonical.pt`, version 9,
  3,921,094 events, 4,034,813,989 bytes, cache state HIT.
- `scripts/check_scientific_invariants.py` exits 0 with all maintained contract
  markers, without an arbitrary timeout.
- All five Study-12 encoders use the same cache and selected EVENT_IDs;
  Transformer, four-vector Transformer and relational Transformer complete
  CPU backward; L-GATr and L-GATr-Slim complete finite CPU forward.
- The historical Transformer baseline checkpoint loads strictly and produces
  identical raw-HDF5/cache inputs and outputs on a fixed 32-event validation
  batch.
- Real-cache mechanism smoke passes CFM Cartesian, CFM MET-residual,
  conditional OT, JointFlow MET-residual, truth-oracle conditioning,
  checkpoint reload, prediction-schema validation and evaluation. JointFlow
  impossible assignment states intentionally use `-inf` logits; NaN, `+inf`,
  loss and gradient checks pass.
- Study 01 parameter controls remain unchanged and match trainable parameters
  within 0.1%: M2 `-0.0441%`, M3 `-0.0579%`. Fixed model seeds are
  `[42, 43, 44]` with data subset seed 42 and validation selection.
- Analysis 14 is mechanically validated as validation-only APS calibration;
  all Studies 01--13 and all 13 Slurm DAG dry plans pass. No training or
  calibration campaign has been run.

The requested VyPER reference inspection found its established fast path in
`VyPER/data/preprocessed.py` and `VyPER/data/datamodule.py`: one-time graph
preprocessing writes a single on-disk database, then DataLoader workers read
serialized records without repeating HDF5 feature construction. JER borrows
the same one-time-preprocess/thin-loader principle while retaining its simpler
plain PyTorch tensor cache and physics-specific fixed-token contract.

## Deferred issues

- Truly exchangeable invisible particles remain unsupported; flavour is never
  substituted for semantic parent role and pT sorting is not used as a proxy.
- The external ttH storage layout remains behind its small maintained adapter.
- Study-01 scientific selection remains a human decision after control,
  transition, depth, ambiguity and runtime review.

## Scientific non-changes

- No assignment, diffusion or flow formulation is added.
- No scientific winner is selected automatically.
- No Study 12 or other new scientific campaign is submitted.
- Historical Study-01 manifests, checkpoints and predictions are not modified.

## Tracked-file inventory

The inventory below covers every tracked source at audit completion (generated
results are excluded). “Fixed” means the file was inspected and changed for a
finding or its direct migration; “docs/static” means declarative or documentary
content was inspected; “no issue” means inspection found no justified change.

- `.github/workflows/synthetic-smoke.yml` — AUDITED — no issue
- `.gitignore` — AUDITED — docs/static
- `PROVENANCE.md` — AUDITED — fixed
- `README.md` — AUDITED — fixed
- `STUDY_PROTOCOL.md` — AUDITED — fixed
- `configs/.gitkeep` — AUDITED — no issue
- `configs/calibration/assignment_sets.yaml` — AUDITED — no issue
- `configs/defaults.yaml` — AUDITED — fixed
- `configs/evaluation/default.yaml` — AUDITED — no issue
- `configs/experiments/ttH1L.yaml` — AUDITED — fixed
- `configs/experiments/ttbar1L.yaml` — AUDITED — fixed
- `configs/experiments/ttbar2L.yaml` — AUDITED — fixed
- `configs/studies/01_recurrence_controls.json` — AUDITED — docs/static
- `configs/studies/01_visible_recurrence.yaml` — AUDITED — fixed
- `configs/studies/02_generator_formulations.yaml` — AUDITED — fixed
- `configs/studies/03_assignment_conditioning.yaml` — AUDITED — fixed
- `configs/studies/04_c2_feedback.yaml` — AUDITED — fixed
- `configs/studies/05_discrete_assignment.yaml` — AUDITED — fixed
- `configs/studies/06_joint_vs_factorized.yaml` — AUDITED — fixed
- `configs/studies/07_joint_recurrence.yaml` — AUDITED — fixed
- `configs/studies/08_generator_inference.yaml` — AUDITED — fixed
- `configs/studies/09_posterior_size.yaml` — AUDITED — fixed
- `configs/studies/10_compatibility_reranking.yaml` — AUDITED — fixed
- `configs/studies/11_data_scaling.yaml` — AUDITED — fixed
- `configs/studies/12_encoder_backbone.yaml` — AUDITED — fixed
- `configs/studies/13_invisible_coordinates.yaml` — AUDITED — fixed
- `configs/studies/UPSTREAM_SELECTIONS.yaml` — AUDITED — fixed
- `configs/topology/synthetic_contract.yaml` — AUDITED — fixed
- `configs/topology/synthetic_multinu.yaml` — AUDITED — fixed
- `configs/topology/ttH1L.yaml` — AUDITED — fixed
- `configs/topology/ttbar1L.yaml` — AUDITED — fixed
- `configs/topology/ttbar2L.yaml` — AUDITED — fixed
- `data/.gitkeep` — AUDITED — no issue
- `docs/FRAMEWORK.md` — AUDITED — fixed
- `docs/audits/2026-09-framework-finalisation.md` — AUDITED — fixed
- `docs/topology.md` — AUDITED — docs/static
- `docs/ttbar2L_data_contract.md` — AUDITED — docs/static
- `pyproject.toml` — AUDITED — fixed
- `requirements.yaml` — AUDITED — no issue
- `scripts/__init__.py` — AUDITED — no issue
- `scripts/analyze_assignment_sets.py` — AUDITED — no issue
- `scripts/analyze_compatibility_scan.py` — AUDITED — no issue
- `scripts/analyze_posterior_convergence.py` — AUDITED — no issue
- `scripts/analyze_recurrent_depth.py` — AUDITED — no issue
- `scripts/analyze_study_01.py` — AUDITED — no issue
- `scripts/audit_data_contract.py` — AUDITED — fixed
- `scripts/benchmark_encoders.py` — AUDITED — fixed
- `scripts/benchmark_synthetic.py` — AUDITED — no issue
- `scripts/calibrate_assignment_sets.py` — AUDITED — no issue
- `scripts/check_scientific_invariants.py` — AUDITED — fixed
- `scripts/data/convert_tth_visible.py` — AUDITED — no issue
- `scripts/data/convert_vyper_joint.py` — AUDITED — no issue
- `scripts/data/make_synthetic_invisible.py` — AUDITED — fixed
- `scripts/data/validate_joint_h5.py` — AUDITED — no issue
- `scripts/inspect_topology.py` — AUDITED — no issue
- `scripts/match_recurrence_controls.py` — AUDITED — no issue
- `scripts/plot_study_summary.py` — AUDITED — no issue
- `scripts/plot_training.py` — AUDITED — no issue
- `scripts/run_study.py` — AUDITED — fixed
- `scripts/smoke_all.py` — AUDITED — fixed
- `scripts/smoke_flows.py` — AUDITED — fixed
- `scripts/smoke_rng.py` — AUDITED — no issue
- `scripts/smoke_synthetic.py` — AUDITED — fixed
- `scripts/submit_study_slurm.py` — AUDITED — no issue
- `scripts/validate_m0_equivalence.py` — AUDITED — no issue
- `src/joint_event_reconstruction/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/assignment_semantics.py` — AUDITED — no issue
- `src/joint_event_reconstruction/calibration/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/calibration/assignment_sets.py` — AUDITED — no issue
- `src/joint_event_reconstruction/cli/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/cli/m0.py` — AUDITED — no issue
- `src/joint_event_reconstruction/cli/run.py` — AUDITED — fixed
- `src/joint_event_reconstruction/configuration.py` — AUDITED — fixed
- `src/joint_event_reconstruction/data/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/data/invisible.py` — AUDITED — no issue
- `src/joint_event_reconstruction/data/structured.py` — AUDITED — fixed
- `src/joint_event_reconstruction/data/subsets.py` — AUDITED — no issue
- `src/joint_event_reconstruction/evaluation.py` — AUDITED — no issue
- `src/joint_event_reconstruction/evaluation_discrete.py` — AUDITED — no issue
- `src/joint_event_reconstruction/evaluation_joint.py` — AUDITED — no issue
- `src/joint_event_reconstruction/flows/continuous.py` — AUDITED — no issue
- `src/joint_event_reconstruction/flows/discrete.py` — AUDITED — no issue
- `src/joint_event_reconstruction/flows/solvers.py` — AUDITED — no issue
- `src/joint_event_reconstruction/inputs.py` — AUDITED — fixed
- `src/joint_event_reconstruction/m0/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/data/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/data/helpers.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/data/labels.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/data/module.py` — AUDITED — fixed
- `src/joint_event_reconstruction/m0/data/scaling.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/data/splits.py` — AUDITED — fixed
- `src/joint_event_reconstruction/m0/evaluation/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/decoder.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/kinematics.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/performance.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/plot_style.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/provenance.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/evaluation/topology.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/model.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/prediction.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/runtime.py` — AUDITED — no issue
- `src/joint_event_reconstruction/m0/training.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/compatibility.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/event.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/feedback.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/coordinates.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/couplings.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/denoiser.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/edm.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/feedback.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/flow_matching.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/generator.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/invisible/standard.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/joint_flow.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/joint_reasoner.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/reconstruction/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/reconstruction/discrete_flow.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/reconstruction/structured.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/recurrent.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/visible/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/models/visible/encoder.py` — AUDITED — fixed
- `src/joint_event_reconstruction/models/visible/lgatr.py` — AUDITED — fixed
- `src/joint_event_reconstruction/models/visible/physical_features.py` — AUDITED — fixed
- `src/joint_event_reconstruction/models/visible/physics_transformer.py` — AUDITED — fixed
- `src/joint_event_reconstruction/models/visible/transformer.py` — AUDITED — no issue
- `src/joint_event_reconstruction/physics.py` — AUDITED — no issue
- `src/joint_event_reconstruction/physics_observables.py` — AUDITED — no issue
- `src/joint_event_reconstruction/plotting.py` — AUDITED — no issue
- `src/joint_event_reconstruction/prediction.py` — AUDITED — no issue
- `src/joint_event_reconstruction/prediction_discrete.py` — AUDITED — no issue
- `src/joint_event_reconstruction/prediction_joint.py` — AUDITED — no issue
- `src/joint_event_reconstruction/prediction_schema.py` — AUDITED — no issue
- `src/joint_event_reconstruction/public_config.py` — AUDITED — fixed
- `src/joint_event_reconstruction/reliability.py` — AUDITED — no issue
- `src/joint_event_reconstruction/studies.py` — AUDITED — no issue
- `src/joint_event_reconstruction/study_metrics.py` — AUDITED — no issue
- `src/joint_event_reconstruction/topology/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/topology/assignments.py` — AUDITED — no issue
- `src/joint_event_reconstruction/topology/candidates.py` — AUDITED — no issue
- `src/joint_event_reconstruction/topology/compiler.py` — AUDITED — fixed
- `src/joint_event_reconstruction/topology/decoder.py` — AUDITED — no issue
- `src/joint_event_reconstruction/topology/spec.py` — AUDITED — fixed
- `src/joint_event_reconstruction/topology/truth.py` — AUDITED — no issue
- `src/joint_event_reconstruction/training/__init__.py` — AUDITED — no issue
- `src/joint_event_reconstruction/training/joint.py` — AUDITED — fixed
