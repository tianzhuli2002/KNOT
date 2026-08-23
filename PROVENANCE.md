# Provenance

JointEventReconstruction was bootstrapped from the committed tree of
ReconstructionAndSB at `8fde9b0a496cfadf9482a94b33c909d6fbdc1062`.

- Local reference: `/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/ReconstructionAndSB_ref`
- Branch at bootstrap: `local/representation-analysis-harmonised`
- Origin: `git@github.com:tianzhuli2002/ReconstructionAndSB.git`
- Upstream: `git@github.com:diegobaronm/ReconstructionAndSB.git`

The reference had staged representation-analysis additions: CKA, representation
extraction and head-transfer modules/scripts/configs; Optuna dependencies supporting
those studies; local ignore entries; and untracked compatibility tests. Inspection of
the complete staged and untracked diff found no change to the maintained Transformer,
candidate construction, labels, HDF5 loading, decoder, training, prediction or
production configuration. These local analysis artefacts were deliberately excluded.

VyPER was inspected as a scientific donor at
`7fc8ed4028ab0c59e993cfbe32b7edc886216dd1` on `main`.

- Local reference: `/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/VyPER`
- Origin: `git@github.com:tianzhuli2002/VyPER.git`
- Inspected SL-safe branch: `origin/VyPER-SL-safe` at `874ee44`

The SL-safe branch is based on older VyPER code. Its only generally useful operational
change is disabling persistent data-loader workers when `num_workers == 0`. The broad
`nan_to_num` loss/feature substitutions, reduced hyperedge depth and disabled validation
diagnostics conceal invalid physics state or change the model and are not carried over.
The current-main prediction-label fix is already represented by the recorded main SHA.

## Canonical real ttbar data

The canonical real joint records are streamed from the following read-only VyPER Run-3
products under `VyPER_ONNX/VyPER_run3/raw`:

- `ttbar_SL_FM_8M_even.h5` and `ttbar_SL_FM_8M_odd.h5`;
- `ttbar_dilep_30_4_run3_even.h5` and `ttbar_dilep_30_4_run3_odd.h5`.

Their `save_even` and `split_mode` attributes identify disjoint source-row parity
partitions. Historical VyPER logs used an internal random train/validation split within
individual parity files, but did not define one consistent parity as test across all
campaigns. JointEventReconstruction therefore records the explicit maintained policy:
even is deterministically divided 95/5 into train/validation and odd is test.

The dilepton sources contain ee, mu-mu and e-mu events. The canonical ttbar2L product
selects exactly one labelled electron and one labelled muon with opposite source branch
roles. Source JET label matching the electron role becomes canonical JET label 1 and the
muon partner becomes label 2. Invisible storage remains unchanged and ordered
`(nu_e, nu_mu)`.
