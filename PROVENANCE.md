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
