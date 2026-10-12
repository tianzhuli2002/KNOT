# KNOT/RISE filesystem migration (2026-10-12)

## Canonical worktrees

- KNOT: `/net/scratch/w00238tl/knot` on `research/knot-production-physics-evaluation-v1`.
- RISE: `/net/scratch/w00238tl/rise` on `research/rise-r0-neural-diagnosis-v1`.
- The four KNOT/RISE worktrees shared `/net/scratch/w00238tl/JointEventReconstruction/.git`; their existing branch histories were already in one repository. Source-only dirty/untracked work was committed on each original branch before the redundant worktrees were retired. Exact old HEADs, preservation commits, current branch tips, and all local branches are in the JSON manifest.

## Result preservation

The three KNOT result trees contained **629 distinct relative paths**. All source copies had matching size and SHA256 when a relative path was shared; there were no differing same-path files. The evaluator tree already held 385 paths. We copied 244 missing paths and 40 missing Slurm logs, then rechecked size/SHA256 for all 629 result paths and all 50 log paths after moving the evaluator worktree to `knot`. Identical duplicate result paths were deduplicated in the canonical tree. The large 2 GB JER feature cache was retained as a reproducibility input.

RISE moved intact to `rise`. Its full result inventory, including `R0/decision.json`, both `R0-qualification` figures, and the updated qualification results, is hashed in the JSON manifest. RISE remains blocked at neural stochastic R0; no detector-level RISE training was started.

## Git/source preservation

- KNOT production old HEAD `eebeb524f877388a10432fc72c7831425f6174e9` remains an ancestor on `research/knot-production-launch-v1`; operational project paths were updated in commit `1904b1c64400311ce2f1ed07e9751c7c988e0ffd`.
- KNOT evaluator old HEAD `37c461d01ce97a28ada2176a9fdc48365cdb8247` remains in `research/knot-production-physics-evaluation-v1`; four untracked analysis/evidence scripts were committed in `1da2b171a8893675df1deed4f447b0457a9c5295`. This is the canonical KNOT worktree.
- KNOT evaldev old HEAD `ae0b70a707bd362b7dcbe1857e93ee1a3b31ad44` remains an ancestor on `research/knot-physics-analysis-dev-v1`; its four tracked changes and 28 untracked source/config/Slurm files were committed in `ff89eba`, and operational paths were updated in `95f34f18cd5dea7f3c18982a9d7fc620703c6979`.
- RISE old HEAD `91fb9bfd5d6bd0f69b37a23484d652a707ab97c7` remains an ancestor on `research/rise-r0-neural-diagnosis-v1`; six tracked R0 qualification updates and `scripts/adapt_canonical_sbi_sidecar.py` were committed in `06b02e0`, with canonical path updates in `8fa8e0fa41f6801a14e9c77be21416adc8de817e`.

## Queue and removal record

`squeue -u "$USER"` was empty both before path changes and before removal. `scontrol show job` reported IDs `22314112`, `22314926`, and `22284319` invalid. No job was cancelled, resubmitted, or otherwise touched; the P5 baseline was not cancelled. Temporary symlinks were not used.

Only reproducible `__pycache__`, `.ruff_cache`, and similar local caches/build temporaries were excluded from scientific migration; their worktree-relative paths, counts, and byte sizes are recorded in the JSON manifest.
