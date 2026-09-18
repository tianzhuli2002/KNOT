# Single-root migration inventory

Inventory captured 2026-09-18 before moving or deleting any external JER path.

The canonical checkout was clean at `5aa6c050b17835766d9f39a9111a6e268f58c7bb`
on `infra/single-root` (created from `study-finalisation`). `squeue -u "$USER"`
reported no active jobs. All live candidates were on the same Lustre device as
the canonical project. Historical JSON and Slurm records are retained without
rewriting their old paths.

| old path | size | action | new location | reason |
| --- | ---: | --- | --- | --- |
| `/net/scratch/w00238tl/jer-source` | 2,688,835 B | DELETE | `runtime/source/<SHA>` | Replace manual source worktree parent with generated source archives. |
| `/net/scratch/w00238tl/jer-source/5aa6c05` | 2,684,739 B | DELETE | `runtime/source/5aa6c050b17835766d9f39a9111a6e268f58c7bb` | No unique required files; code is reproducible from the commit. |
| `/net/scratch/w00238tl/jer_cache` | 4,034,825,472 B | MOVE | `cache/` | Same filesystem; retain the validated tensor cache in the project root. |
| `/net/scratch/w00238tl/jer-cache-audit-ERBSsq` | 766,741 B | MOVE | `results/archive/infrastructure/cache-audit-ERBSsq/` | Preserve small unique audit products. |
| `/net/scratch/w00238tl/jer-plans-CWSQU2` | 440,407 B | MOVE | `results/archive/infrastructure/plans-CWSQU2/` | Preserve historical Slurm plans/submission evidence. |
| `/tmp/jer-hardening` | absent; stale registration | DELETE | — | `git worktree prune`; no active job and no files. |
| `/tmp/jer-paper1-16746fb` | absent; stale registration | DELETE | — | `git worktree prune`; no active job and no files. |
| `/tmp/jer-paper1-5aa6c05` | absent; stale registration | DELETE | — | `git worktree prune`; no active job and no files. |
| `/tmp/jer-submission-provenance` | absent; stale registration | DELETE | — | `git worktree prune`; no active job and no files. |

There were no JER-specific files in the shared top-level `logs/` or
`repo_backups/` directories. The full machine-readable record is
`single_root_migration.json` beside this file.

Migration verification completed 2026-09-18 15:56 UTC: the cache and retained
audit/plan records are under the canonical project, the external source tree
is gone, and `git worktree list --porcelain` contains only the canonical
checkout. No active Slurm job referenced any retired path.
