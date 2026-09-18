# Single-root runtime

The maintained CSF3 project root is
`/net/scratch/w00238tl/JointEventReconstruction` (normally
`~/scratch/JointEventReconstruction`). `configs/site/csf3.yaml` is the only
site-specific path/resource configuration.

## Runtime roots

`JER_PROJECT_ROOT` is the project root for data, cache, results, logs and
runtime state. `JER_SOURCE_ROOT` is the source tree used by the current
process. Interactive commands use the checkout; submitted workers use
`runtime/source/<JER_SOURCE_SHA>/`, an archive created with `git archive`.

The archive is checked using `SOURCE_SNAPSHOT.json`, the full commit SHA and
the required source/configuration files before a worker starts. The worker
exports the source SHA and source-root path, but all scientific artifacts still
resolve from the project root. This is why a user can pull or edit the normal
checkout after submission without changing queued-job code.

## Campaign recovery

Use:

```bash
jer campaign status
jer study status 13
jer study logs 13
```

Status is output-aware. Validated prediction/evaluation/review products remain
scientifically complete even if an historical aggregate process timed out.
For a postprocessing-only failure, use the existing study submission recovery
path after checking the logs; do not rerun valid predictions.

`jer cleanup --dry-run` reports only redundant JER-specific paths. It does not
touch unrelated repositories or remove scientific products.
