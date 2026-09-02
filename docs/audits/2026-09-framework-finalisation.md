# September 2026 framework finalisation audit

## Audit identity

- base SHA: `ce820dc1e39ade128ee86399e0a35761f54f379a`
- audit branch: `framework-final-audit`
- audit date: 2026-09-02
- tracked files at audit start: 148
- excluded artifacts: untracked/generated caches, HDF5 products, plots and Slurm logs

This report is the permanent file-by-file record for the last framework audit
before the encoder-representation campaign. Historical Study-01 products are
immutable inputs and are not counted as tracked source files.

## Findings

### P0

To be completed during the audit.

### P1

- Study-01 postprocessing could not import its sibling recurrent-depth module
  when executed by file path. The repaired entry point writes a distinct
  derived-analysis attempt with source and analysis provenance.

### P2

To be completed during the audit.

## Validation evidence

To be completed after the final source SHA is committed.

## Deferred issues

To be completed during the audit.

## Scientific non-changes

- No assignment, diffusion or flow formulation is added.
- No scientific winner is selected automatically.
- No Study 12 or other new scientific campaign is submitted.
- Historical Study-01 manifests, checkpoints and predictions are not modified.

## Tracked-file inventory

Every file listed by `git ls-files` at the final audit SHA will be classified
here before completion.
