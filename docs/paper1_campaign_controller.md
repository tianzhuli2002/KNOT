# Paper-1 campaign controller

The Paper-1 campaign is an event-driven Slurm state machine. It has no daemon:
each smoke/study stage has a short controller job with an `afterany`
dependency on every relevant parent job. The controller reads the immutable
campaign source snapshot, reconciles Slurm and validated output products,
performs one policy-backed transition, records state atomically, and exits.

Campaign state is kept under:

```text
results/campaigns/paper1/
```

The directory contains the source and policy snapshots, a campaign-local gate
overlay, decision evidence, controller submissions and the authoritative
`campaign.json`. Tracked locked gates are never edited by a worker. A selected
variant that reuses an older checkpoint records both the selection provenance
(`source_study`/variant that won the current decision) and the true training
provenance from the checkpoint identity.

## Operations

```bash
jer campaign launch paper1 --auto
jer campaign status paper1
jer campaign watch paper1
jer campaign resume paper1
jer campaign review paper1
jer campaign approve paper1 invisible_coordinates cartesian
```

`launch` is idempotent for a source SHA and adopts an existing matching smoke.
`resume` repairs only missing postprocessing products, once per stage; failed
training, OOM, NaN, source, data or checkpoint checks stop the campaign.
`approve` is the only way to resolve a `REVIEW_REQUIRED` gate.

The controller uses the existing strict study DAG for train → predict →
evaluate → aggregate dependencies. Cross-study promotion is separate and
uses `afterany`, so a failed or dependency-cancelled DAG still wakes a
controller that can classify it. Scheduler states such as `Priority`,
`Resources`, `QOSGrpGRES` and `QOSMaxGRESPerUser` remain pending states.

The policy in `configs/campaigns/paper1.yaml` is copied into each campaign and
hashed. It uses declared equivalence margins, seed consistency, guardrails and
paired evidence when event-level paired arrays are available. It never picks
the smallest central value without those checks. The campaign ends before
production review if the validated production manifest is not enabled, and it
has no automatic real-test command.
