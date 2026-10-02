# Historical maintained JER versus KNOT-M

Sources inspected read-only: `JointEventReconstruction` at
`be3f4911a234895409dc3e44cc4583a111454c93`, its `models/event.py`,
`models/invisible/generator.py`, `prediction.py`, and `configs/topology/ttbar2L.yaml`.
The newer maintained REALNU path is also inspected in this closure baseline's
`models/event.py` (`_parent_assignment_context`, `_predicted_neutrino_sample`,
`sample_partonic`) and `models/partonic.py`. These are distinct historical paths.

| Property | Legacy maintained JER | REALNU P2 parent flow | KNOT-M |
|---|---|---|---|
| Assignment into N | Explicit `none`, probability-weighted `soft`, argmax `hard`, or decoded `hypothesis`; hypothesis training uses truth, inference uses decoded candidates | The selected invisible assignment mode; production draft J_AN uses `none` | None; q(N|x) |
| Assignment selection | Decoder ranks feasible hypotheses; not a generic random draw of A | Parent context marginalizes feasible complete-state probabilities into soft candidate embeddings | A sampled from feasible complete-state probabilities; independent of continuous marginals |
| Neutrino sampling | Multiple N draws per decoded hypothesis, or repeated views of one assignment-independent posterior | Exactly one N draw per event before parent sampling | Fresh N draws across sample index |
| Parent construction | `compose_physics` deterministically adds selected measured visible vectors and the corresponding N draw | Learned P2 flow consumes soft A context and one predicted N draw | Learned q(P|x), consumes neither A nor N |
| Reuse of one N | No independent learned P sampling; composed parent k uses N k | That single N is reused for every parent k (`invisible_sample_index=0`) | No N-to-P conditioning |
| Meaning of k | Within a decoded branch, deterministic composition follows that branch's A and N k; soft/none N conditioning is not hard sampled ancestry | Parent sample index alone does not constitute new A->N->P ancestry | Storage pairing of independent marginals; never coherent ancestry |
| Truth role order | `top_e`, `top_mu`: electron-side, muon-side | Explicit top/antitop mapping required for P2 | Top/positive side, antitop/negative side |
| Parent truth | Exported truth composites built from truth-labelled measured visible vectors plus truth N | Explicit partonic parent sidecar, checked as after-FSR when configured | Authoritative partonic after-FSR top/antitop |
| Shared encoder | One measured representation reused by assignment and invisible generation; historical example config uses feature Transformer | One encoder reused with optional topology queries and explicit gradient policy | Frozen measured L-GATr plus topology queries; assignment and continuous objectives train shared parameters |

These distinctions are source statements, not an old-metric versus new-metric
comparison. In particular, legacy truth composites are not interchangeable with
partonic after-FSR parent targets, and flavour ordering cannot be treated as
charge ordering in same-flavour dilepton channels.

A harmonized old-JER numerical control would need the same measured population,
charge-side topology, neutrino truth and after-FSR parent truth for evaluation.
A learned legacy parent baseline also needs retraining against those targets.
Its possible scientific value is isolating deterministic physical composition
from separately learned parent marginals. The CPU pilot decision record states
whether that additional control is warranted by the observed joint diagnostics.
No historical numerical metric is reused and no historical model is retrained.
