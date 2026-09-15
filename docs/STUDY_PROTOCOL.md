# Paper-1 study protocol

This is the current validation-only protocol for the first complete
reconstruction paper. It is deliberately a decision path, not a requirement to
finish every historical study.

```text
02 + 08  generator objective and CFM inference budget
   ↓
13       Cartesian versus MET-residual invisible coordinates
   ↓
15       conventional versus Conditional L-GATr invisible backbone
   ↓
10       invisible-physics assignment reranking
   ↓
03       optional slim reverse pass, only if Study 10 helps
   ↓
freeze and evaluate the sealed test split once
```

All choices before the final freeze use the training split for fitted objects
and the validation split for selection. The real test split is not inspected.

Study 10 has five controlled views: the visible assignment baseline, a
capacity-matched visible-only correction, analytic topology physics, a small
learned residual correction, and the non-deployable truth-neutrino oracle. The
physics block is topology-configured (`sum`, invariant mass, mass difference,
and MET transverse residual) and remains a soft compatibility calculation. A
single unconditioned posterior is reused across top-K assignment hypotheses.

Study 03 is intentionally slim: `q(nu|X)`, `q(nu|X,A_refined)`, and a
truth-assignment oracle. It is released only when real reconstructed invisible
physics produces a useful assignment improvement in Study 10.

Studies 04, 06 and 07 are deferred follow-up work. They are not Paper-1
blockers and must not be released by an unfiltered campaign command.
