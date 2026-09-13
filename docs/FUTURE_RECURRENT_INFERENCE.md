# Future recurrent inference

Study 01 locks only the present conclusion: visible-assignment-only recurrence did not improve the `ttbar2L` structured-assignment problem at the 100k-event training scale. It does not establish that recurrence is generally useless. The ideas below remain evidence-gated backlog items and are not part of the current campaign.

## Physics-conditioned multi-hypothesis co-refinement

This is the preferred future candidate. Keep the top `K` visible assignments, form an invisible posterior for each hypothesis, reconstruct the top-pair system and MET-closure compatibility, and use those physical quantities and posterior widths to rerank or correct the hypotheses before an optional next inference step:

```text
top-K visible assignments
        -> neutrino posterior per hypothesis
        -> reconstructed physics and compatibility
        -> rerank / residual correction
        -> optional next inference step
```

Do not implement it until Studies 03, 04, 06 and 10 establish that several hypotheses, posterior-informed correction, joint modelling, or compatibility provide evidence that hidden recurrence is needed.

## Joint assignment--neutrino recurrent state

Consider a state of the form `(A_t, nu_t, reconstructed physics_t, uncertainty_t)` only if Study 03 shows assignment-to-neutrino benefit and/or Study 04 shows neutrino-to-assignment benefit, with Study 06 also justifying joint modelling.

## Transformer and L-GATr multi-view refinement

Do not build a dual-encoder recurrent model merely because both encoders are available. Study 12 must first show substantial event-level complementarity, especially through the oracle union `P(Transformer correct OR L-GATr correct)`. Similar mistakes are evidence against this direction.

Study 12 validation evidence, aggregated across the three seeds, records approximately:

```text
Transformer vs L-GATr disagreement: 7.66%
Transformer-only correct:            3.61%
L-GATr-only correct:                 4.05%
Oracle union:                       87.07%
```

The representations make materially complementary mistakes. This supports considering multi-view reasoning later, but does not justify implementing it before Studies 02–06/10.

## Streamlined future sequence

The maintained programme now keeps the smallest experiments that can change the
physics model. Study 02 selects the invisible generative objective and Study 05
selects the assignment formulation. Study 11 is only a non-blocking 500k-event
M1-versus-M3 stress test; its result cannot reopen the locked Study-01 M1 gate.

The next coordinate question is **How Should the Invisible State Respect MET?**:
Cartesian and MET-residual coordinates are the trained comparison, with
MET-exact retained as a diagnostic projection. Once the objective and coordinate
gates are human-selected, Study 15 asks **Does Lorentz-Equivariant Generation
Improve Invisible Reconstruction?** by comparing the selected conventional
generator against a Conditional L-GATr backbone at matched training and
inference budgets. The baseline should be reused when its checkpoint is
scientifically identical rather than retrained.

The later assignment-to-invisible stage should preserve the explicit posterior
mixture

```text
p(A, nu | X) = sum_k p(A_k | X) p(nu | A_k, X)
```

and compare none, soft, top-K and truth-oracle assignment conditioning before
any top-1 collapse. The merged physics-correction question is **Can Invisible
Physics Resolve the Jet Assignment?**: baseline, one interpretable
posterior-informed correction, and top-K physics compatibility/reranking. A
future topology-conditioned geometric model may use assignment-role,
invisible-particle, W, top and event/ttbar composite tokens; this is a design
note only, not an implementation or a reason to start joint co-refinement.

## Recurrence scaling revisit

Study 11 retains only a deliberately limited M3 diagnostic at 500k events. Compare its mean, seed spread and M1--M3 delta with the existing 100k result; do not reopen the Study 01 gate from this scan.
