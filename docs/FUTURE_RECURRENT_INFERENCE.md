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

## Recurrence scaling revisit

Study 11 retains a deliberately limited M3 diagnostic at 100k, 250k and 500k events. Compare its mean, seed spread and M1--M3 delta; do not reopen the Study 01 gate from this scan.
