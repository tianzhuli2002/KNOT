# Common KNOT posterior contract

The maintained `EventPosterior` is reused. All arrays have an event axis keyed
by `EVENT_ID`; draws have a second explicit event-local `sample_index` axis.
Four-vectors use E,px,py,pz in GeV. Role metadata determines all role axes.
Assignment state, candidate indices, sampled-state log probability, invisible
and parent draws and role support masks are common across M/C/J.

`posterior_semantics` is `marginal` (M), `ancestral` (C), or `joint` (J).
M pairs independent marginal draws for storage only. It does not claim coherent
ancestry. C and J sample indices define coherent event hypotheses.

The pilot writer adds separate TRUTH arrays, assignment-objective support,
measured charge-ordered leptons, spin-definition metadata and C oracle parent
draws. Normal prediction takes a truth-sanitized batch; truth is joined only
in evaluation. Parent oracle calls are explicit C-only diagnostic endpoints.
No truth enters the measured encoder or ordinary posterior prediction.

Neutrino–parent shuffling permutes the whole parent draw relative to the
neutrino draw. Spin observables depend on parents and measured leptons, so their
coherence test instead independently permutes the antitop draw relative to top.
A whole parent-pair permutation leaves all parent-only marginal distributions
unchanged and cannot establish spin sensitivity to N–P ancestry.
The spin quantities probe dependence within P (top versus antitop), which M
can also learn. Even a spin shuffle response does not by itself establish
coherence of the complete A,N,P posterior.

Spin axes and boosts are documented in `knot_physics.py` and stored verbatim in
prediction metadata. Spin observables from M are diagnostic marginal pairings,
not a coherent learned joint posterior. Lab dilepton azimuth is a control.

The primary 68% neutrino-pz coverage uses ranks among the 16 posterior draws.
Each rank represents one of 17 PIT bins; its overlap with [0.16,0.84] is
integrated deterministically. This has 68% expectation under exchangeability.
The raw empirical-quantile coverage is retained separately and has finite-draw
bias; it must not be interpreted as an exactly nominal 68% diagnostic.
