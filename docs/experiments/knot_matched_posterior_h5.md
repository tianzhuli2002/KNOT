# Matched KNOT posterior HDF5

The production-checkpoint evaluator writes one `posterior.h5` per variant. Row
`i` in every event-level dataset belongs to `EVENT_ID[i]`; event IDs are unique
and in the deterministic validation-selection order shared by M, C and J.
`sample_index[i, k]` is local to one event. Four-vectors use `(E, px, py, pz)` in
GeV.

The root `posterior_semantics` attribute distinguishes the posterior contracts:

* `marginal` (M): assignment, invisible and parent components are marginal
  draws. Equal sample indices do not define a coherent physical hypothesis.
* `ancestral` (C): each index follows the sampled assignment through
  `N | A, x` and `P | A, N, x`.
* `joint` (J): each index contains the jointly transported `(N, P)` draw for
  its sampled assignment.

Root datasets `assignment_state`, `assignment_candidates`,
`assignment_log_probability`, `invisible`, `parents`, and their support masks
are the sampled event posterior. The `assignment_nodes`, `invisible_roles`, and
`parent_roles` attributes give the final role-axis order. The `factorisation`
attribute records M/C/J directly.

`TRUTH/` contains the validation truth and `TRUTH/assignment_support` marks
events whose truth assignment exists in the model-native feasible set.
`diagnostics/assignment_MAP_*` stores the MAP assignment and role-level
correctness; `diagnostics/assignment_truth_NLL` is defined only on supported
truth assignments. The measured charge-ordered leptons are stored in
`measured/leptons[:, 0]` for positive lepton/top and `[:, 1]` for negative
lepton/antitop. Generator-level lepton four-vectors are unavailable, so spin
truth comparisons use the documented hybrid of after-FSR truth parents and
measured leptons.

For C, each `oracle/...` dataset is restricted to supported rows, preserving
the order of `EVENT_ID[TRUTH/assignment_support]`. Parent oracle arrays separate
truth-assignment/truth-neutrino and predicted-assignment/truth-neutrino
contexts. `oracle/neutrino_truth_assignment/invisible` and
`oracle/neutrino_predicted_assignment/invisible` compare the two assignment
contexts for the invisible draw.

`spin/truth`, `spin/paired`, and `spin/shuffled` contain the five observables
named in `SPIN_DEFINITION`. `sealed_test_used=false` is stored in the file and
the evaluation manifest; the input is the frozen train-plus-validation file.
