# Partonic 2L model selection rule

This rule is fixed before any reserved-test evaluation. The reserved partition is
not used for checkpoint selection, hyperparameter choices, architecture choices,
or posterior-sampling budget choices.

## Checkpoint selection

Within each run, select the saved checkpoint with the lowest `val_loss` on the
frozen development-validation split. `val_loss` is the configured weighted sum
of the active training objectives. Use the existing `best-loss.ckpt` monitor;
do not choose an epoch by looking at reserved-test results. Record the split
membership hash, objective weights and checkpoint hash with the evaluation.

For Q0/Q1, use matching measured inputs, topology, truth definitions, split,
L-GATr settings, neutrino CFM settings, optimizer settings, objective weights,
validation cadence and training budget. Compare their selected checkpoints using
the same validation and final-evaluation code. Report their parameter counts;
Q1's added query parameters are not capacity-matched.

For models with different active objectives, `val_loss` selects a checkpoint
within that run. It is not interpreted as a cross-model physics metric when the
objective weights differ. The Phase 1 Q0/Q1 comparison is the controlled shared
objective comparison; later partonic and coupled phases report their component
validation losses and predeclared physics metrics separately.

## Sealed evaluation

After the model/configuration is frozen using the development split, evaluate
the selected checkpoint on the reserved partition once with the declared
inference budget. Do not revise training, conditioning, posterior reduction,
or sample count based on those results. Any revision requires a new development
study and a separately reserved evaluation population.

Real-data Q0/Q1 selection remains deferred until the source neutrino-to-charge
side convention is verified. Synthetic neutrino fixtures are software checks
only and cannot satisfy this condition.
