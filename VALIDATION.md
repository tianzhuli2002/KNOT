# Bounded implementation checks

Source commit: `ec54f6b13817009b5ec0472e084a034f044980a4`. CPU only; sealed_test_used=false.

```json
{
  "M": {
    "loss": 6.695586204528809,
    "gradients": {
      "event_model.visible": 1.9374546050756503,
      "event_model.topology_queries": 1.28590881273161,
      "event_model.reconstruction": 0.1947040109009217,
      "parent_generator": 5.451991412497831
    },
    "encoder_calls_per_posterior": 1,
    "checkpoint_reload": true,
    "inactive_objectives_skipped": true,
    "invisible_query_projection_gradient": 0.020404726183091186,
    "joint_state_interaction": null,
    "posterior_shape": [
      4,
      3,
      2,
      4
    ]
  },
  "C": {
    "loss": 6.944906711578369,
    "gradients": {
      "event_model.visible": 0.8378139722782234,
      "event_model.topology_queries": 0.810784347856287,
      "event_model.reconstruction": 0.3471437423104581,
      "parent_generator": 4.3875328406148135
    },
    "encoder_calls_per_posterior": 1,
    "checkpoint_reload": true,
    "inactive_objectives_skipped": true,
    "invisible_query_projection_gradient": 0.004620998733307592,
    "joint_state_interaction": null,
    "posterior_shape": [
      4,
      3,
      2,
      4
    ]
  },
  "J": {
    "loss": 4.815422534942627,
    "gradients": {
      "event_model.visible": 4.074302123339972,
      "event_model.topology_queries": 2.817397714696112,
      "event_model.reconstruction": 2.2896811265301005,
      "joint_continuous": 3.416476837089881
    },
    "encoder_calls_per_posterior": 1,
    "checkpoint_reload": true,
    "inactive_objectives_skipped": true,
    "invisible_query_projection_gradient": 0.6562619970990475,
    "joint_state_interaction": true,
    "posterior_shape": [
      4,
      3,
      2,
      4
    ]
  },
  "source_commit": "ec54f6b13817009b5ec0472e084a034f044980a4",
  "branch": "research/knot-implementation-v1",
  "worktree": "/net/scratch/w00238tl/knot",
  "sealed_test_used": false,
  "population": "32 synthetic events, 24 train / 8 validation; no test rows",
  "knot_c_coherent_ancestral_sampling": "PASS",
  "knot_j_joint_continuous_sampling": "PASS",
  "configs": {
    "configs/research/knot_m.yaml": "e87f7f4ad3c9bab793bde5337725566530eff77813950bc5107c25d657408eca",
    "configs/research/knot_c_jer.yaml": "db990c2df6dee673183189e9f35a0e0d2a64af0d2e388bf06e7d6f1b9cb486a9",
    "configs/research/knot_j.yaml": "dbb7d9ee1a3a26bcaf6aa00f7ffa283b4013a6b3c6ac1a5d2bdbc925a0efbc21",
    "configs/research/knot_j_jer.yaml": "1f5856c45f5c054a92c871b2332fc7d76f2b3603e38d28964403308d8fa08c04",
    "configs/research/knot_c.yaml": "3669db4a5e165487c122414a616bc498a2140d57cfd662b7f97ec8fd836dd983",
    "configs/research/knot_m_jer.yaml": "a983fa9d726a951d1190b2eadaf0d7881aeb657a0309396b80402bc00affd6bd"
  },
  "inputs": {
    "/net/scratch/w00238tl/knot/results/experiments/knot/development/measured.h5": "d2bb76f0555e6ea1820ee97248ccca0ca2091928cdd9cf402050b09edee98bee",
    "/net/scratch/w00238tl/knot/results/experiments/knot/development/parents.h5": "8affdf33119af9b215ff610efb2567e6da9fc3dfeeaacd1ec9b2490e3a82066b"
  },
  "syntax_import_config": "PASS",
  "ruff": "PASS",
  "targeted_existing_scaler_tests": "3 PASS",
  "cli_smoke": "PASS: measured-only reload prediction"
}
```

No collider performance or parameter-constraint comparison was run.
