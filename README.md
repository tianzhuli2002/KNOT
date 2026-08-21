# JointEventReconstruction

Particle-physics event reconstruction with topology-derived visible assignments and
conditional invisible-particle inference. The project is an independent research
repository: neither source repository is imported at runtime.

The first maintained baseline, M0, is the frozen semantic-candidate Transformer from
ReconstructionAndSB. Its encoder, candidate construction, mean/max candidate encoder,
dynamic candidate attention, role summaries, semantic loss and constrained decoder are
isolated under `joint_event_reconstruction.m0` for controlled comparisons.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

The physics datasets and stable split files are explicit in `configs/m0_*.yaml`.
Missing files are fatal.

## M0 validation

```bash
python scripts/validate_m0_equivalence.py \
  --reference-root /mnt/iusers01/fatpou01/phy01/w00238tl/scratch/ReconstructionAndSB_ref \
  --checkpoint /path/to/best_reconstruction.ckpt --events 32

python scripts/evaluate_m0.py --checkpoint /path/to/best_reconstruction.ckpt \
  --max-events 4096 --device cuda
```

See `PROVENANCE.md` for exact source revisions and the local-state decision.
