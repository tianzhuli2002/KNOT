# JointEventReconstruction

Particle-physics event reconstruction with topology-derived visible assignments and
conditional invisible-particle inference. The project is an independent research
repository: neither source repository is imported at runtime.

The frozen M0 baseline is the semantic-candidate Transformer from
ReconstructionAndSB. Its encoder, candidate construction, mean/max candidate encoder,
dynamic candidate attention, role summaries, semantic loss and constrained decoder are
isolated under `joint_event_reconstruction.m0` for controlled comparisons.

The maintained joint model composes:

- topology-compiled, role-normalised visible reconstruction with exact symmetry only
  inside repeated physical roles;
- hierarchy-preserving candidates (the actual child embedding is reused by its parent);
- an optional weight-shared recurrent Transformer over detector and topology tokens;
- optional differentiable soft candidate feedback, never a hard in-forward decode;
- a common Transformer denoiser for Gaussian VP diffusion or EDM;
- soft, hard-diagnostic or truth-oracle assignment conditioning;
- detached posterior-mean/variance feedback for a second visible reasoning round; and
- exact top-K complete assignments with `logZ` over the same non-null state space,
  followed by batched `B × K × M` invisible generation.

There is deliberately no neutrino-regression mode.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

The physics datasets and stable split files are explicit in `configs/m0_*.yaml`.
Missing files are fatal.

## Topology and configuration

`configs/topology/*.yaml` uses four constructs: a leaf `[SOURCE, truth_label]`, a
repeated exchangeable role containing several leaves, a composite reference by node
name, and an invisible `[NEUTRINO, index]` leaf. References compile into a DAG; config
declaration order is irrelevant. MET remains observed context and is not a decay
daughter.

The two primary configurations are `configs/experiments/ttbar1L.yaml` and
`configs/experiments/ttH1L.yaml`. Behaviour is controlled by the component switches
under `model.reasoning`, `model.reconstruction.feedback`, `model.invisible`,
`model.coupling`, and `inference`; stage names do not select code paths. Invalid
combinations fail during configuration validation.

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml

python scripts/smoke_all.py --topology ttbar1L --device cpu --events 8
python scripts/smoke_all.py --topology ttH1L --device cuda --events 16
```

## Invisible truth

The canonical sidecar contains integer `EVENT_ID [N]` and structured float32
`LABELS/NEUTRINO [N,Rnu]` fields `px`, `py`, `pz`. It is joined strictly by event ID;
row-wise fallback, missing IDs, duplicate IDs and malformed fields are fatal. Scaling
statistics are fit on the selected training split only and stored with checkpoints.

Real event-aligned neutrino truth is not currently available for these maintained
visible samples. The following command creates deliberately shuffled synthetic truth
for integration tests:

```bash
PYTHONPATH=src python scripts/data/make_synthetic_invisible.py \
  --experiment configs/experiments/ttbar1L.yaml \
  --output data/synthetic/ttbar1L.h5 --max-events 32
```

Synthetic invisible and coupled results are **pipeline validation only, not physics
performance**. They require `data.invisible.allow_synthetic: true` explicitly.

## M0 validation

```bash
python scripts/validate_m0_equivalence.py \
  --reference-root /mnt/iusers01/fatpou01/phy01/w00238tl/scratch/ReconstructionAndSB_ref \
  --checkpoint /path/to/best_reconstruction.ckpt --events 32

python scripts/evaluate_m0.py --checkpoint /path/to/best_reconstruction.ckpt \
  --max-events 4096 --device cuda
```

See `PROVENANCE.md` for exact source revisions and the local-state decision.
