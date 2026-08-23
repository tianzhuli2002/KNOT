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
  followed by batched `B × K × M` invisible generation; and
- an optional compact compatibility head for visible/invisible reranking. Its logit is
  kept separate from the visible score and is not presented as a likelihood.

There is deliberately no neutrino-regression mode.

## Environment

```bash
conda env create -f requirements.yaml
conda activate joint-event-reconstruction
python -m pip install -e .
```

The physics datasets and stable split files are explicit in `configs/real_*.yaml`.
Missing files are fatal.

## Authoritative real data

The maintained joint products are streamed directly from the event-aligned VyPER
Run-3 HDF5 files; unrelated SB and VyPER rows are never joined. Semileptonic data have
one role, `tlep.nu`, and dilepton data have two ordered roles,
`(top_e.nu, top_mu.nu) = (nu_e, nu_mu)`. The electron and muon branches remain
distinguishable.

```bash
python scripts/data/convert_vyper_joint.py --topology ttbar1L \
  --output data/real/ttbar1L_canonical.h5
python scripts/data/convert_vyper_joint.py --topology ttbar2L \
  --output data/real/ttbar2L_canonical.h5

python -m joint_event_reconstruction.cli.run validate-data \
  --experiment configs/experiments/ttbar2L.yaml
```

`EVENT_ID = source_partition * 2^32 + source_row`, where partition zero is the
source even file and partition one is the source odd file. The even partition is split
deterministically into 95% training and 5% validation; the independent odd partition
is test. Invisible scaling is fitted independently per role and Cartesian component
on the selected training events only.

## Topology and configuration

`configs/topology/*.yaml` uses four constructs: a leaf `[SOURCE, truth_label]`, a
repeated exchangeable role containing several leaves, a composite reference by node
name, and an invisible `[NEUTRINO, index]` leaf. References compile into a DAG; config
declaration order is irrelevant. MET remains observed context and is not a decay
daughter.

The primary configurations are `configs/experiments/ttbar1L.yaml`,
`configs/experiments/ttbar2L.yaml`, and `configs/experiments/ttH1L.yaml`. Behaviour is controlled by the component switches
under `model.reasoning`, `model.reconstruction.feedback`, `model.invisible`,
`model.coupling`, and `inference`; stage names do not select code paths. Invalid
combinations fail during configuration validation.

```bash
python -m joint_event_reconstruction.cli.run train \
  --experiment configs/experiments/ttbar1L.yaml

python scripts/smoke_all.py --topology ttbar1L --device cpu --events 8
python scripts/smoke_all.py --topology ttbar2L --real --device cuda --events 8
python scripts/smoke_all.py --topology ttH1L --device cuda --events 16
```

Prediction and evaluation use one CLI and a validated, atomically replaced HDF5
product:

```bash
python -m joint_event_reconstruction.cli.run predict \
  --experiment configs/experiments/ttbar2L.yaml --checkpoint results/run/checkpoints/best-loss.ckpt \
  --split test --output results/run/test_predictions.h5 --device cuda
python -m joint_event_reconstruction.cli.run evaluate \
  --experiment configs/experiments/ttbar2L.yaml --predictions results/run/test_predictions.h5
```

## Invisible truth and software smoke data

The canonical sidecar contains integer `EVENT_ID [N]` and structured float32
`LABELS/NEUTRINO [N,Rnu]` fields `px`, `py`, `pz`. It is joined strictly by event ID;
row-wise fallback, missing IDs, duplicate IDs and malformed fields are fatal. Scaling
statistics are fit on the selected training split only and stored with checkpoints.

Real event-aligned neutrino truth is available for the canonical ttbar1L and selected
e-mu ttbar2L products. The following command remains available solely for fast,
deliberately shuffled integration tests of the frozen visible cache:

```bash
PYTHONPATH=src python scripts/data/make_synthetic_invisible.py \
  --experiment configs/experiments/ttbar1L.yaml \
  --visible-config configs/m0_ttbar1L.yaml \
  --tensor-cache /path/to/frozen/cache.pt \
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
