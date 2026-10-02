"""Cheap full-architecture execution check for common posterior/oracle/spin paths."""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from run_knot_cpu_physics_pilot import model, setup
from run_knot_real_cpu_smoke import prediction_batch

from joint_event_reconstruction.knot_physics import spin_observables


def main():
    torch.set_num_threads(1)
    config, visible, data = setup()
    batch = data.structured_collator([data.val_dataset[i] for i in range(4)])
    module = model(config, visible, data, "C").eval()
    with torch.no_grad():
        predicted = module.posterior(
            prediction_batch(batch),
            samples=4,
            generator=torch.Generator().manual_seed(17),
        )
        oracle1 = module.posterior(
            batch,
            samples=4,
            generator=torch.Generator().manual_seed(17),
            oracle="truth_assignment_truth_neutrino",
        )
        oracle2 = module.posterior(
            batch,
            samples=4,
            generator=torch.Generator().manual_seed(17),
            oracle="predicted_assignment_truth_neutrino",
        )
    assert torch.equal(predicted.assignment_state, oracle2.assignment_state)
    assert torch.equal(
        oracle1.assignment_state,
        batch["structured"]["assignments"]["truth_state"][:, None].expand(-1, 4),
    )
    assert torch.equal(oracle1.invisible, oracle2.invisible)
    assert not torch.equal(predicted.invisible, oracle2.invisible)
    positions = (
        (batch["type_ids"][0] == data.schema.source_type_ids["LEPTON"])
        .nonzero()
        .flatten()
    )
    observables = spin_observables(
        predicted.parents.numpy(),
        batch["physics_vectors"][:, positions].numpy()[:, None],
    )
    assert all(
        np.isfinite(value).all() and value.shape == (4, 4)
        for value in observables.values()
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "posterior.h5"
        predicted.write(path)
        with h5py.File(path) as saved:
            assert saved.attrs["posterior_semantics"] == "ancestral"
            assert saved["assignment_log_probability"].shape == (4, 4)
    print(
        "full-architecture C default/oracle endpoints, parent spin, common HDF5: PASS"
    )


if __name__ == "__main__":
    main()
