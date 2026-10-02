"""Describe only the selected pilot population, using its existing cache/truth."""

import numpy as np
import torch
from run_knot_cpu_physics_pilot import OUT, save_json, setup
from torch.utils.data import DataLoader

from joint_event_reconstruction.knot_physics import mass


def main():
    torch.set_num_threads(1)
    _, _, data = setup()
    report = {}
    for split, dataset in [
        ("train", data.train_dataset),
        ("validation", data.val_dataset),
    ]:
        counts = {"emu": 0, "ee": 0, "mumu": 0}
        jets, mtt, states, support = [], [], [], []
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=data.structured_collator,
            num_workers=0,
        )
        for batch in loader:
            mask = batch["type_ids"] == data.schema.source_type_ids["LEPTON"]
            identity = batch["tokens"][..., data.schema.names.index("id:none")][
                mask
            ].reshape(-1, 2)
            n_e = (identity == 2).sum(-1)
            assert ((identity == 2) | (identity == 3)).all()
            for channel, number in [("emu", 1), ("ee", 2), ("mumu", 0)]:
                counts[channel] += int((n_e == number).sum())
            jets.append((~batch["padding_mask"][:, :8]).sum(-1).numpy())
            mtt.append(mass(batch["partonic_truth"].numpy().sum(axis=1)))
            states.append(
                batch["structured"]["assignments"]["valid_mask"].sum(-1).numpy()
            )
            support.append(batch["visible_assignment_support"].numpy())
        jets, mtt, states, support = map(np.concatenate, (jets, mtt, states, support))
        report[split] = {
            "events": len(jets),
            "channels": counts,
            "jet_multiplicity_counts": {
                str(i): int((jets == i).sum()) for i in range(2, 9)
            },
            "truth_mttbar_quantiles_GeV": np.quantile(
                mtt, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            "feasible_assignment_count_quantiles": np.quantile(
                states, [0, 0.1, 0.5, 0.9, 1]
            ).tolist(),
            "truth_assignment_supported": int(support.sum()),
            "truth_assignment_unsupported": int((~support).sum()),
        }
    save_json(OUT / "population_summary.json", report)
    print(report)


if __name__ == "__main__":
    main()
