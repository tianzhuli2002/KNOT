"""Check recorded unsupported events directly; never rescan the population."""

import copy
import json
from dataclasses import replace

import numpy as np
import torch
from omegaconf import OmegaConf
from run_knot_cpu_physics_pilot import OUT, ROOT, setup

from joint_event_reconstruction.data.structured import StructuredCachedDataset
from joint_event_reconstruction.knot import KnotModule


def main():
    torch.set_num_threads(1)
    config, visible, data = setup()
    previous = json.loads(
        (
            ROOT
            / "results/experiments/knot/scientific-closure-run2/real_sample/model_native_assignment_support_train_validation.json"
        ).read_text()
    )
    examples = previous["unsupported_event_id_examples"][:3]
    ids = data.cache_payload["event_id"].numpy()
    rows = np.array([int(np.flatnonzero(ids == event)[0]) for event in examples])
    subset = StructuredCachedDataset(data.cache_payload, rows)
    batch = data.structured_collator(subset.__getitems__(list(range(len(rows)))))
    assert not batch["visible_assignment_support"].any()
    assert not batch["structured"]["objective_support"].any()
    research = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs/research/knot_m_production.yaml"), resolve=True
    )
    config = copy.deepcopy(config)
    config.model.parent_cfm.enabled = False
    module = KnotModule(
        config,
        visible,
        research=research,
        invisible_scaler=data.invisible_scaler,
        partonic_scaler=data.partonic_scaler,
        synthetic_data=False,
    )
    output = module(batch)
    assert output["assignment_loss"] == 0 and torch.isfinite(output["loss"])
    try:
        KnotModule(
            config,
            visible,
            research=research,
            invisible_scaler=data.invisible_scaler,
            partonic_scaler=replace(
                data.partonic_scaler, role_names=("antitop", "top")
            ),
            synthetic_data=False,
        )
    except ValueError as error:
        assert "compiled parent roots" in str(error)
    else:
        raise AssertionError("Reversed parent scaler roles were silently accepted")
    report = {
        "recorded_unsupported_EVENT_IDs": examples,
        "assignment_loss": float(output["assignment_loss"].detach()),
        "finite_continuous_loss": True,
        "reversed_parent_order_rejected": True,
        "parent_tensor_order": ["top", "antitop"],
        "population_rescanned": False,
        "sealed_test_used": False,
    }
    (OUT / "preflight_support.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
