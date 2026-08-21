"""Shared physics definitions for Transformer topology performance."""

from __future__ import annotations


TOPOLOGY_SPECS = {
    "ttbar1L": {
        "required_jet_roles": (1, 2, 3, 4),
        "jet_bins": ("4", "5", "6", "7", "8", "9+"),
        "components": {
            "blep": (("JET", 1),),
            "bhad": (("JET", 2),),
            "whad": (("JET", 3), ("JET", 4)),
            "tlep": (("JET", 1), ("LEPTON", 1), ("MET", 1)),
            "thad": (("JET", 2), ("JET", 3), ("JET", 4)),
        },
        "component_labels": {
            "blep": r"$b_{\mathrm{lep}}$",
            "bhad": r"$b_{\mathrm{had}}$",
            "whad": r"$W_{\mathrm{had}}$",
            "tlep": r"$t_{\mathrm{lep}}$",
            "thad": r"$t_{\mathrm{had}}$",
        },
    },
    "ttH": {
        "required_jet_roles": (1, 2, 3, 4, 5, 6),
        "jet_bins": ("6", "7", "8", "9", "10+"),
        "components": {
            "blep": (("JET", 4),),
            "bhad": (("JET", 1),),
            "whad": (("JET", 2), ("JET", 3)),
            "tlep": (("JET", 4), ("LEPTON", 1), ("MET", 1)),
            "thad": (("JET", 1), ("JET", 2), ("JET", 3)),
            "higgs": (("JET", 5), ("JET", 6)),
        },
        "component_labels": {
            "blep": r"$b_{\mathrm{lep}}$",
            "bhad": r"$b_{\mathrm{had}}$",
            "whad": r"$W_{\mathrm{had}}$",
            "tlep": r"$t_{\mathrm{lep}}$",
            "thad": r"$t_{\mathrm{had}}$",
            "higgs": r"$H\to b\bar{b}$",
        },
    },
}


def topology_spec(topology: str) -> dict:
    try:
        return TOPOLOGY_SPECS[str(topology)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Transformer topology {topology!r}; expected one of {tuple(TOPOLOGY_SPECS)}."
        ) from exc


def component_names(topology: str) -> tuple[str, ...]:
    return tuple(topology_spec(topology)["components"])


def component_column(component: str) -> str:
    return f"{component}_correct"


def complete_column(topology: str) -> str:
    return "event_correct"


def njet_bin(topology: str, n_jets: int) -> str:
    """Return the configured source-event jet category."""
    value = int(n_jets)
    bins = topology_spec(topology)["jet_bins"]
    first = int(bins[0].rstrip("+"))
    if value < first:
        raise ValueError(
            f"Source event has n_jets={value}, below the minimum configured {topology} category {bins[0]!r}."
        )
    for name in bins[:-1]:
        if value == int(name):
            return name
    return bins[-1]


def required_truth_members(topology: str) -> dict[str, tuple[tuple[str, int], ...]]:
    return dict(topology_spec(topology)["components"])


def fully_matched_from_labels(event_label: int, jet_labels, lepton_labels, met_labels, topology: str) -> bool:
    """Apply the exact fully matched definition to complete source-event labels."""
    if int(event_label) != 1:
        return False
    spec = topology_spec(topology)
    return (
        sum(int(value) == 1 for value in lepton_labels if value == value) == 1
        and sum(int(value) == 1 for value in met_labels if value == value) == 1
        and all(
            sum(int(value) == role for value in jet_labels if value == value) == 1
            for role in spec["required_jet_roles"]
        )
    )
