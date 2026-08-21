"""Deterministic Transformer topology decoders."""

from __future__ import annotations

import math


DECODER_EPSILON = 1.0e-12


def _jets(candidate):
    return tuple(
        int(local)
        for source, local in zip(candidate["member_sources"], candidate["source_indices"])
        if source == "JET"
    )


def _members(candidate, source):
    return tuple(
        int(local)
        for member_source, local in zip(candidate["member_sources"], candidate["source_indices"])
        if member_source == source
    )


def _score(candidates, epsilon):
    return sum(math.log(max(float(candidate["probability"]), epsilon)) for candidate in candidates)


def _invalid():
    return {
        "reco_valid": False,
        "topology_score": float("nan"),
        "topology_confidence": float("nan"),
        "selected": {},
        "complete_topology_truth_correct": False,
    }


def _finish(selected, score, roles):
    tlep = selected["tlep"]
    thad = selected["thad"]
    whad = selected["Whad"]
    tlep_jets = _jets(tlep)
    thad_jets = set(_jets(thad))
    whad_jets = set(_jets(whad))
    if len(tlep_jets) != 1 or len(thad_jets - whad_jets) != 1:
        return _invalid()
    result = {
        "reco_valid": True,
        "topology_score": float(score),
        "topology_confidence": float(math.exp(score / len(selected))),
        "selected": selected,
        "blep": int(tlep_jets[0]),
        "bhad": int(next(iter(thad_jets - whad_jets))),
        "Wlep": {"LEPTON": _members(tlep, "LEPTON"), "MET": _members(tlep, "MET")},
        "Whad_daughters": tuple(_jets(whad)),
        "H_daughters": tuple(_jets(selected["H"])) if "H" in selected else (),
    }
    if roles.get("tlep_b"):
        tlep_b = max(roles["tlep_b"], key=lambda candidate: float(candidate["probability"]))
        result["tlep_b_agreement"] = bool(_jets(tlep_b) and _jets(tlep_b)[0] == result["blep"])
        result["selected_tlep_b"] = tlep_b
    required_roles = {"tlep", "thad", "Whad"}
    if "H" in selected:
        required_roles.add("H")
    result["complete_topology_truth_correct"] = all(
        bool(selected[role].get("truth_correct", False)) for role in required_roles
    )
    return result


def decode_ttbar(roles, epsilon=DECODER_EPSILON):
    """Decode the highest-scoring valid ttbar1L assignment."""
    best = None
    for tlep in roles.get("tlep", ()):
        tlep_jets = set(_jets(tlep))
        if len(tlep_jets) != 1:
            continue
        for thad in roles.get("thad", ()):
            thad_jets = set(_jets(thad))
            if tlep_jets & thad_jets:
                continue
            for whad in roles.get("Whad", ()):
                whad_jets = set(_jets(whad))
                if len(whad_jets) != 2 or not whad_jets < thad_jets or len(thad_jets - whad_jets) != 1:
                    continue
                score = _score((tlep, thad, whad), epsilon)
                if best is None or score > best[0]:
                    best = score, {"tlep": tlep, "thad": thad, "Whad": whad}
    return _invalid() if best is None else _finish(best[1], best[0], roles)


def decode_tth(roles, epsilon=DECODER_EPSILON):
    """Decode the highest-scoring valid ttH assignment."""
    best = None
    for tlep in roles.get("tlep", ()):
        tlep_jets = set(_jets(tlep))
        if len(tlep_jets) != 1:
            continue
        for thad in roles.get("thad", ()):
            thad_jets = set(_jets(thad))
            if tlep_jets & thad_jets:
                continue
            for whad in roles.get("Whad", ()):
                whad_jets = set(_jets(whad))
                if len(whad_jets) != 2 or not whad_jets < thad_jets or len(thad_jets - whad_jets) != 1:
                    continue
                for higgs in roles.get("H", ()):
                    higgs_jets = set(_jets(higgs))
                    if len(higgs_jets) != 2 or higgs_jets & thad_jets or higgs_jets & tlep_jets:
                        continue
                    score = _score((tlep, thad, whad, higgs), epsilon)
                    if best is None or score > best[0]:
                        best = score, {"tlep": tlep, "thad": thad, "Whad": whad, "H": higgs}
    return _invalid() if best is None else _finish(best[1], best[0], roles)
