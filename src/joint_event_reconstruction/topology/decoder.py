"""Exact decoder over complete, hierarchy-consistent root assignments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, log
from typing import Mapping, Sequence

from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.spec import TopologySpec


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidateRecord
    probability: float


@dataclass(frozen=True)
class CompleteAssignment:
    roots: tuple[tuple[str, int], ...]
    log_score: float
    leaf_ownership: frozenset[tuple[str, int]]
    truth: bool


@dataclass(frozen=True)
class DecodeResult:
    assignments: tuple[CompleteAssignment, ...]
    log_normalisation: float
    valid_assignment_count: int


def decode_topk(
    spec: TopologySpec,
    candidates: Mapping[str, Sequence[ScoredCandidate]],
    *,
    k: int = 1,
    epsilon: float = 1.0e-12,
) -> tuple[CompleteAssignment, ...]:
    return decode_complete(spec, candidates, k=k, epsilon=epsilon).assignments


def decode_complete(
    spec: TopologySpec,
    candidates: Mapping[str, Sequence[ScoredCandidate]],
    *,
    k: int = 1,
    epsilon: float = 1.0e-12,
) -> DecodeResult:
    """Decode non-null complete root assignments and normalise that same state space."""
    if k <= 0:
        raise ValueError("Decoder k must be positive")
    missing = [root for root in spec.roots if root not in candidates]
    if missing:
        raise KeyError(f"Decoder is missing root candidate pools {missing}")
    complete = []
    for selection in product(*(candidates[root] for root in spec.roots)):
        ownership: set[tuple[str, int]] = set()
        valid = True
        score = 0.0
        for item in selection:
            if ownership & item.candidate.leaf_ownership:
                valid = False
                break
            if not 0.0 <= item.probability <= 1.0:
                raise ValueError(f"Candidate probability must lie in [0,1], found {item.probability}")
            ownership.update(item.candidate.leaf_ownership)
            score += log(max(item.probability, epsilon))
        if valid:
            complete.append(CompleteAssignment(
                roots=tuple((root, item.candidate.index) for root, item in zip(spec.roots, selection)),
                log_score=score, leaf_ownership=frozenset(ownership),
                truth=all(item.candidate.truth for item in selection),
            ))
    complete.sort(key=lambda item: (-item.log_score, item.roots))
    if not complete:
        return DecodeResult((), float("-inf"), 0)
    maximum = complete[0].log_score
    log_normalisation = maximum + log(sum(exp(item.log_score - maximum) for item in complete))
    return DecodeResult(tuple(complete[:k]), log_normalisation, len(complete))
