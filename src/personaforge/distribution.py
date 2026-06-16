from __future__ import annotations

from collections import Counter
from typing import Iterable

from personaforge.schema import Interaction, Signature, UserLog
from personaforge.signature import extract_signature


Distribution = dict[str, float]


def distribution_from_logs(logs: Iterable[UserLog]) -> Distribution:
    return normalize(Counter(extract_signature(log).key() for log in logs))


def distribution_from_interactions(interactions: Iterable[Interaction]) -> Distribution:
    return normalize(Counter(interaction.signature.key() for interaction in interactions))


def distribution_from_signatures(signatures: Iterable[Signature]) -> Distribution:
    return normalize(Counter(signature.key() for signature in signatures))


def normalize(counts: Counter[str] | dict[str, float]) -> Distribution:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}

