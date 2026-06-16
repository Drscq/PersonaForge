from __future__ import annotations

import math


def js_divergence(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence using natural logs; lower is better."""

    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    p_vec = [max(eps, p.get(key, 0.0)) for key in keys]
    q_vec = [max(eps, q.get(key, 0.0)) for key in keys]
    p_vec = _renormalize(p_vec)
    q_vec = _renormalize(q_vec)
    m_vec = [(a + b) / 2 for a, b in zip(p_vec, q_vec)]
    return 0.5 * _kl(p_vec, m_vec) + 0.5 * _kl(q_vec, m_vec)


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def _renormalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]

