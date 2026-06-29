"""Scenario grid construction for the shadow-feature experiment.

Each scenario is a dict with these keys (all hash inputs for scenario_id):
    regime, n, p, delta, n_real, n_noise, control, backend, n_repeats
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterable


def scenario_id(params: dict[str, Any]) -> str:
    """Stable 12-char id from a scenario param dict."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _repeats_for(n: int, base: int) -> int:
    return min(base, 15) if n >= 50000 else base


def _cross(
    regime: str,
    ns: Iterable[int],
    ps: Iterable[float],
    deltas: Iterable[float],
    *,
    n_real: int = 5,
    n_noise: int = 5,
    control_too: bool = True,
    backend: str = "xgboost",
    base_repeats: int = 25,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, p, delta in itertools.product(ns, ps, deltas):
        if round(n * p) < 5:
            continue
        for control in ([False, True] if control_too else [False]):
            out.append(
                {
                    "regime": regime,
                    "n": int(n),
                    "p": float(p),
                    "delta": float(delta),
                    "n_real": int(n_real),
                    "n_noise": int(n_noise),
                    "control": bool(control),
                    "backend": backend,
                    "n_repeats": _repeats_for(n, base_repeats),
                }
            )
    return out


SMOKE_PROFILE = "smoke"
FULL_PROFILE = "full"


def _smoke() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(
        _cross("A", ns=[500, 1000], ps=[0.10, 0.20], deltas=[0.5], base_repeats=2)
    )
    scenarios.extend(
        _cross("B", ns=[5000], ps=[0.01], deltas=[0.5], base_repeats=2)
    )
    scenarios.extend(
        _cross(
            "A",
            ns=[1000],
            ps=[0.10],
            deltas=[0.5],
            n_noise=20,
            control_too=False,
            base_repeats=2,
        )
    )
    scenarios.extend(
        _cross(
            "A",
            ns=[1000],
            ps=[0.10],
            deltas=[0.5],
            control_too=False,
            backend="sklearn_gbm",
            base_repeats=2,
        )
    )
    return _dedupe(scenarios)


def _full() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    scenarios.extend(
        _cross(
            "A",
            ns=[500, 1000, 2500, 5000],
            ps=[0.05, 0.10, 0.20],
            deltas=[0.3, 0.5, 1.0],
        )
    )
    scenarios.extend(
        _cross(
            "B",
            ns=[10000, 50000, 200000],
            ps=[0.001, 0.005, 0.01],
            deltas=[0.3, 0.5, 1.0],
        )
    )
    scenarios.extend(
        _cross(
            "A",
            ns=[500, 1000, 2500, 5000],
            ps=[0.05, 0.10, 0.20],
            deltas=[0.5],
            n_noise=20,
            control_too=False,
        )
    )
    scenarios.extend(
        _cross(
            "B",
            ns=[10000, 50000, 200000],
            ps=[0.001, 0.005, 0.01],
            deltas=[0.5],
            n_noise=20,
            control_too=False,
        )
    )
    scenarios.extend(
        _cross(
            "A",
            ns=[1000, 2500, 5000],
            ps=[0.10, 0.20],
            deltas=[0.5],
            control_too=False,
            backend="sklearn_gbm",
        )
    )
    return _dedupe(scenarios)


def _dedupe(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        seen[scenario_id(s)] = s
    return list(seen.values())


def build_scenarios(profile: str) -> list[dict[str, Any]]:
    if profile == SMOKE_PROFILE:
        return _smoke()
    if profile == FULL_PROFILE:
        return _full()
    raise ValueError(f"unknown profile: {profile!r}")


if __name__ == "__main__":
    import sys

    prof = sys.argv[1] if len(sys.argv) > 1 else SMOKE_PROFILE
    scenarios = build_scenarios(prof)
    print(f"profile={prof} n_scenarios={len(scenarios)}")
    for s in scenarios[:5]:
        print(scenario_id(s), s)
