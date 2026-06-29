import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scenarios import (
    FULL_PROFILE,
    SMOKE_PROFILE,
    build_scenarios,
    scenario_id,
)


def test_scenario_id_deterministic():
    p1 = {
        "regime": "A",
        "n": 1000,
        "p": 0.1,
        "delta": 0.5,
        "n_real": 5,
        "n_noise": 5,
        "control": False,
        "backend": "xgboost",
        "n_repeats": 25,
    }
    p2 = dict(p1)
    assert scenario_id(p1) == scenario_id(p2)


def test_scenario_id_differs_on_any_param_change():
    base = {
        "regime": "A",
        "n": 1000,
        "p": 0.1,
        "delta": 0.5,
        "n_real": 5,
        "n_noise": 5,
        "control": False,
        "backend": "xgboost",
        "n_repeats": 25,
    }
    sid = scenario_id(base)
    for key in base:
        modified = dict(base)
        if isinstance(base[key], bool):
            modified[key] = not base[key]
        elif isinstance(base[key], int):
            modified[key] = base[key] + 1
        elif isinstance(base[key], float):
            modified[key] = base[key] + 0.01
        else:
            modified[key] = base[key] + "_x"
        assert scenario_id(modified) != sid, f"id collision on {key}"


def test_smoke_profile_is_small():
    scenarios = build_scenarios(SMOKE_PROFILE)
    assert 5 <= len(scenarios) <= 40


def test_full_profile_covers_both_regimes():
    scenarios = build_scenarios(FULL_PROFILE)
    regimes = {s["regime"] for s in scenarios}
    assert regimes == {"A", "B"}


def test_full_profile_has_pure_noise_control():
    scenarios = build_scenarios(FULL_PROFILE)
    assert any(s["control"] for s in scenarios)
    assert any(not s["control"] for s in scenarios)


def test_full_profile_includes_n_noise_20_robustness():
    scenarios = build_scenarios(FULL_PROFILE)
    assert any(s["n_noise"] == 20 for s in scenarios)


def test_full_profile_includes_sklearn_cross_check():
    scenarios = build_scenarios(FULL_PROFILE)
    backends = {s["backend"] for s in scenarios}
    assert "sklearn_gbm" in backends


def test_degenerate_scenarios_excluded():
    scenarios = build_scenarios(FULL_PROFILE)
    for s in scenarios:
        n_min = round(s["n"] * s["p"])
        assert n_min >= 5, f"degenerate scenario kept: {s}"


def test_n_min_overlap_between_regimes():
    scenarios = build_scenarios(FULL_PROFILE)
    a_n_min = {round(s["n"] * s["p"]) for s in scenarios if s["regime"] == "A"}
    b_n_min = {round(s["n"] * s["p"]) for s in scenarios if s["regime"] == "B"}
    assert min(a_n_min) <= max(b_n_min)
    assert min(b_n_min) <= max(a_n_min)


def test_repeats_reduced_for_large_n():
    scenarios = build_scenarios(FULL_PROFILE)
    for s in scenarios:
        if s["n"] >= 50000:
            assert s["n_repeats"] <= 15
