"""Idempotent orchestration loop for the shadow-feature sweep.

Enumerates the scenario set for the chosen profile, skips scenarios whose
results already exist, executes the parameterized notebook via papermill for
the rest, retries transient failures, and loops until coverage is reached or
all remaining are marked failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import papermill as pm
import yaml

from scenarios import build_scenarios, scenario_id


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_failed(failed_path: Path) -> dict[str, dict[str, Any]]:
    if not failed_path.exists():
        return {}
    with open(failed_path) as f:
        return json.load(f)


def write_failed(failed_path: Path, failed: dict[str, dict[str, Any]]) -> None:
    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)


def is_done(results_dir: Path, sid: str) -> bool:
    p = results_dir / f"{sid}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except Exception:
        return False


def run_one(scenario: dict[str, Any], sid: str, orch: dict[str, Any]) -> None:
    out_nb = Path(orch["executed_dir"]) / f"{sid}.ipynb"
    Path(orch["executed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(orch["results_dir"]).mkdir(parents=True, exist_ok=True)
    parameters = {
        "scenario_id": sid,
        "regime": scenario["regime"],
        "n": scenario["n"],
        "p": scenario["p"],
        "delta": scenario["delta"],
        "n_real": scenario["n_real"],
        "n_noise": scenario["n_noise"],
        "control": scenario["control"],
        "backend": scenario["backend"],
        "n_repeats": scenario["n_repeats"],
        "results_dir": orch["results_dir"],
    }
    pm.execute_notebook(
        orch["notebook"],
        str(out_nb),
        parameters=parameters,
        kernel_name=orch["kernel"],
        progress_bar=False,
    )


def _compact(s: dict[str, Any]) -> str:
    return (
        f"regime={s['regime']} n={s['n']} p={s['p']} delta={s['delta']} "
        f"control={s['control']} backend={s['backend']} "
        f"n_noise={s['n_noise']} n_repeats={s['n_repeats']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", default=None, help="smoke | full (default from config.yaml)"
    )
    args = parser.parse_args()

    cfg = load_config()
    profile = args.profile or cfg["default_profile"]
    orch = cfg["orchestration"]

    results_dir = Path(orch["results_dir"])
    failed_path = Path(orch["failed_path"])
    max_passes = int(orch["max_passes"])
    max_attempts = int(orch["max_attempts_per_pass"])

    scenarios = build_scenarios(profile)
    targets = [(scenario_id(s), s) for s in scenarios]
    print(f"[sweep] profile={profile} target_count={len(targets)}")

    failed = load_failed(failed_path)

    for pass_idx in range(1, max_passes + 1):
        remaining = [
            (sid, s)
            for sid, s in targets
            if not is_done(results_dir, sid) and sid not in failed
        ]
        if not remaining:
            print(
                f"[sweep] pass {pass_idx}: nothing to do (all covered or failed)"
            )
            break
        covered_now = sum(1 for sid, _ in targets if is_done(results_dir, sid))
        print(
            f"[sweep] pass {pass_idx}: remaining={len(remaining)} "
            f"covered={covered_now} failed={len(failed)}"
        )
        for i, (sid, s) in enumerate(remaining, 1):
            t0 = time.time()
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    run_one(s, sid, orch)
                    elapsed = time.time() - t0
                    print(
                        f"[sweep]   [{i}/{len(remaining)}] {sid} OK "
                        f"({elapsed:.1f}s, attempt {attempt}) {_compact(s)}"
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[sweep]   [{i}/{len(remaining)}] {sid} FAIL "
                        f"attempt {attempt}: {exc!r}"
                    )
            if last_exc is not None:
                failed[sid] = {
                    "scenario": s,
                    "pass": pass_idx,
                    "exception": repr(last_exc),
                    "traceback": traceback.format_exception(
                        type(last_exc), last_exc, last_exc.__traceback__
                    ),
                }
                write_failed(failed_path, failed)
                print(
                    f"[sweep]   [{i}/{len(remaining)}] {sid} marked FAILED"
                )

    covered = sum(1 for sid, _ in targets if is_done(results_dir, sid))
    still_missing = [
        sid
        for sid, _ in targets
        if not is_done(results_dir, sid) and sid not in failed
    ]
    print(
        f"[sweep] DONE: covered={covered}/{len(targets)} "
        f"failed={len(failed)} still_missing={len(still_missing)}"
    )
    return 0 if not still_missing else 2


if __name__ == "__main__":
    sys.exit(main())
