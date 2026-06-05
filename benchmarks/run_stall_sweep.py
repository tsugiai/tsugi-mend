"""Seeded stall-sweep benchmark driver for tsugi-mend.

The sweep exercises a reproducible peer-straggler injector in the benchmark
harness. Each grid point runs n>=5 paired baseline/sdk trials, asserts
bit-exact loss equivalence on every trial, drops the fastest and slowest
per-run uplift, then writes the uplift-vs-injected-stall curve as a public-safe
result bundle.

Quick smoke:

    python benchmarks/run_stall_sweep.py --quick
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch

# Allow `python benchmarks/run_stall_sweep.py` from the repo root without an
# editable install (mirrors benchmarks/run_paired.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.metrics import aggregate_seeded_uplift  # noqa: E402
from benchmarks.run_paired import BenchConfig, CELLS, RESULTS_ROOT, run_cell  # noqa: E402
from tsugi_mend.failslow import FailSlowDetector  # noqa: E402

FULL_DELAYS_MS = (0, 50, 100, 250, 500, 1000)
FULL_STRAGGLER_COUNTS = (0, 1, 2, 4)
QUICK_DELAYS_MS = (0, 50)
QUICK_STRAGGLER_COUNTS = (0, 1)

RunCell = Callable[[BenchConfig], Optional[dict[str, Any]]]


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        return ()
    try:
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers; got {value!r}"
        ) from exc


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence")
    return float(statistics.fmean(values))


def _peer_straggler_ranks(count: int, ranks: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError(f"straggler count must be >= 0; got {count}")
    if count == 0:
        return ()
    if count > ranks - 1:
        raise ValueError(
            f"straggler count {count} needs at least {count + 1} ranks so rank 0 "
            "can observe a slow peer; increase --ranks"
        )
    return tuple(range(1, count + 1))


def _base_config(args: argparse.Namespace) -> BenchConfig:
    quick = bool(args.quick)
    cfg = replace(CELLS[args.cell]) if args.cell in CELLS else BenchConfig(cell=args.cell)
    cfg.cell = args.cell
    cfg.launch = "selfspawn"
    cfg.backend = "gloo"
    cfg.ranks = args.ranks if args.ranks is not None else (2 if quick else 5)
    cfg.steps = args.steps if args.steps is not None else (18 if quick else cfg.steps)
    cfg.warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else (2 if quick else cfg.warmup_steps)
    )
    cfg.sync_period_steps = (
        args.sync_period_steps
        if args.sync_period_steps is not None
        else (12 if quick else cfg.sync_period_steps)
    )
    cfg.apply_lag_steps = (
        args.apply_lag_steps
        if args.apply_lag_steps is not None
        else (4 if quick else cfg.apply_lag_steps)
    )
    cfg.seed = args.seed
    cfg.batch = args.batch if args.batch is not None else (16 if quick else cfg.batch)
    cfg.in_dim = args.in_dim if args.in_dim is not None else (32 if quick else cfg.in_dim)
    cfg.hidden = args.hidden if args.hidden is not None else (64 if quick else cfg.hidden)
    cfg.out_dim = args.out_dim if args.out_dim is not None else (16 if quick else cfg.out_dim)
    cfg.sequence_length = (
        args.sequence_length
        if args.sequence_length is not None
        else (32 if quick else cfg.sequence_length)
    )
    cfg.simulated_merge_delay_ms = args.simulated_merge_delay_ms
    cfg.bootstrap_resamples = args.step_bootstrap_resamples
    cfg.process_group_timeout_s = args.process_group_timeout_s
    cfg.hardware_label = (
        "local CPU (gloo); reproducible stall-sweep quick smoke"
        if quick
        else "local CPU (gloo); reproducible stall-sweep full grid"
    )
    cfg._include_rank_timings = True
    return cfg


def _detect_failslow_events(
    per_rank_step_ms: dict[str, dict[str, list[float]]],
    *,
    path: str,
    window_steps: int,
    zscore_threshold: float,
    min_samples: int,
) -> dict[str, Any]:
    detector = FailSlowDetector(
        window_steps=window_steps,
        zscore_threshold=zscore_threshold,
        min_samples=min_samples,
    )
    events: list[dict[str, Any]] = []
    max_z_by_rank: dict[str, float] = {}
    for rank_id in sorted(per_rank_step_ms, key=int):
        samples = per_rank_step_ms[rank_id][path]
        for step, step_time_ms in enumerate(samples):
            decision = detector.observe(f"rank-{rank_id}", step_time_ms)
            max_z_by_rank[rank_id] = max(
                max_z_by_rank.get(rank_id, float("-inf")),
                float(decision.z_score),
            )
            if decision.is_slow:
                events.append(
                    {
                        "rank": int(rank_id),
                        "step": step,
                        "step_time_ms": step_time_ms,
                        "z_score": decision.z_score,
                        "window_mean_ms": decision.window_mean_ms,
                        "window_std_ms": decision.window_std_ms,
                        "reason": decision.reason,
                    }
                )
    flagged = sorted({event["rank"] for event in events})
    return {
        "path": path,
        "flagged_ranks": flagged,
        "max_z_score_by_rank": max_z_by_rank,
        "events": events,
    }


def _merge_detector_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flagged = sorted({rank for report in reports for rank in report["flagged_ranks"]})
    max_z: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    for seed_index, report in enumerate(reports):
        for rank_id, z_score in report["max_z_score_by_rank"].items():
            max_z[rank_id] = max(max_z.get(rank_id, float("-inf")), float(z_score))
        for event in report["events"]:
            events.append({"seed_index": seed_index, **event})
    return {
        "flagged_ranks": flagged,
        "max_z_score_by_rank": max_z,
        "events": events,
    }


def _path_metric_mean(bundles: Sequence[dict[str, Any]], path: str, metric: str) -> float:
    return _mean([float(bundle["metrics"][path][metric]) for bundle in bundles])


def _run_grid_point(
    base_cfg: BenchConfig,
    *,
    delay_ms: int,
    straggler_count: int,
    n_seeds: int,
    seed_start: int,
    aggregate_bootstrap_resamples: int,
    detector_window_steps: int,
    detector_zscore_threshold: float,
    detector_min_samples: int,
    run_cell_fn: RunCell,
) -> dict[str, Any]:
    straggler_ranks = _peer_straggler_ranks(straggler_count, base_cfg.ranks)
    seed_bundles: list[dict[str, Any]] = []
    per_run_uplifts: list[float] = []
    per_seed: list[dict[str, Any]] = []
    detector_reports = {"baseline": [], "sdk": []}
    max_abs_loss_diff = 0.0

    for seed_index in range(n_seeds):
        seed = seed_start + seed_index
        cfg = replace(
            base_cfg,
            seed=seed,
            straggler_delay_ms=delay_ms,
            straggler_ranks=straggler_ranks,
            path_order="baseline_sdk" if seed_index % 2 == 0 else "sdk_baseline",
            _include_rank_timings=True,
        )
        bundle = run_cell_fn(cfg)
        if bundle is None:
            raise RuntimeError("stall sweep requires the selfspawn rank-0 result bundle")
        bit_exact = bundle["bit_exact_loss_equivalence"]
        if not bit_exact["passed"]:
            raise RuntimeError(
                f"bit-exact loss equivalence failed at delay={delay_ms}ms, "
                f"straggler_count={straggler_count}, seed={seed}"
            )
        max_abs_loss_diff = max(max_abs_loss_diff, float(bit_exact["max_abs_loss_diff"]))

        uplift = float(bundle["metrics"]["uplift"]["tokens_per_second_pct"])
        per_run_uplifts.append(uplift)
        seed_bundles.append(bundle)
        per_seed.append(
            {
                "seed": seed,
                "path_order": cfg.path_order,
                "uplift_pct": uplift,
                "baseline_tokens_per_second": bundle["metrics"]["baseline"][
                    "tokens_per_second"
                ],
                "sdk_tokens_per_second": bundle["metrics"]["sdk"]["tokens_per_second"],
                "max_abs_loss_diff": bit_exact["max_abs_loss_diff"],
            }
        )

        per_rank = bundle.get("per_rank_step_ms")
        if isinstance(per_rank, dict):
            for path in ("baseline", "sdk"):
                detector_reports[path].append(
                    _detect_failslow_events(
                        per_rank,
                        path=path,
                        window_steps=detector_window_steps,
                        zscore_threshold=detector_zscore_threshold,
                        min_samples=detector_min_samples,
                    )
                )

    uplift_summary = aggregate_seeded_uplift(
        per_run_uplifts,
        n_resamples=aggregate_bootstrap_resamples,
        seed=seed_start + delay_ms * 17 + straggler_count,
    )

    return {
        "delay_ms": delay_ms,
        "straggler_count": straggler_count,
        "straggler_ranks": list(straggler_ranks),
        "n_seeds": n_seeds,
        "seed_start": seed_start,
        "mean_uplift_pct": uplift_summary.mean_uplift_pct,
        "sample_variance_pct2": uplift_summary.sample_variance_pct2,
        "ci95_low_pct": uplift_summary.ci_low_pct,
        "ci95_high_pct": uplift_summary.ci_high_pct,
        "dropped_low_uplift_pct": uplift_summary.dropped_low_pct,
        "dropped_high_uplift_pct": uplift_summary.dropped_high_pct,
        "surviving_uplifts_pct": list(uplift_summary.surviving_uplifts_pct),
        "bit_exact_pass": True,
        "max_abs_loss_diff": max_abs_loss_diff,
        "baseline_tokens_per_second_mean": _path_metric_mean(
            seed_bundles, "baseline", "tokens_per_second"
        ),
        "sdk_tokens_per_second_mean": _path_metric_mean(
            seed_bundles, "sdk", "tokens_per_second"
        ),
        "p50_baseline_ms": _path_metric_mean(seed_bundles, "baseline", "p50_step_time_ms"),
        "p95_baseline_ms": _path_metric_mean(seed_bundles, "baseline", "p95_step_time_ms"),
        "p99_baseline_ms": _path_metric_mean(seed_bundles, "baseline", "p99_step_time_ms"),
        "p50_sdk_ms": _path_metric_mean(seed_bundles, "sdk", "p50_step_time_ms"),
        "p95_sdk_ms": _path_metric_mean(seed_bundles, "sdk", "p95_step_time_ms"),
        "p99_sdk_ms": _path_metric_mean(seed_bundles, "sdk", "p99_step_time_ms"),
        "failslow_observe_only": {
            "baseline": _merge_detector_reports(detector_reports["baseline"]),
            "sdk": _merge_detector_reports(detector_reports["sdk"]),
        },
        "per_seed": per_seed,
    }


def build_sweep_bundle(
    *,
    output_cell: str,
    base_cfg: BenchConfig,
    delays_ms: Sequence[int],
    straggler_counts: Sequence[int],
    n_seeds: int,
    seed_start: int,
    aggregate_bootstrap_resamples: int,
    detector_window_steps: int,
    detector_zscore_threshold: float,
    detector_min_samples: int,
    quick: bool,
    run_cell_fn: RunCell = run_cell,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for straggler_count in straggler_counts:
        _peer_straggler_ranks(straggler_count, base_cfg.ranks)
    for delay_ms in delays_ms:
        if delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0; got {delay_ms}")
        for straggler_count in straggler_counts:
            rows.append(
                _run_grid_point(
                    base_cfg,
                    delay_ms=delay_ms,
                    straggler_count=straggler_count,
                    n_seeds=n_seeds,
                    seed_start=seed_start,
                    aggregate_bootstrap_resamples=aggregate_bootstrap_resamples,
                    detector_window_steps=detector_window_steps,
                    detector_zscore_threshold=detector_zscore_threshold,
                    detector_min_samples=detector_min_samples,
                    run_cell_fn=run_cell_fn,
                )
            )

    return {
        "schema_version": "1.0",
        "cell": output_cell,
        "kind": "stall_sweep",
        "reproducible": "cheap",
        "protocol": "docs/benchmark_protocol.md",
        "quick": quick,
        "hardware": {
            "label": base_cfg.hardware_label,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": base_cfg.backend,
            "ranks": base_cfg.ranks,
            "launch": base_cfg.launch,
        },
        "base_workload": {
            "cell": base_cfg.cell,
            "batch": base_cfg.batch,
            "in_dim": base_cfg.in_dim,
            "hidden": base_cfg.hidden,
            "out_dim": base_cfg.out_dim,
            "sequence_length": base_cfg.sequence_length,
            "lr": base_cfg.lr,
        },
        "sweep": {
            "delays_ms": list(delays_ms),
            "straggler_counts": list(straggler_counts),
            "rank_selection": "peer ranks 1..N; rank 0 remains the reporting observer",
            "n_seeds": n_seeds,
            "seed_start": seed_start,
            "drop_rule": "drop single lowest and single highest per-run uplift",
            "aggregate_bootstrap_resamples": aggregate_bootstrap_resamples,
            "detector_observe_only": {
                "enabled": True,
                "window_steps": detector_window_steps,
                "zscore_threshold": detector_zscore_threshold,
                "min_samples": detector_min_samples,
                "mitigation_called": False,
            },
            "config": {
                "steps": base_cfg.steps,
                "warmup_steps": base_cfg.warmup_steps,
                "sync_period_steps": base_cfg.sync_period_steps,
                "apply_lag_steps": base_cfg.apply_lag_steps,
                "simulated_merge_delay_ms": base_cfg.simulated_merge_delay_ms,
                "straggler_delay_ms": "grid",
                "straggler_ranks": "grid",
            },
        },
        "bit_exact_loss_equivalence": {
            "passed": all(row["bit_exact_pass"] for row in rows),
            "method": "every seeded paired run must pass elementwise IEEE-754 loss equality",
            "grid_points": len(rows),
            "max_abs_loss_diff": max(
                (float(row["max_abs_loss_diff"]) for row in rows),
                default=0.0,
            ),
        },
        "grid": rows,
    }


def write_sweep_bundle(bundle: dict[str, Any], results_root: Path = RESULTS_ROOT) -> Path:
    out_dir = results_root / str(bundle["cell"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n")
    return out_path


def _print_summary(bundle: dict[str, Any]) -> None:
    print("=" * 86)
    print(f"cell: {bundle['cell']}  ({'quick' if bundle['quick'] else 'full'} stall sweep)")
    print(
        "bit-exact loss equivalence: "
        f"{'PASS' if bundle['bit_exact_loss_equivalence']['passed'] else 'FAIL'} "
        f"(max |loss diff| = {bundle['bit_exact_loss_equivalence']['max_abs_loss_diff']:.3e})"
    )
    for row in bundle["grid"]:
        status = "PASS" if row["bit_exact_pass"] else "FAIL"
        print(
            f"delay={row['delay_ms']:>4} ms  stragglers={row['straggler_count']}  "
            f"bit_exact={status}  uplift={row['mean_uplift_pct']:+7.2f}% "
            f"CI95 [{row['ci95_low_pct']:+7.2f}%, {row['ci95_high_pct']:+7.2f}%]  "
            f"p95 baseline/sdk={row['p95_baseline_ms']:.2f}/{row['p95_sdk_ms']:.2f} ms"
        )
    print("=" * 86)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="cpu_gloo_2rank_mlp")
    parser.add_argument("--quick", action="store_true", help="run a small CPU smoke subset")
    parser.add_argument("--output-cell", default=None)
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--sync-period-steps", type=int, default=None)
    parser.add_argument("--apply-lag-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20240527)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--delays-ms", type=_parse_int_tuple, default=None)
    parser.add_argument("--straggler-counts", type=_parse_int_tuple, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--in-dim", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--out-dim", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--simulated-merge-delay-ms", type=int, default=0)
    parser.add_argument("--process-group-timeout-s", type=float, default=180.0)
    parser.add_argument("--step-bootstrap-resamples", type=int, default=None)
    parser.add_argument("--aggregate-bootstrap-resamples", type=int, default=None)
    parser.add_argument("--failslow-window-steps", type=int, default=50)
    parser.add_argument("--failslow-zscore-threshold", type=float, default=3.0)
    parser.add_argument("--failslow-min-samples", type=int, default=10)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--write",
        dest="write",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="write benchmarks/results/<output-cell>/result.json (default: --write)",
    )
    args = parser.parse_args(argv)
    if args.n_seeds is None:
        args.n_seeds = 5 if args.quick else 7
    if args.n_seeds < 5:
        raise SystemExit("--n-seeds must be >= 5")
    if args.step_bootstrap_resamples is None:
        args.step_bootstrap_resamples = 200 if args.quick else 10000
    if args.aggregate_bootstrap_resamples is None:
        args.aggregate_bootstrap_resamples = 500 if args.quick else 10000
    if args.delays_ms is None:
        args.delays_ms = QUICK_DELAYS_MS if args.quick else FULL_DELAYS_MS
    if args.straggler_counts is None:
        args.straggler_counts = (
            QUICK_STRAGGLER_COUNTS if args.quick else FULL_STRAGGLER_COUNTS
        )
    if args.output_cell is None:
        args.output_cell = (
            "cpu_gloo_stall_sweep_quick" if args.quick else "cpu_gloo_stall_sweep"
        )
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    base_cfg = _base_config(args)
    bundle = build_sweep_bundle(
        output_cell=str(args.output_cell),
        base_cfg=base_cfg,
        delays_ms=args.delays_ms,
        straggler_counts=args.straggler_counts,
        n_seeds=int(args.n_seeds),
        seed_start=int(args.seed),
        aggregate_bootstrap_resamples=int(args.aggregate_bootstrap_resamples),
        detector_window_steps=int(args.failslow_window_steps),
        detector_zscore_threshold=float(args.failslow_zscore_threshold),
        detector_min_samples=int(args.failslow_min_samples),
        quick=bool(args.quick),
    )
    _print_summary(bundle)
    if args.write:
        path = write_sweep_bundle(bundle, results_root=args.results_root)
        print(f"wrote bundle: {path.relative_to(Path.cwd())}")
    return 0 if bundle["bit_exact_loss_equivalence"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
