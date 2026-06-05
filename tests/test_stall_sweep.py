"""Unit tests for the stall-sweep orchestration layer."""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import pytest

_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.run_paired import BenchConfig, _build_result_bundle  # noqa: E402
from benchmarks.run_stall_sweep import (  # noqa: E402
    _peer_straggler_ranks,
    build_sweep_bundle,
)


def _fake_per_rank_step_ms(cfg: BenchConfig) -> dict[str, dict[str, list[float]]]:
    per_rank: dict[str, dict[str, list[float]]] = {}
    for rank in range(cfg.ranks):
        baseline = [10.0] * cfg.steps
        sdk = [8.0] * cfg.steps
        for step in range(cfg.sync_period_steps, cfg.steps, cfg.sync_period_steps):
            if cfg.straggler_delay_ms > 0 and rank in cfg.straggler_ranks:
                baseline[step] += float(cfg.straggler_delay_ms)
        per_rank[str(rank)] = {"baseline": baseline, "sdk": sdk}
    return per_rank


def _fake_run_cell(cfg: BenchConfig) -> Optional[dict[str, Any]]:
    assert cfg._include_rank_timings is True
    jitter = float(cfg.seed % 5)
    baseline_ms = [10.0 + jitter] * cfg.steps
    sdk_ms = [8.0 + jitter] * cfg.steps
    for step in range(cfg.sync_period_steps, cfg.steps, cfg.sync_period_steps):
        if cfg.straggler_delay_ms > 0 and cfg.straggler_ranks:
            baseline_ms[step] += float(cfg.straggler_delay_ms)
            sdk_ms[step] += float(cfg.straggler_delay_ms) * 0.25
    losses = [1.0 - 0.001 * step for step in range(cfg.steps)]
    return _build_result_bundle(
        cfg,
        base_losses=losses,
        base_ms=baseline_ms,
        sdk_losses=list(losses),
        sdk_ms=sdk_ms,
        per_rank_step_ms=_fake_per_rank_step_ms(cfg),
    )


def test_peer_straggler_ranks_keep_rank_zero_as_observer():
    assert _peer_straggler_ranks(0, ranks=2) == ()
    assert _peer_straggler_ranks(1, ranks=2) == (1,)
    assert _peer_straggler_ranks(4, ranks=5) == (1, 2, 3, 4)


def test_peer_straggler_ranks_require_a_reporting_peer():
    with pytest.raises(ValueError, match="increase --ranks"):
        _peer_straggler_ranks(2, ranks=2)


def test_build_sweep_bundle_aggregates_every_grid_point():
    base = BenchConfig(
        cell="fake",
        ranks=2,
        steps=14,
        warmup_steps=2,
        sync_period_steps=10,
        apply_lag_steps=2,
        batch=4,
        in_dim=8,
        hidden=16,
        out_dim=4,
        bootstrap_resamples=10,
        _include_rank_timings=True,
    )
    bundle = build_sweep_bundle(
        output_cell="fake_stall_sweep",
        base_cfg=base,
        delays_ms=(0, 50),
        straggler_counts=(0, 1),
        n_seeds=5,
        seed_start=100,
        aggregate_bootstrap_resamples=20,
        detector_window_steps=12,
        detector_zscore_threshold=3.0,
        detector_min_samples=10,
        quick=True,
        run_cell_fn=_fake_run_cell,
    )
    assert bundle["cell"] == "fake_stall_sweep"
    assert bundle["bit_exact_loss_equivalence"]["passed"] is True
    assert len(bundle["grid"]) == 4
    delayed_peer_row = next(
        row
        for row in bundle["grid"]
        if row["delay_ms"] == 50 and row["straggler_count"] == 1
    )
    assert delayed_peer_row["straggler_ranks"] == [1]
    assert delayed_peer_row["n_seeds"] == 5
    assert len(delayed_peer_row["surviving_uplifts_pct"]) == 3
    assert delayed_peer_row["bit_exact_pass"] is True
    assert delayed_peer_row["max_abs_loss_diff"] == 0.0
    assert bundle["sweep"]["detector_observe_only"]["mitigation_called"] is False


def test_build_sweep_bundle_fails_invalid_bit_exact_run():
    def bad_run_cell(cfg: BenchConfig) -> Optional[dict[str, Any]]:
        bundle = _fake_run_cell(cfg)
        assert bundle is not None
        bundle["bit_exact_loss_equivalence"]["passed"] = False
        bundle["bit_exact_loss_equivalence"]["max_abs_loss_diff"] = 1.0
        return bundle

    base = BenchConfig(
        ranks=2,
        steps=12,
        warmup_steps=2,
        sync_period_steps=10,
        apply_lag_steps=1,
        bootstrap_resamples=10,
        _include_rank_timings=True,
    )
    with pytest.raises(RuntimeError, match="bit-exact loss equivalence failed"):
        build_sweep_bundle(
            output_cell="bad",
            base_cfg=base,
            delays_ms=(50,),
            straggler_counts=(1,),
            n_seeds=5,
            seed_start=0,
            aggregate_bootstrap_resamples=10,
            detector_window_steps=12,
            detector_zscore_threshold=3.0,
            detector_min_samples=10,
            quick=True,
            run_cell_fn=bad_run_cell,
        )
