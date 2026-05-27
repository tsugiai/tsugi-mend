"""Measurement helpers for the paired-run benchmark protocol.

These functions implement the numeric core of ``docs/benchmark_protocol.md``:

- ``bit_exact_equal``      verify the load-bearing invariant that the SDK's
                           default (lossless) path reproduces the baseline's
                           per-step loss trajectory exactly.
- ``steady_state``         drop warmup steps and summarize the steady-state
                           portion of a per-step series (tokens/s, p50/p95/p99
                           step time).
- ``bootstrap_uplift_ci``  paired-resample bootstrap 95% CI for the tokens/s
                           uplift of the SDK path over the baseline path.

Everything here is pure Python + the standard library (``random``,
``statistics``, ``math``). No torch import, no network, no global state, so
the helpers are cheap to unit-test in isolation.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


def bit_exact_equal(
    baseline_losses: Sequence[float],
    sdk_losses: Sequence[float],
) -> bool:
    """Return True iff the two per-step loss trajectories are bit-exact.

    The default (lossless) SDK path must reproduce the synchronous
    baseline's loss step-for-step. We compare the IEEE-754 bit patterns
    rather than using a tolerance: ``abs(a - b) < eps`` would silently pass
    a path that introduced a tiny numerical drift, which is exactly the
    regression this check exists to catch. ``math.isnan`` is handled so two
    NaNs at the same step are treated as a mismatch (a NaN trajectory is
    never a valid PASS).
    """
    if len(baseline_losses) != len(sdk_losses):
        return False
    for a, b in zip(baseline_losses, sdk_losses):
        if math.isnan(a) or math.isnan(b):
            return False
        if math.copysign(1.0, a) != math.copysign(1.0, b):
            # Distinguish +0.0 from -0.0; also catches sign flips.
            return False
        if a != b:
            return False
    return True


@dataclass(frozen=True)
class StepSummary:
    """Steady-state summary of one path's per-step series."""

    n_steps_total: int
    warmup_steps: int
    n_steps_steady: int
    tokens_per_second: float
    mean_step_time_ms: float
    p50_step_time_ms: float
    p95_step_time_ms: float
    p99_step_time_ms: float


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence.

    ``q`` is a fraction in [0, 1]. Matches the "linear" / numpy default
    method so percentiles are stable and well-defined for small n.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = q * (len(sorted_values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[int(rank)])
    frac = rank - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def steady_state(
    step_times_ms: Sequence[float],
    tokens_per_step: int,
    warmup_steps: int,
) -> StepSummary:
    """Summarize the steady-state portion of a per-step timing series.

    Drops the first ``warmup_steps`` samples (protocol: exclude warmup),
    then reports tokens/s (observed at rank 0) plus p50/p95/p99 step time.
    Raises ValueError if fewer than one steady-state step remains, so a
    misconfigured short run fails loudly rather than emitting a divide-by-
    zero artifact.
    """
    if tokens_per_step <= 0:
        raise ValueError(f"tokens_per_step must be > 0; got {tokens_per_step}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0; got {warmup_steps}")
    n_total = len(step_times_ms)
    steady = list(step_times_ms[warmup_steps:])
    if len(steady) < 1:
        raise ValueError(
            f"need >= 1 steady-state step after dropping {warmup_steps} warmup "
            f"steps from {n_total} total; got {len(steady)}"
        )
    mean_ms = statistics.fmean(steady)
    ordered = sorted(steady)
    # tokens/s = tokens_per_step / mean_step_seconds.
    tps = tokens_per_step / (mean_ms / 1000.0)
    return StepSummary(
        n_steps_total=n_total,
        warmup_steps=warmup_steps,
        n_steps_steady=len(steady),
        tokens_per_second=tps,
        mean_step_time_ms=mean_ms,
        p50_step_time_ms=_percentile(ordered, 0.50),
        p95_step_time_ms=_percentile(ordered, 0.95),
        p99_step_time_ms=_percentile(ordered, 0.99),
    )


@dataclass(frozen=True)
class UpliftCI:
    """Bootstrap 95% CI for SDK-over-baseline tokens/s uplift (percent)."""

    point_estimate_pct: float
    ci_low_pct: float
    ci_high_pct: float
    n_paired_steps: int
    n_resamples: int
    confidence: float


def _uplift_pct_from_times(
    baseline_step_times_ms: Sequence[float],
    sdk_step_times_ms: Sequence[float],
) -> float:
    """Tokens/s uplift (percent) from two step-time series of equal length.

    Tokens-per-step cancels in the ratio, so the uplift only depends on the
    mean step times: uplift = (t_base / t_sdk) - 1. A faster SDK (smaller
    t_sdk) yields a positive uplift.
    """
    mean_base = statistics.fmean(baseline_step_times_ms)
    mean_sdk = statistics.fmean(sdk_step_times_ms)
    if mean_sdk <= 0:
        raise ValueError("SDK mean step time must be > 0")
    return (mean_base / mean_sdk - 1.0) * 100.0


def bootstrap_uplift_ci(
    baseline_step_times_ms: Sequence[float],
    sdk_step_times_ms: Sequence[float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> UpliftCI:
    """Paired bootstrap CI for the tokens/s uplift of SDK over baseline.

    The two series are paired step-for-step (protocol: alternate / interleave
    so cluster drift is shared). Each bootstrap resample draws the SAME step
    indices for both paths (paired resampling preserves the pairing), then
    recomputes the uplift. The reported CI is the percentile interval of the
    resampled uplifts. Defaults to 10000 resamples per the protocol.
    """
    if len(baseline_step_times_ms) != len(sdk_step_times_ms):
        raise ValueError(
            f"paired series must have equal length; got "
            f"{len(baseline_step_times_ms)} vs {len(sdk_step_times_ms)}"
        )
    n = len(baseline_step_times_ms)
    if n < 2:
        raise ValueError(f"need >= 2 paired steps for a bootstrap CI; got {n}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n_resamples}")

    point = _uplift_pct_from_times(baseline_step_times_ms, sdk_step_times_ms)
    rng = random.Random(seed)
    base = list(baseline_step_times_ms)
    sdk = list(sdk_step_times_ms)
    resampled: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        b_sample = [base[i] for i in idx]
        s_sample = [sdk[i] for i in idx]
        resampled.append(_uplift_pct_from_times(b_sample, s_sample))
    resampled.sort()
    alpha = 1.0 - confidence
    low = _percentile(resampled, alpha / 2.0)
    high = _percentile(resampled, 1.0 - alpha / 2.0)
    return UpliftCI(
        point_estimate_pct=point,
        ci_low_pct=low,
        ci_high_pct=high,
        n_paired_steps=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )


__all__ = [
    "bit_exact_equal",
    "steady_state",
    "StepSummary",
    "bootstrap_uplift_ci",
    "UpliftCI",
]
