"""FALCON-style fail-slow detection.

Reference: arXiv:2410.12588 (October 2024). FALCON reports that fail-slows
on a >10,000 GPU shared cluster cause 1.34% average job-completion-time
delay, and FALCON's mitigation removes 60.1% of the slowdown.

This module implements just the detection idea (sliding-window z-score on
per-rank step times). The mitigation logic lives in the runtime: when a
rank crosses the z-score threshold, the runtime calls
`GraceWindowSyncer.mark_failslow(rank_id)` to exclude that rank from the
current outer round's quorum.

We do NOT implement FALCON's CockroachDB-style timeout machinery; that is
overkill for the SDK's scope. The detection layer is the load-bearing
piece.

Patent-independence note: FALCON's fail-slow detection logic is published
prior art (October 2024) and is unrelated to TsugiCinema's patent estates.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class FailSlowDecision:
    """One detector decision."""
    rank_id: str
    is_slow: bool
    z_score: float
    window_mean_ms: float
    window_std_ms: float
    window_size: int
    reason: str  # "ok", "slow", "warmup", "degenerate_window"


class FailSlowDetector:
    """Sliding-window z-score detector for per-rank step time.

    The detector tracks one rolling deque per rank-id. After each step
    completes for a rank, the caller emits the rank's wall-clock step
    duration in milliseconds; the detector returns a `FailSlowDecision`
    indicating whether the rank is currently stragglering.

    Decision rule:
        is_slow = (window_size >= min_samples) AND
                  (z_score > zscore_threshold)
        z_score = (current_ms - window_mean_ms) / window_std_ms

    The current_ms sample is INCLUDED in the window mean/std for the
    decision so a single very large step does not get smoothed away by
    later samples. If the standard deviation is zero (perfectly constant
    history), z_score is +inf if current_ms > mean else 0.
    """

    def __init__(
        self,
        window_steps: int,
        zscore_threshold: float,
        min_samples: int,
    ) -> None:
        if window_steps < 2:
            raise ValueError(f"window_steps must be >= 2; got {window_steps}")
        if zscore_threshold <= 0:
            raise ValueError(f"zscore_threshold must be > 0; got {zscore_threshold}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2; got {min_samples}")
        if min_samples > window_steps:
            raise ValueError(
                f"min_samples ({min_samples}) cannot exceed window_steps ({window_steps})"
            )
        self.window_steps = window_steps
        self.zscore_threshold = zscore_threshold
        self.min_samples = min_samples
        self._windows: dict[str, Deque[float]] = {}

    def observe(self, rank_id: str, step_time_ms: float) -> FailSlowDecision:
        if step_time_ms < 0:
            raise ValueError(f"step_time_ms must be >= 0; got {step_time_ms}")
        window = self._windows.setdefault(
            rank_id, deque(maxlen=self.window_steps)
        )
        window.append(float(step_time_ms))
        n = len(window)
        if n < self.min_samples:
            return FailSlowDecision(
                rank_id=rank_id,
                is_slow=False,
                z_score=0.0,
                window_mean_ms=sum(window) / n,
                window_std_ms=0.0,
                window_size=n,
                reason="warmup",
            )
        mean = sum(window) / n
        var = sum((x - mean) ** 2 for x in window) / n
        std = math.sqrt(var)
        if std == 0.0:
            if step_time_ms > mean:
                z = float("inf")
            else:
                z = 0.0
            return FailSlowDecision(
                rank_id=rank_id,
                is_slow=z > self.zscore_threshold,
                z_score=z,
                window_mean_ms=mean,
                window_std_ms=0.0,
                window_size=n,
                reason="degenerate_window" if not (z > self.zscore_threshold) else "slow",
            )
        z = (step_time_ms - mean) / std
        is_slow = z > self.zscore_threshold
        return FailSlowDecision(
            rank_id=rank_id,
            is_slow=is_slow,
            z_score=z,
            window_mean_ms=mean,
            window_std_ms=std,
            window_size=n,
            reason="slow" if is_slow else "ok",
        )

    def reset(self, rank_id: str) -> None:
        """Drop the window for a rank (e.g., after operator-confirmed
        node-replacement)."""
        self._windows.pop(rank_id, None)

    def clear(self) -> None:
        """Drop all windows."""
        self._windows.clear()
