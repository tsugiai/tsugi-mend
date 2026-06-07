"""Online runtime autotuning for fail-slow detection and grace-window wait.

The autotuner is deliberately scheduling-only. It adapts detector sensitivity
and the wall-clock grace-window wait from observed step-time statistics, but it
does not change merge cadence, merge math, quorum membership, or rank
exclusion.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class RuntimeAutotuneDecision:
    """One online runtime-autotuner decision."""

    rank_id: str
    step_time_ms: float
    window_size: int
    window_mean_ms: float
    window_std_ms: float
    jitter_ratio: float
    aggregate_jitter_ratio: float
    detector_z_score: float
    detector_flagged_slow: bool
    effective_failslow_zscore_threshold: float
    effective_grace_window_ms: int
    threshold_action: str
    grace_action: str
    reason: str


class RuntimeAutotuner:
    """Continuously adapt observe-only runtime controls.

    The control law is intentionally simple and deterministic:

    * Detector threshold tracks rolling coefficient-of-variation jitter. A
      clean cluster stays near the configured base threshold; benign jitter
      raises the threshold within a bounded range to avoid false positives.
    * Grace-window wait widens when the detector flags a slow step, using the
      observed lag magnitude. It decays back toward the configured base grace
      window when the cluster returns to clean observations.

    Both outputs affect only diagnostics and wall-clock waiting. They do not
    choose a different merge boundary and never exclude a rank.
    """

    def __init__(
        self,
        *,
        base_failslow_zscore_threshold: float,
        failslow_zscore_min: float,
        failslow_zscore_max: float,
        base_grace_window_ms: int,
        grace_window_max_ms: int,
        window_steps: int,
        min_samples: int,
    ) -> None:
        if base_failslow_zscore_threshold <= 0:
            raise ValueError("base_failslow_zscore_threshold must be > 0")
        if failslow_zscore_min <= 0:
            raise ValueError("failslow_zscore_min must be > 0")
        if failslow_zscore_max < failslow_zscore_min:
            raise ValueError("failslow_zscore_max must be >= failslow_zscore_min")
        if base_grace_window_ms < 0:
            raise ValueError("base_grace_window_ms must be >= 0")
        if grace_window_max_ms < 0:
            raise ValueError("grace_window_max_ms must be >= 0")
        if window_steps < 2:
            raise ValueError("window_steps must be >= 2")
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        if min_samples > window_steps:
            raise ValueError("min_samples cannot exceed window_steps")

        self.base_failslow_zscore_threshold = base_failslow_zscore_threshold
        self.failslow_zscore_floor = min(
            failslow_zscore_min,
            base_failslow_zscore_threshold,
        )
        self.failslow_zscore_ceiling = max(
            failslow_zscore_max,
            base_failslow_zscore_threshold,
        )
        self.base_grace_window_ms = base_grace_window_ms
        self.grace_window_ceiling_ms = max(grace_window_max_ms, base_grace_window_ms)
        self.window_steps = window_steps
        self.min_samples = min_samples

        self._windows: dict[str, Deque[float]] = {}
        self._rank_jitter_ratios: dict[str, float] = {}
        self._effective_failslow_zscore_threshold = base_failslow_zscore_threshold
        self._effective_grace_window_ms = base_grace_window_ms

    @property
    def effective_failslow_zscore_threshold(self) -> float:
        """Current detector z-score threshold chosen by the autotuner."""

        return self._effective_failslow_zscore_threshold

    @property
    def effective_grace_window_ms(self) -> int:
        """Current wall-clock grace-window wait chosen by the autotuner."""

        return self._effective_grace_window_ms

    def observe(
        self,
        *,
        rank_id: str,
        step_time_ms: float,
        detector_z_score: float,
        detector_flagged_slow: bool,
    ) -> RuntimeAutotuneDecision:
        """Ingest one rank's step time and update effective controls."""

        if step_time_ms < 0:
            raise ValueError(f"step_time_ms must be >= 0; got {step_time_ms}")
        window = self._windows.setdefault(rank_id, deque(maxlen=self.window_steps))
        window.append(float(step_time_ms))

        window_size = len(window)
        mean = sum(window) / window_size
        variance = sum((sample - mean) ** 2 for sample in window) / window_size
        std = math.sqrt(variance)
        jitter_ratio = (std / mean) if mean > 0.0 else 0.0

        if window_size >= self.min_samples:
            self._rank_jitter_ratios[rank_id] = jitter_ratio
        aggregate_jitter = max(self._rank_jitter_ratios.values(), default=0.0)

        old_threshold = self._effective_failslow_zscore_threshold
        old_grace = self._effective_grace_window_ms

        if window_size < self.min_samples:
            threshold_action = "hold"
            grace_action = "hold"
            reason = "warmup"
        else:
            self._effective_failslow_zscore_threshold = self._threshold_from_jitter(
                aggregate_jitter
            )
            threshold_action = self._action(
                old_threshold,
                self._effective_failslow_zscore_threshold,
                tolerance=1e-9,
            )

            if detector_flagged_slow:
                lag_ms = max(0.0, step_time_ms - mean)
                proposed_grace = self.base_grace_window_ms + int(math.ceil(lag_ms))
                self._effective_grace_window_ms = min(
                    self.grace_window_ceiling_ms,
                    max(self._effective_grace_window_ms, proposed_grace),
                )
                reason = "slow_observation"
            else:
                self._effective_grace_window_ms = self._decay_grace_window()
                reason = "clean_observation"
            grace_action = self._action(old_grace, self._effective_grace_window_ms)

        return RuntimeAutotuneDecision(
            rank_id=rank_id,
            step_time_ms=step_time_ms,
            window_size=window_size,
            window_mean_ms=mean,
            window_std_ms=std,
            jitter_ratio=jitter_ratio,
            aggregate_jitter_ratio=aggregate_jitter,
            detector_z_score=detector_z_score,
            detector_flagged_slow=detector_flagged_slow,
            effective_failslow_zscore_threshold=self._effective_failslow_zscore_threshold,
            effective_grace_window_ms=self._effective_grace_window_ms,
            threshold_action=threshold_action,
            grace_action=grace_action,
            reason=reason,
        )

    def _threshold_from_jitter(self, aggregate_jitter: float) -> float:
        # A 25% coefficient of variation maps to roughly +2 z-score points.
        # This keeps the detector quieter under benign jitter while preserving
        # sensitivity to large spikes in clean periods.
        target = self.base_failslow_zscore_threshold + (aggregate_jitter * 8.0)
        return min(
            self.failslow_zscore_ceiling,
            max(self.failslow_zscore_floor, target),
        )

    def _decay_grace_window(self) -> int:
        if self._effective_grace_window_ms <= self.base_grace_window_ms:
            return self.base_grace_window_ms
        decay_step = max(1, int(math.ceil(max(self.base_grace_window_ms, 1) * 0.1)))
        return max(
            self.base_grace_window_ms,
            self._effective_grace_window_ms - decay_step,
        )

    @staticmethod
    def _action(old: float | int, new: float | int, *, tolerance: float = 0.0) -> str:
        if new > old + tolerance:
            return "increase"
        if new < old - tolerance:
            return "decrease"
        return "hold"
