"""Online runtime autotuner: fail-slow detection sensitivity + grace window.

This module implements a small, deterministic control law that adapts two
operational knobs from the runtime's own observed per-rank step-time stream,
continuously (every step past a warmup window), not one-shot:

1. **Fail-slow detection sensitivity** -- the effective z-score threshold the
   `failslow.FailSlowDetector` uses. The control law tracks the live
   coefficient of variation (CoV) of recent step times and moves the
   threshold so the detector follows the cluster's actual jitter: a jittery
   cluster gets a higher threshold (fewer false straggler flags under benign
   jitter), a clean cluster gets a lower threshold toward a floor (a real
   fail-slow is still caught). This is the lightweight online
   performance-monitoring idea from Guard (arXiv:2605.17879).

2. **Grace-window wait** -- the effective wall-clock `grace_window_ms` the
   syncer waits after quorum. The control law widens the wait when a
   sustained straggler is observed (recent peak/median step-time ratio high)
   so the syncer can absorb a real laggard, and narrows it back toward the
   static baseline when the cluster is clean. This is the operational
   recovery-wait heuristic from "From Detection to Recovery"
   (arXiv:2605.09370).

Numerical-safety contract (why this is bit-exact-safe by construction):

- The detection threshold is OBSERVE-ONLY. The detector's decision is a
  diagnostic flag; no mitigation/exclusion is wired off it here. Changing the
  threshold changes only which steps are FLAGGED, never any tensor value.
- The grace window is a WALL-CLOCK WAIT. In default (lossless) mode the
  syncer waits for the same fragments regardless of the wait length, and the
  merged delta is the token-weighted merge of the same fragment set applied
  at the same logical boundary. Changing the wait changes timing/overlap,
  never WHICH fragments merge or the apply boundary.
- This module deliberately does NOT touch the merge cadence
  (sync_period_steps / momentum cadence / apply lag). Adapting the merge
  cadence from measured step times would make a paired baseline vs sdk run
  choose different cadences (their step times differ because the sdk overlaps
  the merge) and would break bit-exact loss equivalence. The cadence is out
  of scope here on purpose.

The control law is pure Python (stdlib only), holds a bounded deque of recent
samples, and is fully deterministic for a fixed input stream so a run's
adaptation is reproducible. It performs no I/O.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class AutotuneDecision:
    """One online autotuner decision, emitted after a step is observed.

    Attributes:
        adapted: True if the autotuner produced an updated effective value
            this step (i.e. it was past warmup with enough samples).
        effective_zscore_threshold: the z-score threshold the detector
            should use going forward (observe-only; numerically inert).
        effective_grace_window_ms: the wall-clock grace window the syncer
            should wait going forward.
        observed_cov: coefficient of variation of the recent step-time
            window (std / mean); 0.0 during warmup or for a flat window.
        observed_peak_ratio: recent max / recent median step time; 1.0 when
            flat or during warmup. The straggler-magnitude signal that
            drives the grace-window widening.
        window_size: number of samples currently in the rolling window.
        reason: "warmup" while the window is below min_samples; otherwise
            "adapted".
    """

    adapted: bool
    effective_zscore_threshold: float
    effective_grace_window_ms: int
    observed_cov: float
    observed_peak_ratio: float
    window_size: int
    reason: str


class RuntimeAutotuner:
    """Deterministic online control law for detection sensitivity + grace window.

    The runtime feeds each rank's per-step wall-clock duration (ms) into
    :meth:`observe`, which returns an :class:`AutotuneDecision`. The runtime
    then applies the effective values to the (observe-only) fail-slow
    detector and to the syncer's wall-clock grace window.

    Control law (all bounded, monotone-clamped, deterministic):

    - Detection threshold.  Let ``cov`` be the coefficient of variation of
      the rolling window.  The effective threshold is::

          z_eff = clamp(base_z + cov_gain * cov,
                        zscore_min, zscore_max)

      A higher live CoV (jittery cluster) raises ``z_eff`` so benign jitter
      does not trip a straggler flag; a near-flat window pulls ``z_eff`` down
      toward ``base_z`` (and never below ``zscore_min``), keeping the
      detector sharp enough to catch a real fail-slow.

    - Grace window.  Let ``peak_ratio`` be recent max / recent median.  The
      effective grace window is::

          g_eff = clamp(round(base_grace_ms * (1 + grace_gain * max(0, peak_ratio - 1))),
                        grace_min_ms, grace_max_ms)

      A sustained straggler (peak_ratio well above 1) widens the wall-clock
      wait so the syncer can absorb the laggard; a clean window (peak_ratio
      near 1) returns the wait toward the static baseline.

    Both laws are pure functions of the current window contents, so the
    sequence of decisions is fully reproducible for a fixed input stream.
    """

    def __init__(
        self,
        *,
        base_zscore_threshold: float,
        base_grace_window_ms: int,
        window_steps: int,
        min_samples: int,
        zscore_min: float,
        zscore_max: float,
        grace_min_ms: int,
        grace_max_ms: int,
        cov_gain: float,
        grace_gain: float,
    ) -> None:
        if window_steps < 2:
            raise ValueError(f"window_steps must be >= 2; got {window_steps}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2; got {min_samples}")
        if min_samples > window_steps:
            raise ValueError(
                f"min_samples ({min_samples}) cannot exceed window_steps ({window_steps})"
            )
        if zscore_min <= 0:
            raise ValueError(f"zscore_min must be > 0; got {zscore_min}")
        if zscore_max < zscore_min:
            raise ValueError(
                f"zscore_max ({zscore_max}) must be >= zscore_min ({zscore_min})"
            )
        if grace_min_ms < 0:
            raise ValueError(f"grace_min_ms must be >= 0; got {grace_min_ms}")
        if grace_max_ms < grace_min_ms:
            raise ValueError(
                f"grace_max_ms ({grace_max_ms}) must be >= grace_min_ms ({grace_min_ms})"
            )
        if cov_gain < 0:
            raise ValueError(f"cov_gain must be >= 0; got {cov_gain}")
        if grace_gain < 0:
            raise ValueError(f"grace_gain must be >= 0; got {grace_gain}")

        self.base_zscore_threshold = float(base_zscore_threshold)
        self.base_grace_window_ms = int(base_grace_window_ms)
        self.window_steps = window_steps
        self.min_samples = min_samples
        self.zscore_min = float(zscore_min)
        self.zscore_max = float(zscore_max)
        self.grace_min_ms = int(grace_min_ms)
        self.grace_max_ms = int(grace_max_ms)
        self.cov_gain = float(cov_gain)
        self.grace_gain = float(grace_gain)

        self._window: Deque[float] = deque(maxlen=window_steps)
        # Effective values seed at the static baseline so that, before the
        # window fills, the runtime behaves exactly as the un-tuned config.
        self._effective_zscore_threshold = self.base_zscore_threshold
        self._effective_grace_window_ms = self.base_grace_window_ms

    @property
    def effective_zscore_threshold(self) -> float:
        return self._effective_zscore_threshold

    @property
    def effective_grace_window_ms(self) -> int:
        return self._effective_grace_window_ms

    def observe(self, step_time_ms: float) -> AutotuneDecision:
        """Feed one per-step wall-clock duration (ms); return the decision.

        During warmup (window below ``min_samples``) the effective values are
        held at the static baseline and ``adapted`` is False.
        """
        if step_time_ms < 0:
            raise ValueError(f"step_time_ms must be >= 0; got {step_time_ms}")
        self._window.append(float(step_time_ms))
        n = len(self._window)
        if n < self.min_samples:
            return AutotuneDecision(
                adapted=False,
                effective_zscore_threshold=self._effective_zscore_threshold,
                effective_grace_window_ms=self._effective_grace_window_ms,
                observed_cov=0.0,
                observed_peak_ratio=1.0,
                window_size=n,
                reason="warmup",
            )

        cov = self._coefficient_of_variation()
        peak_ratio = self._peak_ratio()

        z_eff = self.base_zscore_threshold + self.cov_gain * cov
        z_eff = _clamp(z_eff, self.zscore_min, self.zscore_max)

        widen = 1.0 + self.grace_gain * max(0.0, peak_ratio - 1.0)
        g_raw = int(round(self.base_grace_window_ms * widen))
        g_eff = int(_clamp(float(g_raw), float(self.grace_min_ms), float(self.grace_max_ms)))

        self._effective_zscore_threshold = z_eff
        self._effective_grace_window_ms = g_eff
        return AutotuneDecision(
            adapted=True,
            effective_zscore_threshold=z_eff,
            effective_grace_window_ms=g_eff,
            observed_cov=cov,
            observed_peak_ratio=peak_ratio,
            window_size=n,
            reason="adapted",
        )

    def _coefficient_of_variation(self) -> float:
        # CoV of the BASELINE jitter, computed over the window with the
        # single largest sample removed. Excluding the top sample keeps a
        # lone transient straggler from inflating the steady-state jitter
        # estimate and thereby raising the detection threshold high enough
        # to mask the very spike it should catch. (Guard's online monitoring
        # tracks the cluster's benign jitter, not its outliers.) With
        # n >= min_samples >= 2 this leaves at least one sample.
        ordered = sorted(self._window)
        body = ordered[:-1] if len(ordered) > 1 else ordered
        m = len(body)
        mean = sum(body) / m
        if mean <= 0.0:
            return 0.0
        var = sum((x - mean) ** 2 for x in body) / m
        std = math.sqrt(var)
        return std / mean

    def _peak_ratio(self) -> float:
        ordered = sorted(self._window)
        n = len(ordered)
        # Median that ignores the largest sample so a single sustained
        # straggler does not inflate the "typical" baseline it is compared
        # against. With n >= min_samples >= 2 this leaves at least one
        # sample to take the median over.
        body = ordered[:-1] if n > 1 else ordered
        m = len(body)
        if m % 2 == 1:
            median = body[m // 2]
        else:
            median = (body[m // 2 - 1] + body[m // 2]) / 2.0
        peak = ordered[-1]
        if median <= 0.0:
            return 1.0
        return peak / median

    def reset(self) -> None:
        """Drop the rolling window and re-seed the effective values at the
        static baseline. Useful after an operator-confirmed node swap."""
        self._window.clear()
        self._effective_zscore_threshold = self.base_zscore_threshold
        self._effective_grace_window_ms = self.base_grace_window_ms


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = ["AutotuneDecision", "RuntimeAutotuner"]
