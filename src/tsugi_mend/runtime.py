"""Top-level runtime: mend_init / mend_shutdown.

Wires the SDK components (reducer, desync_optimizer, async_tp, failslow,
topology, sideband, diagnostics) into the user training script.

Public API:
    mend_init(model, config, rank_id=None)
        Initialize the runtime. Call once after model wrap (FSDP, TP, etc.).
    mend_shutdown(model)
        Tear down the runtime. Call once before dist.destroy_process_group.

This is the integration layer. The unit tests exercise the individual
modules deterministically; the runtime adds the wiring that calls those
modules at the right points in a training loop.

Patent-independence note: this is the public-art integration of
Decoupled DiLoCo + DES-LOC + async-TP + FALCON. It does not exercise
TsugiCinema's K-Pool LoRA or Infinity patent estates. There is no
variance-threshold trigger, no K-of-N adapter routing, no LoRA-adapter-
granularity reduction.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import threading
import time
from typing import Optional

import torch.nn as nn

from tsugi_mend.async_tp import enable_async_tp
from tsugi_mend.concurrent import ConcurrentOuterStep, FragmentProvider
from tsugi_mend.config import MendConfig
from tsugi_mend.desync_optimizer import DesynchronizedSyncSchedule
from tsugi_mend.diagnostics import DiagnosticsWriter
from tsugi_mend.failslow import FailSlowDetector
from tsugi_mend.reducer import GraceWindowSyncer, MergeResult
from tsugi_mend.sideband import Sideband
from tsugi_mend.topology import Topology, detect, local_hostname

_LOG = logging.getLogger(__name__)


class _MaxRuntime:
    """Per-model runtime container."""

    def __init__(self, config: MendConfig, rank_id: str) -> None:
        self.config = config
        self.rank_id = rank_id
        self.hostname = local_hostname()

        self.sched = DesynchronizedSyncSchedule(
            sync_period_steps=config.sync_period_steps,
            momentum_sync_period_steps=config.momentum_sync_period_steps,
        )
        self.failslow = FailSlowDetector(
            window_steps=config.failslow_window_steps,
            zscore_threshold=config.failslow_zscore_threshold,
            min_samples=config.failslow_min_samples,
        )
        self.syncer = GraceWindowSyncer(
            quorum_min_learners=config.quorum_min_learners,
            grace_window_ms=config.grace_window_ms,
            token_weighted=config.token_weighted_merge,
            simulated_merge_delay_ms=config.simulated_merge_delay_ms,
            simulated_merge_delay_distribution=config.simulated_merge_delay_distribution,
        )
        self.sideband: Optional[Sideband] = None
        if config.sideband_peers:
            self.sideband = Sideband(
                rank_id=rank_id,
                hostname=self.hostname,
                addr=config.sideband_addr,
                peers=config.sideband_peers,
                heartbeat_ms=config.sideband_heartbeat_ms,
                connect_timeout_s=config.sideband_connect_timeout_s,
            )
        self.diagnostics = DiagnosticsWriter(config.diagnostics_dir)
        self.topology: Optional[Topology] = None
        self.async_tp_active = False

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._step_start_s: Optional[float] = None
        # Phase 2 Week 1: concurrent outer-step orchestrator. Allocated
        # in start() if config.concurrent_outer_step is True. Owns the
        # GraceWindowSyncer instance once allocated; callers should not
        # touch self.syncer directly when the orchestrator is present.
        self._concurrent_orch: Optional[ConcurrentOuterStep] = None
        # Auto-tuner for sync_period_steps. The
        # runtime collects per-step wall-clock times during the warmup
        # window and at step `auto_tune_sync_period_warmup_steps` sets
        # the effective N to ceil(grace_window_ms / T_step), clamped to
        # [auto_tune_sync_period_min, sync_period_steps]. Decision is
        # one-shot at warmup boundary; the orchestrator's behavior past
        # that point uses the new N.
        self._effective_sync_period_steps: int = config.sync_period_steps
        self._auto_tune_decided: bool = not config.auto_tune_sync_period
        self._warmup_step_times_ms: list[float] = []

    def start(self, model: nn.Module) -> None:
        if self.config.async_tp_enabled:
            self.async_tp_active = enable_async_tp(model)
        # Spawn the asyncio event loop on a background thread if EITHER
        # the sideband or the concurrent outer-step orchestrator needs
        # one. The loop is shared between them.
        needs_loop = (
            self.sideband is not None
            or self.config.concurrent_outer_step
        )
        if needs_loop:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._run_loop, args=(self._loop,), daemon=True
            )
            self._loop_thread.start()
        if self.sideband is not None:
            assert self._loop is not None
            future = asyncio.run_coroutine_threadsafe(
                self.sideband.start(), self._loop
            )
            future.result(timeout=5.0)
        # Allocate the concurrent outer-step orchestrator if enabled.
        # It wraps self.syncer; from this point on, direct calls into
        # self.syncer for outer-round orchestration should go via the
        # orchestrator's submit_async / collect API.
        if self.config.concurrent_outer_step:
            assert self._loop is not None
            self._concurrent_orch = ConcurrentOuterStep(
                syncer=self.syncer,
                loop=self._loop,
            )
        # Initial topology classification. The hostname map may be sparse
        # at startup; the runtime can call refresh_topology() later once
        # peers have heartbeated.
        self.topology = detect(
            rank_count=1,  # Stage A: single-process default. Multi-rank wiring
                           # is the Stage B+ integration concern, not Stage A.
            nccl_topo_file=self.config.nccl_topo_file,
            rank_to_hostname={0: self.hostname},
        )
        self.diagnostics.emit(
            "mend_init",
            rank_id=self.rank_id,
            hostname=self.hostname,
            async_tp_active=self.async_tp_active,
            concurrent_outer_step_active=self._concurrent_orch is not None,
            topology_method=self.topology.detection_method,
            n_racks=self.topology.n_racks(),
        )

    def stop(self) -> None:
        # Drain in-flight orchestrator round (best-effort). The orchestrator's
        # asyncio task has its own deadline bound; we don't block on it.
        if self._concurrent_orch is not None and self._concurrent_orch.is_pending():
            _LOG.warning(
                "max_runtime.stop: concurrent outer-step still PENDING at shutdown; "
                "the in-flight merge will be abandoned"
            )
        if self.sideband is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.sideband.stop(), self._loop
                ).result(timeout=5.0)
            except Exception as e:  # pylint: disable=broad-except
                _LOG.warning("max_runtime.stop: sideband shutdown error: %s", e)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
        self.diagnostics.emit("mend_shutdown", rank_id=self.rank_id)
        self.diagnostics.close()

    # -------- training-loop hooks (Stage B+ wiring) --------

    def step_begin(self, step: int) -> None:
        """Hook the runtime expects at the top of each optimizer step.
        Updates the sideband progress state and records step-start time
        for fail-slow detection."""
        self._step_start_s = time.monotonic()
        if self.sideband is not None:
            self.sideband.set_local_state(
                step_id=step,
                queue_depth=0,
                health_bit=True,
            )

    def step_end(self, step: int) -> None:
        """Hook the runtime expects at the bottom of each optimizer step.
        Observes the step time, runs fail-slow detection, emits
        diagnostics."""
        if self._step_start_s is None:
            return
        step_time_ms = (time.monotonic() - self._step_start_s) * 1000.0
        self._step_start_s = None
        decision = self.failslow.observe(self.rank_id, step_time_ms)
        if decision.is_slow:
            self.diagnostics.emit(
                "failslow_decision",
                step=step,
                rank_id=decision.rank_id,
                z_score=decision.z_score,
                window_mean_ms=decision.window_mean_ms,
                window_std_ms=decision.window_std_ms,
                reason=decision.reason,
            )
        # Auto-tuner warmup-window measurement. Active only
        # when config.auto_tune_sync_period is True AND a decision has
        # not yet been made. Collects step_time_ms samples; at the
        # warmup boundary, computes the median of the LAST 25 samples
        # (or half the warmup window, whichever is smaller, to dodge
        # cudagraph capture and HF dataset I/O artifacts on the
        # earliest steps), derives N* = ceil(G / T_step), and updates
        # the DES-LOC sync schedule in place.
        if not self._auto_tune_decided:
            self._warmup_step_times_ms.append(step_time_ms)
            if len(self._warmup_step_times_ms) >= self.config.auto_tune_sync_period_warmup_steps:
                self._decide_auto_tune(step)

    def schedule_for(self, step: int):
        """Convenience pass-through to the DES-LOC schedule."""
        return self.sched.tick(step)

    def effective_sync_period_steps(self) -> int:
        """Current N used by the DES-LOC schedule. Equals
        config.sync_period_steps when the auto-tuner is disabled OR has
        not yet fired; equals the auto-tuned N* once the warmup
        boundary has been crossed."""
        return self._effective_sync_period_steps

    def _decide_auto_tune(self, step: int) -> None:
        """Compute N* from the warmup window's median step time and
        update the DES-LOC schedule. Called from step_end once enough
        warmup samples accumulate."""
        # Median of the LAST 25 (or warmup/2) samples to skip
        # cudagraph-capture and HF-dataset I/O on the first few steps.
        warmup = self.config.auto_tune_sync_period_warmup_steps
        tail = min(25, max(2, warmup // 2))
        median_window = self._warmup_step_times_ms[-tail:]
        t_step_ms = statistics.median(median_window)
        # Effective grace G is whichever wait dominates per outer-period.
        # In production, simulated_merge_delay_ms is 0 (default) and G ==
        # grace_window_ms (the Decoupled DiLoCo adaptive-grace window).
        # In delay-sweep benchmarks, grace_window_ms is set
        # small (e.g. 20ms; quorum=1 reaches quorum immediately) and the
        # simulated_merge_delay_ms is the dominant per-period wait the
        # orchestrator can overlap. max() picks the right one in both cases.
        effective_g_ms = max(
            self.config.grace_window_ms,
            self.config.simulated_merge_delay_ms,
        )
        if t_step_ms <= 0:
            # Defensive: should not happen on real hardware, but if a
            # mock clock returned 0, skip auto-tune and lock the
            # static default.
            n_star = self.config.sync_period_steps
        else:
            raw = (effective_g_ms + t_step_ms - 1) // t_step_ms
            n_star = int(raw)
        clamped = max(
            self.config.auto_tune_sync_period_min,
            min(self.config.sync_period_steps, n_star),
        )
        self.sched.update_sync_period(clamped)
        self._effective_sync_period_steps = clamped
        self._auto_tune_decided = True
        self.diagnostics.emit(
            "auto_tune_sync_period_decided",
            step=step,
            t_step_ms=t_step_ms,
            grace_window_ms=self.config.grace_window_ms,
            simulated_merge_delay_ms=self.config.simulated_merge_delay_ms,
            effective_g_ms=effective_g_ms,
            n_star_raw=n_star,
            effective_sync_period_steps=clamped,
            min_clamp=self.config.auto_tune_sync_period_min,
            max_clamp=self.config.sync_period_steps,
            warmup_samples=warmup,
            median_tail_samples=tail,
        )
        _LOG.info(
            "auto-tuned sync_period_steps: T_step=%.1fms effective_G=%dms n_star=%d (clamped to %d)",
            t_step_ms,
            effective_g_ms,
            n_star,
            clamped,
        )

    # -------- Phase 2 Week 1: concurrent outer-step hooks --------

    def outer_step_begin(
        self,
        round_id: int,
        fragment_provider: FragmentProvider,
    ) -> None:
        """Begin an asynchronous outer-round cross-rack merge.

        Non-blocking. The orchestrator schedules the merge on the
        asyncio loop thread; the training loop should continue issuing
        inner steps (async-TP-overlapped forward / backward keeps GPUs
        busy through the grace window).

        Raises RuntimeError if concurrent_outer_step is False in the
        config (caller should use the synchronous syncer path instead),
        or if a previous round is still PENDING (caller must collect()
        the previous round's MergeResult before submitting a new one).
        """
        if self._concurrent_orch is None:
            raise RuntimeError(
                "outer_step_begin called but concurrent_outer_step is "
                "False in MendConfig; use the synchronous GraceWindowSyncer "
                "path or enable concurrent_outer_step"
            )
        self._concurrent_orch.submit_async(
            round_id=round_id, fragment_provider=fragment_provider
        )
        self.diagnostics.emit(
            "outer_step_begin",
            rank_id=self.rank_id,
            round_id=round_id,
        )

    def outer_step_collect(self) -> Optional[MergeResult]:
        """Non-blocking. Returns the MergeResult of the most recent
        outer-round submission if it has completed; otherwise None.

        After returning a non-None result, the orchestrator is IDLE
        and ready for the next outer_step_begin() call. The caller is
        responsible for actually applying the merged_delta to the
        local model.

        If the asyncio task raised (e.g., quorum not satisfied), the
        captured exception is re-raised on the training thread here.
        """
        if self._concurrent_orch is None:
            return None
        result = self._concurrent_orch.collect()
        if result is not None:
            self.diagnostics.emit(
                "outer_step_collect",
                rank_id=self.rank_id,
                round_id=result.round_id,
                learners_merged=result.learners_merged,
                learners_excluded=result.learners_excluded,
                elapsed_grace_ms=result.elapsed_grace_ms,
                reason=result.reason,
            )
        return result

    def outer_step_in_flight(self) -> bool:
        """Convenience for the training-loop poll. True if an outer-round
        has been submitted via outer_step_begin and the merge has not
        yet been collected via outer_step_collect."""
        if self._concurrent_orch is None:
            return False
        return self._concurrent_orch.is_pending()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()


_RUNTIMES: dict[int, _MaxRuntime] = {}


def mend_init(
    model: nn.Module,
    config: MendConfig,
    rank_id: Optional[str] = None,
) -> None:
    """Wire the tsugiai-mend-sdk runtime onto a (typically already-wrapped)
    PyTorch model.

    Call once after the model has been wrapped (FSDP / TP / etc.). The
    runtime attaches its hooks and starts the sideband.

    `rank_id` should be a stable per-rank identifier. If None, a
    placeholder derived from `id(model)` is used (acceptable for Stage A
    single-process tests).
    """
    if id(model) in _RUNTIMES:
        raise RuntimeError("mend_init already called for this model")
    if rank_id is None:
        rank_id = f"rank-local-{id(model)}"
    runtime = _MaxRuntime(config, rank_id)
    runtime.start(model)
    _RUNTIMES[id(model)] = runtime
    setattr(model, "_max_runtime", runtime)


def mend_shutdown(model: nn.Module) -> None:
    """Tear down the runtime. Idempotent (no-op if not initialized)."""
    runtime = _RUNTIMES.pop(id(model), None)
    if runtime is None:
        return
    runtime.stop()
    if hasattr(model, "_max_runtime"):
        delattr(model, "_max_runtime")


def get_runtime(model: nn.Module) -> _MaxRuntime:
    """Test/benchmark hook to access the runtime for introspection."""
    if id(model) not in _RUNTIMES:
        raise RuntimeError("mend_init not called for this model")
    return _RUNTIMES[id(model)]


__all__ = ["mend_init", "mend_shutdown", "get_runtime"]
