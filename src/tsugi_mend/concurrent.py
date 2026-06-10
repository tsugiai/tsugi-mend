"""Concurrent outer-step orchestrator (Phase 2 Week 1).

Wraps `GraceWindowSyncer` (Decoupled DiLoCo Algorithm 2 state machine) in
an asyncio-task-based orchestrator that lets the local rank's inner-step
training loop continue running async-TP-overlapped forward / backward
while the cross-rack reduce-scatter waits in its grace window.

Convergence guarantee: Decoupled DiLoCo Algorithm 2 already merges between
staggered inner-step blocks. Applying the merged outer-step delta D inner
steps late (D typically in {1..8} in measured workloads, bounded by
ceil(grace_window_ms / T_step) in the worst case) is structurally
equivalent to running the syncer schedule with offset t_p and the learner
schedule with offset t_p + D, both within the valid {0, ..., H-1} offset
range that Algorithm 2 allows. See
`docs/convergence_equivalence_sketch.md` (2026-05-23) for the formal
argument and the empirical validation that loss equivalence is preserved
across the measured D range. The original design discussion is at
`docs/phase2_week1_async_tp_overlap.md`.

Patent-independence note: the orchestrator wraps the public-art Decoupled
DiLoCo GraceWindowSyncer without modifying its control law (quorum +
grace + token-weighted merge). It does NOT introduce a variance-threshold
trigger or K-of-N adapter routing; those mechanisms belong to the
companion patent-aligned SDK (tsugiai-kpool-sdk) and are not present here.

Public API (used by `runtime.py`):

    orch = ConcurrentOuterStep(
        syncer=GraceWindowSyncer(...),
        loop=asyncio_event_loop,
        clock=time.monotonic,
    )
    orch.submit_async(round_id, local_fragment_provider)
    # ... continue inner-step training ...
    result = orch.collect()  # non-blocking; None until merge completes
    if result is not None:
        apply(result.merged_delta)

Stage A unit tests drive this with a controlled clock and a thread-pool
event loop; Stage B+ benchmarks validate the CUDA-stream-level overlap on
real hardware.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional

from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment, MergeResult

if TYPE_CHECKING:
    from concurrent.futures import Future

_LOG = logging.getLogger(__name__)


class _State(Enum):
    """ConcurrentOuterStep state machine."""
    IDLE = auto()           # ready to accept submit_async()
    PENDING = auto()        # asyncio task in flight; collect() returns None
    READY = auto()          # merge complete; collect() returns the MergeResult
    FAILED = auto()         # asyncio task raised; collect() re-raises


# Type alias for the callable that supplies fragments to the asyncio task.
# The local rank provides this callable when it submits. It should yield
# fragments as they arrive (via the sideband, RPC, or in-process queue).
# In Stage A tests the provider yields from a pre-populated list with
# controlled timing. In Stage B+ the provider polls the sideband.
FragmentProvider = Callable[[], "asyncio.Queue[LearnerFragment]"]


@dataclass
class _PendingRound:
    """Mutable state for one outer-round in flight."""
    round_id: int
    task: Future[MergeResult]
    # Captured at submit_async() time so collect() can re-raise deterministically.
    submitted_at_s: float


class ConcurrentOuterStep:
    """Orchestrates `GraceWindowSyncer` on an asyncio event loop so the
    inner-step training loop is not blocked by the outer-step grace
    window.

    Constructor parameters:
        syncer: the GraceWindowSyncer state machine. The orchestrator
            calls syncer.start_round / syncer.submit / syncer.tick on
            the asyncio thread; it does NOT call syncer.set_clock().
        loop: an asyncio event loop running on a non-main thread (the
            same loop the sideband uses, typically).
        clock: monotonic clock callable. Defaults to time.monotonic.
        tick_interval_s: how often the asyncio task polls syncer.tick()
            while waiting for the grace window. Default 0.005 s = 5ms.
            Stage A tests override this to make the unit test
            deterministic.

    The orchestrator is single-round: only one outer-round may be in
    flight at any time. Calling submit_async() while a round is PENDING
    raises RuntimeError. Callers must collect() the previous round before
    submitting the next.

    Thread safety: all public methods (submit_async, collect, last_error)
    are safe to call from the main training thread. Internally they
    marshal work to the asyncio loop thread via run_coroutine_threadsafe.
    """

    def __init__(
        self,
        syncer: GraceWindowSyncer,
        loop: asyncio.AbstractEventLoop,
        clock: Optional[Callable[[], float]] = None,
        tick_interval_s: float = 0.005,
        expected_learner_ids: Optional[frozenset[str]] = None,
    ) -> None:
        self._syncer = syncer
        self._loop = loop
        self._clock = clock or time.monotonic
        self._tick_interval_s = tick_interval_s
        # Default expected-learner roster threaded to syncer.start_round so
        # the world-size-aware early-finalize path goes live. None ->
        # byte-for-byte the historical quorum-then-full-grace behavior. A
        # per-round override may be supplied to submit_async(). See the
        # roster-id contract on MendConfig.expected_learner_ids.
        self._expected_learner_ids: Optional[frozenset[str]] = expected_learner_ids

        self._state: _State = _State.IDLE
        self._pending: Optional[_PendingRound] = None
        self._result: Optional[MergeResult] = None
        self._error: Optional[BaseException] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (training-thread-safe)
    # ------------------------------------------------------------------

    def submit_async(
        self,
        round_id: int,
        fragment_provider: FragmentProvider,
        expected_learner_ids: Optional[frozenset[str]] = None,
    ) -> None:
        """Submit an outer-round for asynchronous merge. Returns immediately.

        The fragment_provider is called once, on the asyncio thread, to
        obtain a queue from which the asyncio task pulls fragments. The
        queue may be populated by the caller after submit_async() returns;
        the asyncio task waits on queue.get() with a timeout bounded by
        the grace window.

        `expected_learner_ids` optionally overrides the orchestrator-level
        roster for THIS round only. When omitted, the constructor-supplied
        default roster is used; when both are None the syncer takes the
        historical quorum-then-full-grace path (byte-for-byte). See the
        roster-id contract on MendConfig.expected_learner_ids; a roster that
        cannot be reconciled with quorum falls back to the None path rather
        than raising or hanging.

        Raises RuntimeError if a previous round is still PENDING.
        """
        roster = (
            expected_learner_ids
            if expected_learner_ids is not None
            else self._expected_learner_ids
        )
        with self._lock:
            if self._state == _State.PENDING:
                raise RuntimeError(
                    "submit_async called while round in PENDING state; "
                    "call collect() first or wait for the merge to complete"
                )
            # Clear any prior READY/FAILED state on new submission.
            self._state = _State.PENDING
            self._result = None
            self._error = None

        # Schedule the merge coroutine on the asyncio thread.
        coro = self._run_merge(round_id, fragment_provider, roster)
        task = asyncio.run_coroutine_threadsafe(coro, self._loop)
        # Attach a done-callback so we transition to READY/FAILED without
        # the caller having to await.
        task.add_done_callback(self._on_task_done)

        with self._lock:
            self._pending = _PendingRound(
                round_id=round_id,
                task=task,
                submitted_at_s=self._clock(),
            )

    def collect(self) -> Optional[MergeResult]:
        """Non-blocking. Returns the MergeResult if the asyncio task has
        completed; otherwise returns None.

        If the asyncio task raised, the captured exception is re-raised
        here on the training thread (deterministic exception ordering).
        After re-raise, the orchestrator returns to IDLE.
        """
        with self._lock:
            if self._state == _State.IDLE:
                return None
            if self._state == _State.PENDING:
                return None
            if self._state == _State.FAILED:
                err = self._error
                self._reset_locked()
                assert err is not None
                raise err
            # READY
            result = self._result
            self._reset_locked()
            return result

    def last_error(self) -> Optional[BaseException]:
        """Inspection-only; returns the captured exception if any.
        Does not change state. Useful for diagnostics."""
        with self._lock:
            return self._error

    def state(self) -> str:
        """Inspection-only; returns the current state name. Useful for
        the diagnostics writer and for testing."""
        with self._lock:
            return self._state.name

    def is_pending(self) -> bool:
        """Convenience for the training-loop poll. Equivalent to
        state() == 'PENDING'."""
        with self._lock:
            return self._state == _State.PENDING

    # ------------------------------------------------------------------
    # Internal (asyncio thread)
    # ------------------------------------------------------------------

    async def _run_merge(
        self,
        round_id: int,
        fragment_provider: FragmentProvider,
        expected_learner_ids: Optional[frozenset[str]] = None,
    ) -> MergeResult:
        """Coroutine body: pull fragments from the provider's queue,
        drive the GraceWindowSyncer until tick() returns a MergeResult."""
        # SAFE FALLBACK: only hand the roster to the world-size-aware
        # start_round path when it can be reconciled with quorum. A roster
        # smaller than quorum_min could never satisfy quorum and would make
        # start_round raise; in that misdeclared case we drop to the
        # expected=None path (quorum, then full grace) so the round neither
        # hangs nor changes the merged result. A roster naming learners that
        # never arrive is already safe in the syncer: early-finalize simply
        # never fires and the round falls through to grace expiry, with the
        # absentees surfaced in MergeResult.learners_absent.
        roster = expected_learner_ids
        if roster is not None and self._syncer.quorum_min > len(roster):
            _LOG.warning(
                "ConcurrentOuterStep round %s: declared roster of %d learner(s) "
                "is smaller than quorum_min=%d; ignoring roster and using the "
                "quorum-then-grace path",
                round_id,
                len(roster),
                self._syncer.quorum_min,
            )
            roster = None
        if roster is None:
            self._syncer.start_round(round_id)
        else:
            self._syncer.start_round(round_id, expected_learner_ids=set(roster))
        queue = fragment_provider()

        # Bound the total wait by 2x the grace window plus the per-tick
        # interval. This is generous; the syncer will normally finalize
        # earlier than this bound. If the caller never delivers quorum,
        # the bound triggers a fail-open by raising; the runtime's
        # exception handler logs and continues with the local rank's
        # params (the desired fail-open behavior).
        grace_s = self._syncer.grace_window_ms / 1000.0
        deadline_s = self._clock() + max(2.0 * grace_s, 1.0)

        while True:
            remaining_s = deadline_s - self._clock()
            if remaining_s <= 0:
                # Force finalize; this raises if quorum is not satisfied,
                # which is the desired fail-fast behavior.
                return self._syncer.finalize_on_timeout()
            try:
                # Wait for the next fragment (or until the next tick).
                fragment = await asyncio.wait_for(
                    queue.get(),
                    timeout=self._tick_interval_s,
                )
            except asyncio.TimeoutError:
                fragment = None

            if fragment is not None:
                result = self._syncer.submit(fragment)
                if result is not None:
                    return result

            # Even without a new fragment, tick() may complete the round
            # if the grace window elapsed since the last fragment arrived.
            result = self._syncer.tick()
            if result is not None:
                return result

    def _on_task_done(self, task: Future[MergeResult]) -> None:
        """asyncio thread callback: transition the state machine on
        completion or exception."""
        try:
            result = task.result()
        except BaseException as e:  # pylint: disable=broad-except
            with self._lock:
                self._error = e
                self._state = _State.FAILED
            _LOG.warning(
                "ConcurrentOuterStep round %s failed: %s",
                self._pending.round_id if self._pending else "?",
                e,
            )
            return
        with self._lock:
            self._result = result
            self._state = _State.READY

    def _reset_locked(self) -> None:
        """Caller must hold self._lock. Resets to IDLE for next round."""
        self._state = _State.IDLE
        self._result = None
        self._error = None
        self._pending = None


__all__ = ["ConcurrentOuterStep", "FragmentProvider"]
