"""Stage A unit tests for ConcurrentOuterStep (Phase 2 Week 1).

These tests exercise the asyncio-task-based orchestrator that wraps
GraceWindowSyncer. They drive a real asyncio event loop on a background
thread (the same pattern the runtime uses for the sideband) and submit
fragments through a real asyncio.Queue.

Patent-independence note: these tests do not exercise variance-threshold
triggers, K-of-N adapter routing, or LoRA-adapter-granularity reduction.
They exercise the public-art Decoupled DiLoCo grace-window control law
under a new orchestration layer.

Stage A scope: CPU-only, deterministic, fast. CUDA-stream-level overlap
is validated by the Stage B+ benchmarks.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
import torch

from tsugi_mend.concurrent import ConcurrentOuterStep
from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop_on_thread():
    """Run a fresh asyncio event loop on a background thread, the way
    the production runtime runs the sideband loop. Tears down cleanly."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(timeout=1.0)

    yield loop

    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)


def _make_fragment(learner_id: str, round_id: int, n_params: int = 2) -> LearnerFragment:
    """A tiny fragment for the unit tests. Two parameters of shape (3,)."""
    return LearnerFragment(
        learner_id=learner_id,
        round_id=round_id,
        params_delta=[torch.full((3,), float(i)) for i in range(n_params)],
        tokens_consumed=100 + ord(learner_id[-1]),  # varied weights
    )


def _provider_from_list(fragments_to_send, send_delay_s: float = 0.0):
    """Make a fragment_provider that drips fragments into the orchestrator's
    queue with a controlled delay between submissions."""
    queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()

    async def _drip() -> None:
        for f in fragments_to_send:
            if send_delay_s > 0:
                await asyncio.sleep(send_delay_s)
            await queue.put(f)

    def provider() -> "asyncio.Queue[LearnerFragment]":
        # Schedule the drip coroutine on the same loop as the merge task.
        asyncio.get_event_loop().create_task(_drip())
        return queue

    return provider


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initial_state_is_idle(event_loop_on_thread):
    """Orchestrator starts in IDLE; collect() returns None."""
    syncer = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=100)
    orch = ConcurrentOuterStep(syncer=syncer, loop=event_loop_on_thread)
    assert orch.state() == "IDLE"
    assert not orch.is_pending()
    assert orch.collect() is None
    assert orch.last_error() is None


def test_submit_async_returns_immediately_and_collect_returns_result(event_loop_on_thread):
    """The submit_async() call must return immediately (not block on the
    grace window), and collect() must eventually return the MergeResult."""
    syncer = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=20)
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )
    frags = [
        _make_fragment("rack-a", round_id=1),
        _make_fragment("rack-b", round_id=1),
    ]
    provider = _provider_from_list(frags, send_delay_s=0.0)

    t0 = time.monotonic()
    orch.submit_async(round_id=1, fragment_provider=provider)
    submit_elapsed_ms = (time.monotonic() - t0) * 1000.0

    # submit_async must not block on the grace window.
    assert submit_elapsed_ms < 10.0, f"submit_async blocked {submit_elapsed_ms:.1f}ms"
    # Immediately after submit, we're PENDING.
    assert orch.is_pending() or orch.state() == "READY"

    # Poll for completion with a generous bound.
    deadline = time.monotonic() + 2.0
    result = None
    while time.monotonic() < deadline:
        result = orch.collect()
        if result is not None:
            break
        time.sleep(0.005)

    assert result is not None, "merge did not complete within 2s"
    assert result.round_id == 1
    assert sorted(result.learners_merged) == ["rack-a", "rack-b"]
    assert len(result.merged_delta) == 2
    assert result.merged_delta[0].shape == (3,)
    # After collect(), state should be IDLE again.
    assert orch.state() == "IDLE"


def test_double_submit_raises_runtime_error(event_loop_on_thread):
    """Submitting a second round while the first is PENDING must raise."""
    syncer = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=500)
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )
    # Provider that never delivers quorum: we want the round to stay PENDING.
    def never_provider() -> "asyncio.Queue[LearnerFragment]":
        return asyncio.Queue()

    orch.submit_async(round_id=1, fragment_provider=never_provider)

    with pytest.raises(RuntimeError, match="PENDING"):
        orch.submit_async(round_id=2, fragment_provider=never_provider)

    # Drain the in-flight task by waiting for it to fail-fast on the
    # 2 * grace_window_ms timeout (=1.0s); then verify the FAILED state.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if orch.state() in ("READY", "FAILED"):
            break
        time.sleep(0.01)
    assert orch.state() == "FAILED"
    # collect() re-raises and resets to IDLE.
    with pytest.raises(RuntimeError):
        orch.collect()
    assert orch.state() == "IDLE"


def test_collect_returns_none_while_pending(event_loop_on_thread):
    """collect() must be non-blocking and return None while PENDING."""
    syncer = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=200)
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )
    # Provider that delivers slowly.
    frags = [
        _make_fragment("rack-a", round_id=1),
        _make_fragment("rack-b", round_id=1),
    ]
    provider = _provider_from_list(frags, send_delay_s=0.1)

    orch.submit_async(round_id=1, fragment_provider=provider)
    # Immediately check collect(); the merge should not be done yet.
    t0 = time.monotonic()
    poll_result = orch.collect()
    poll_elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert poll_elapsed_ms < 5.0, f"collect() blocked {poll_elapsed_ms:.1f}ms"
    assert poll_result is None

    # Now wait for completion.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if orch.state() == "READY":
            break
        time.sleep(0.01)
    final = orch.collect()
    assert final is not None
    assert sorted(final.learners_merged) == ["rack-a", "rack-b"]


def test_failure_is_reraised_on_collect(event_loop_on_thread):
    """Asyncio-task exceptions must be deterministically re-raised on
    the training thread via collect(), not silently swallowed."""
    syncer = GraceWindowSyncer(quorum_min_learners=4, grace_window_ms=20)
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )
    # Only one fragment; quorum=4 means the finalize_on_timeout call will
    # raise inside the asyncio task.
    frags = [_make_fragment("rack-a", round_id=1)]
    provider = _provider_from_list(frags, send_delay_s=0.0)

    orch.submit_async(round_id=1, fragment_provider=provider)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if orch.state() == "FAILED":
            break
        time.sleep(0.01)
    assert orch.state() == "FAILED"
    assert isinstance(orch.last_error(), RuntimeError)
    with pytest.raises(RuntimeError, match="finalize_on_timeout"):
        orch.collect()
    # After re-raise we're back to IDLE for the next round.
    assert orch.state() == "IDLE"
    assert orch.last_error() is None


def test_consecutive_rounds_succeed(event_loop_on_thread):
    """After collect()-ing round 1, we can submit round 2 without issue."""
    syncer = GraceWindowSyncer(quorum_min_learners=2, grace_window_ms=20)
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )

    def run_round(round_id: int) -> None:
        frags = [
            _make_fragment("rack-a", round_id=round_id),
            _make_fragment("rack-b", round_id=round_id),
        ]
        orch.submit_async(round_id=round_id, fragment_provider=_provider_from_list(frags))
        deadline = time.monotonic() + 2.0
        result = None
        while time.monotonic() < deadline:
            result = orch.collect()
            if result is not None:
                break
            time.sleep(0.005)
        assert result is not None
        assert result.round_id == round_id

    for rid in (1, 2, 3):
        run_round(rid)
    assert orch.state() == "IDLE"


def test_token_weighted_merge_is_preserved(event_loop_on_thread):
    """The orchestrator must not change the merge semantics; the
    token-weighted-merge math comes through unchanged from
    GraceWindowSyncer._finalize."""
    syncer = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=20,
        token_weighted=True,
    )
    orch = ConcurrentOuterStep(
        syncer=syncer,
        loop=event_loop_on_thread,
        tick_interval_s=0.002,
    )
    # Two learners with different token counts and different deltas.
    f1 = LearnerFragment(
        learner_id="rack-a",
        round_id=1,
        params_delta=[torch.tensor([1.0, 1.0, 1.0])],
        tokens_consumed=100,
    )
    f2 = LearnerFragment(
        learner_id="rack-b",
        round_id=1,
        params_delta=[torch.tensor([3.0, 3.0, 3.0])],
        tokens_consumed=300,
    )
    # Expected merge: (100*1 + 300*3) / (100+300) = 1000/400 = 2.5
    orch.submit_async(round_id=1, fragment_provider=_provider_from_list([f1, f2]))
    deadline = time.monotonic() + 2.0
    result = None
    while time.monotonic() < deadline:
        result = orch.collect()
        if result is not None:
            break
        time.sleep(0.005)
    assert result is not None
    assert torch.allclose(result.merged_delta[0], torch.tensor([2.5, 2.5, 2.5]))
