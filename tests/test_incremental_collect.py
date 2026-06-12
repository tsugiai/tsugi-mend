"""Ordering-proof tests for the opt-in straggler-aware incremental collect.

The acceptance bar (bit-exact by construction): for EVERY arrival pattern,
the incremental-collect syncer's merged tensors must be `torch.equal`-identical
(and raw-bit-identical, which is stronger: `==` treats -0.0 and +0.0 as equal)
to the default frozen path driven with the byte-identical arrival sequence and
clock, and every other `MergeResult` field must match exactly.

Anti-vacuity discipline: every cell that claims the incremental path was
exercised asserts `syncer.last_merge_used_incremental is True`, so a bug that
silently falls back to the frozen path on every round cannot pass these tests
by accident. Fallback cells assert `is False` for the same reason.
"""
from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

from tsugi_mend.config import MendConfig
from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment, MergeResult

GRACE_MS = 500.0

# Deliberately disparate token weights (orders of magnitude apart) so the
# token-weighted variant's accumulation order genuinely matters in floating
# point. Indexed by learner index.
_TOKENS = [3, 1_000_003, 7, 88, 5, 12_345, 2, 999_983]

_ORDER_NAMES = [
    "canonical",
    "reverse",
    "random0",
    "random1",
    "laggard_last",
    "laggard_first",
]

_BIT_VIEW_DTYPE = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float64: torch.int64,
}


class FakeClock:
    """Deterministic monotonic clock for tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _rack_ids(n: int) -> list[str]:
    return [f"rack-{i}" for i in range(n)]


def _adversarial_fragment(
    learner_id: str, idx: int, round_id: int, dtype: torch.dtype
) -> LearnerFragment:
    """Deterministic per-learner payload with adversarial float content:
    -0.0, positive and negative subnormals (denormals), large/small magnitude
    mix with cancellation partners, near-underflow values. No NaNs (torch.equal
    is the acceptance operator and NaN != NaN)."""
    g = torch.Generator().manual_seed(10_000 + 97 * idx)
    tiny = torch.finfo(dtype).tiny
    p0 = torch.randn(13, generator=g, dtype=torch.float32).to(dtype)
    p0[0] = -0.0
    p0[1] = tiny / 4.0  # positive subnormal
    p0[2] = -tiny / 4.0  # negative subnormal
    p0[3] = 3.7e30  # large magnitude
    p0[4] = -3.7e30  # cancellation partner
    p0[5] = 1.1e-30  # small magnitude
    p0[6] = 0.0
    p1 = (torch.randn(4, 5, generator=g, dtype=torch.float32) * 1.0e6).to(dtype)
    p1[0, 0] = -0.0
    p1[1, 1] = tiny / 2.0
    p1[2, 2] = -1.0e-20
    p1[3, 3] = float(idx + 1) * 1.0e-38
    return LearnerFragment(
        learner_id=learner_id,
        round_id=round_id,
        params_delta=[p0, p1],
        tokens_consumed=_TOKENS[idx],
    )


def _arrival_plan(order_name: str, n: int) -> tuple[list[int], list[float]]:
    """Return (learner indices in arrival order, pre-submit clock gaps in ms)."""
    idxs = list(range(n))
    if order_name == "canonical":
        return idxs, [10.0] * n
    if order_name == "reverse":
        return list(reversed(idxs)), [10.0] * n
    if order_name.startswith("random"):
        rng = random.Random(int(order_name[-1]) + 17 * n)
        rng.shuffle(idxs)
        return idxs, [10.0] * n
    if order_name == "laggard_last":
        # Learner 0 is the laggard; it arrives after the grace window has
        # already expired, so (for n >= 3) its own submit() finalizes the
        # round with the full fragment set.
        return idxs[1:] + [0], [10.0] * (n - 1) + [GRACE_MS + 250.0]
    if order_name == "laggard_first":
        # Last round's laggard reports first this round.
        return [n - 1] + idxs[: n - 1], [10.0] * n
    raise AssertionError(f"unknown order {order_name!r}")


def _drive(
    *,
    incremental: bool,
    arrival_idxs: list[int],
    gaps_ms: list[float],
    n: int,
    token_weighted: bool,
    dtype: torch.dtype,
    quorum_min: int | None = None,
    grace_window_ms: int = int(GRACE_MS),
    expected_learner_ids: set[str] | None = None,
    round_id: int = 11,
) -> tuple[MergeResult, GraceWindowSyncer]:
    """Drive one syncer round with a fully deterministic arrival sequence.

    Fragments are built fresh inside the drive (deterministic seeds make the
    payload bits identical across drives), so the reference and incremental
    runs see byte-identical inputs and clocks.
    """
    ids = _rack_ids(n)
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=quorum_min if quorum_min is not None else min(2, n),
        grace_window_ms=grace_window_ms,
        token_weighted=token_weighted,
        clock=clock,
        incremental_collect=incremental,
    )
    sync.start_round(round_id=round_id, expected_learner_ids=expected_learner_ids)
    out: MergeResult | None = None
    for gap_ms, idx in zip(gaps_ms, arrival_idxs):
        clock.advance_ms(gap_ms)
        out = sync.submit(_adversarial_fragment(ids[idx], idx, round_id, dtype))
        if out is not None:
            break
    if out is None:
        clock.advance_ms(grace_window_ms + 100.0)
        out = sync.tick()
    assert out is not None
    return out, sync


def _assert_bitwise_identical(ref: list[Tensor], got: list[Tensor]) -> None:
    assert len(got) == len(ref)
    for ta, tb in zip(ref, got):
        assert tb.dtype == ta.dtype
        assert tb.shape == ta.shape
        # The contract operator.
        assert torch.equal(ta, tb)
        # Strictly stronger raw-bit identity (catches -0.0 vs +0.0).
        view_dtype = _BIT_VIEW_DTYPE[ta.dtype]
        assert torch.equal(
            ta.contiguous().view(view_dtype), tb.contiguous().view(view_dtype)
        )


def _assert_same_result(ref: MergeResult, got: MergeResult) -> None:
    assert got.round_id == ref.round_id
    assert got.learners_merged == ref.learners_merged
    assert got.learners_excluded == ref.learners_excluded
    assert got.learners_absent == ref.learners_absent
    assert got.elapsed_grace_ms == ref.elapsed_grace_ms
    assert got.reason == ref.reason
    _assert_bitwise_identical(ref.merged_delta, got.merged_delta)


# ----------------------------------------------------------------------
# The ordering proof: every arrival pattern x learner count x merge variant
# x payload dtype must be bit-identical to the frozen default path.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "token_weighted", [True, False], ids=["token_weighted", "uniform"]
)
@pytest.mark.parametrize("n", [2, 3, 4, 8])
@pytest.mark.parametrize("order_name", _ORDER_NAMES)
def test_ordering_proof_bit_exact(
    order_name: str, n: int, token_weighted: bool, dtype: torch.dtype
) -> None:
    arrival_idxs, gaps_ms = _arrival_plan(order_name, n)
    ref, ref_sync = _drive(
        incremental=False,
        arrival_idxs=arrival_idxs,
        gaps_ms=gaps_ms,
        n=n,
        token_weighted=token_weighted,
        dtype=dtype,
    )
    got, inc_sync = _drive(
        incremental=True,
        arrival_idxs=arrival_idxs,
        gaps_ms=gaps_ms,
        n=n,
        token_weighted=token_weighted,
        dtype=dtype,
    )
    # Anti-vacuity: the incremental accumulation, not a silent fallback,
    # produced this round's merge. The reference run must report "mode off".
    assert inc_sync.last_merge_used_incremental is True
    assert ref_sync.last_merge_used_incremental is None
    _assert_same_result(ref, got)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "token_weighted", [True, False], ids=["token_weighted", "uniform"]
)
@pytest.mark.parametrize("n", [2, 4, 8])
def test_ordering_proof_all_present_early_finalize(
    n: int, token_weighted: bool, dtype: torch.dtype
) -> None:
    """The expected-roster early-finalize ("all_present") path must be
    bit-identical too: the round finalizes inside the last expected learner's
    submit(), with the incremental accumulation already complete."""
    arrival_idxs, gaps_ms = _arrival_plan("reverse", n)
    expected = set(_rack_ids(n))
    ref, _ = _drive(
        incremental=False,
        arrival_idxs=arrival_idxs,
        gaps_ms=gaps_ms,
        n=n,
        token_weighted=token_weighted,
        dtype=dtype,
        grace_window_ms=1_000_000,
        expected_learner_ids=expected,
    )
    got, inc_sync = _drive(
        incremental=True,
        arrival_idxs=arrival_idxs,
        gaps_ms=gaps_ms,
        n=n,
        token_weighted=token_weighted,
        dtype=dtype,
        grace_window_ms=1_000_000,
        expected_learner_ids=expected,
    )
    assert ref.reason == "all_present"
    assert inc_sync.last_merge_used_incremental is True
    _assert_same_result(ref, got)


# ----------------------------------------------------------------------
# Fallback cases: anything the incremental mode cannot prove order-identical
# must take the frozen path for the round and still match the default mode
# exactly (results AND error behavior).
# ----------------------------------------------------------------------


def _drive_with_resubmission(
    *,
    incremental: bool,
    token_weighted: bool,
    dtype: torch.dtype,
    resubmit_position: str,
) -> tuple[MergeResult, GraceWindowSyncer]:
    """rack-0 submits, then resubmits a different payload (mid-stream or
    last). The frozen dict semantics keep rack-0's ORIGINAL insertion
    position with the NEW value, which incremental accumulation cannot
    reproduce bit-exactly; the round must fall back."""
    n = 4
    ids = _rack_ids(n)
    round_id = 23
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=int(GRACE_MS),
        token_weighted=token_weighted,
        clock=clock,
        incremental_collect=incremental,
    )
    sync.start_round(round_id=round_id)
    # rack-0's second submission carries a DIFFERENT payload and token count
    # (the idx-5 generator) than its original idx-0 fragment.
    resubmit = _adversarial_fragment(ids[0], 5, round_id, dtype)
    submissions = [_adversarial_fragment(ids[i], i, round_id, dtype) for i in range(n)]
    if resubmit_position == "mid":
        seq = [submissions[0], submissions[1], resubmit, submissions[2], submissions[3]]
    else:
        seq = submissions + [resubmit]
    out: MergeResult | None = None
    for frag in seq:
        clock.advance_ms(10.0)
        out = sync.submit(frag)
        assert out is None
    clock.advance_ms(GRACE_MS + 100.0)
    out = sync.tick()
    assert out is not None
    return out, sync


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "token_weighted", [True, False], ids=["token_weighted", "uniform"]
)
@pytest.mark.parametrize("resubmit_position", ["mid", "last"])
def test_fallback_on_resubmission_is_bit_exact(
    resubmit_position: str, token_weighted: bool, dtype: torch.dtype
) -> None:
    ref, _ = _drive_with_resubmission(
        incremental=False,
        token_weighted=token_weighted,
        dtype=dtype,
        resubmit_position=resubmit_position,
    )
    got, inc_sync = _drive_with_resubmission(
        incremental=True,
        token_weighted=token_weighted,
        dtype=dtype,
        resubmit_position=resubmit_position,
    )
    # The round must DETECT the resubmission and fall back to the frozen path.
    assert inc_sync.last_merge_used_incremental is False
    _assert_same_result(ref, got)


def _drive_mismatched_param_count(incremental: bool) -> tuple[str, GraceWindowSyncer]:
    """Two fragments with different param counts: the frozen path raises
    ValueError at finalize. The incremental mode must reproduce the exact
    error (message and finalize-time timing), via fallback."""
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=int(GRACE_MS),
        token_weighted=True,
        clock=clock,
        incremental_collect=incremental,
    )
    sync.start_round(round_id=5)
    two_param = _adversarial_fragment("rack-0", 0, 5, torch.float32)
    one_param = LearnerFragment(
        learner_id="rack-1",
        round_id=5,
        params_delta=[torch.tensor([1.0, 2.0])],
        tokens_consumed=100,
    )
    # Neither submit may raise; the error belongs to finalize in both modes.
    assert sync.submit(two_param) is None
    assert sync.submit(one_param) is None
    clock.advance_ms(GRACE_MS + 100.0)
    with pytest.raises(ValueError, match="params; expected") as excinfo:
        sync.tick()
    return str(excinfo.value), sync


def test_fallback_on_param_count_mismatch_raises_identically() -> None:
    ref_msg, _ = _drive_mismatched_param_count(incremental=False)
    got_msg, _ = _drive_mismatched_param_count(incremental=True)
    assert got_msg == ref_msg


@pytest.mark.parametrize("tokens_pair", [(0, 0), (5, -5)], ids=["zero", "cancel"])
def test_fallback_on_non_positive_token_total_raises_identically(
    tokens_pair: tuple[int, int],
) -> None:
    """token_weighted_merge owns the non-positive-total ValueError; the
    incremental mode must fall back so the frozen path raises it
    identically."""

    def drive(incremental: bool) -> str:
        clock = FakeClock()
        sync = GraceWindowSyncer(
            quorum_min_learners=2,
            grace_window_ms=int(GRACE_MS),
            token_weighted=True,
            clock=clock,
            incremental_collect=incremental,
        )
        sync.start_round(round_id=6)
        for i, tokens in enumerate(tokens_pair):
            frag = _adversarial_fragment(f"rack-{i}", i, 6, torch.float32)
            frag.tokens_consumed = tokens
            assert sync.submit(frag) is None
        clock.advance_ms(GRACE_MS + 100.0)
        with pytest.raises(ValueError, match="positive total tokens") as excinfo:
            sync.tick()
        return str(excinfo.value)

    assert drive(True) == drive(False)


def _drive_with_post_submit_mutation(
    *, incremental: bool, mutation: str, token_weighted: bool
) -> tuple[MergeResult, GraceWindowSyncer]:
    """rack-1's fragment is altered AFTER submit but BEFORE finalize. The
    frozen path reads fragments at finalize time, so the default mode merges
    the post-mutation values; the incremental accumulation baked in the
    pre-mutation values and must DETECT the alteration at finalize and fall
    back, landing on the same post-mutation result."""
    n = 3
    round_id = 41
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=int(GRACE_MS),
        token_weighted=token_weighted,
        clock=clock,
        incremental_collect=incremental,
    )
    sync.start_round(round_id=round_id)
    frags = [
        _adversarial_fragment(f"rack-{i}", i, round_id, torch.float32) for i in range(n)
    ]
    for frag in frags:
        clock.advance_ms(10.0)
        assert sync.submit(frag) is None
    # Post-submit alteration of rack-1's already-accumulated fragment.
    victim = frags[1]
    if mutation == "inplace_tensor":
        victim.params_delta[0].add_(1.0)  # bumps the PyTorch version counter
    elif mutation == "rebound_tensor":
        victim.params_delta[1] = torch.full((4, 5), 2.5)
    elif mutation == "tokens":
        victim.tokens_consumed += 1
    else:
        raise AssertionError(f"unknown mutation {mutation!r}")
    clock.advance_ms(GRACE_MS + 100.0)
    out = sync.tick()
    assert out is not None
    return out, sync


@pytest.mark.parametrize(
    "token_weighted", [True, False], ids=["token_weighted", "uniform"]
)
@pytest.mark.parametrize("mutation", ["inplace_tensor", "rebound_tensor", "tokens"])
def test_fallback_on_post_submit_mutation_is_bit_exact(
    mutation: str, token_weighted: bool
) -> None:
    # Note the ("tokens", uniform) cell: uniform_merge never reads
    # tokens_consumed, but the integrity check is deliberately variant-blind
    # (cheap and simple), so the round still falls back and still matches
    # the default path bit-for-bit.
    ref, _ = _drive_with_post_submit_mutation(
        incremental=False, mutation=mutation, token_weighted=token_weighted
    )
    got, inc_sync = _drive_with_post_submit_mutation(
        incremental=True, mutation=mutation, token_weighted=token_weighted
    )
    # The mutation must be DETECTED; the frozen path then reads the
    # post-mutation values exactly like the default mode does.
    assert inc_sync.last_merge_used_incremental is False
    _assert_same_result(ref, got)


# ----------------------------------------------------------------------
# Rejected submissions must not contaminate the accumulation, and per-round
# state must reset cleanly.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "token_weighted", [True, False], ids=["token_weighted", "uniform"]
)
def test_rejected_fragments_do_not_contaminate_accumulation(
    token_weighted: bool,
) -> None:
    """Fail-slow-excluded and stale-round fragments are rejected before
    storage; the incremental accumulation must skip them exactly like the
    fragment store does."""

    def drive(incremental: bool) -> tuple[MergeResult, GraceWindowSyncer]:
        clock = FakeClock()
        sync = GraceWindowSyncer(
            quorum_min_learners=2,
            grace_window_ms=int(GRACE_MS),
            token_weighted=token_weighted,
            clock=clock,
            incremental_collect=incremental,
        )
        sync.start_round(round_id=9)
        sync.mark_failslow("rack-9")
        poisoned = _adversarial_fragment("rack-9", 7, 9, torch.float32)
        assert sync.submit(poisoned) is None  # excluded; must not accumulate
        stale = _adversarial_fragment("rack-5", 5, 8, torch.float32)  # wrong round
        assert sync.submit(stale) is None
        for i in range(3):
            clock.advance_ms(10.0)
            sync.submit(_adversarial_fragment(f"rack-{i}", i, 9, torch.float32))
        clock.advance_ms(GRACE_MS + 100.0)
        out = sync.tick()
        assert out is not None
        return out, sync

    ref, _ = drive(False)
    got, inc_sync = drive(True)
    assert inc_sync.last_merge_used_incremental is True
    assert got.learners_excluded == ["rack-9"]
    _assert_same_result(ref, got)


def test_failslow_after_arrival_keeps_stored_fragment_identically() -> None:
    """mark_failslow only blocks FUTURE submissions; a fragment that already
    arrived stays in the store and merges (frozen semantics). The
    incremental accumulation already folded it in, so both modes must agree
    bit-for-bit, including the excluded-list bookkeeping."""

    def drive(incremental: bool) -> tuple[MergeResult, GraceWindowSyncer]:
        clock = FakeClock()
        sync = GraceWindowSyncer(
            quorum_min_learners=2,
            grace_window_ms=int(GRACE_MS),
            token_weighted=True,
            clock=clock,
            incremental_collect=incremental,
        )
        sync.start_round(round_id=12)
        sync.submit(_adversarial_fragment("rack-0", 0, 12, torch.float32))
        sync.mark_failslow("rack-0")  # after arrival: fragment stays merged
        for i in (1, 2):
            clock.advance_ms(10.0)
            sync.submit(_adversarial_fragment(f"rack-{i}", i, 12, torch.float32))
        clock.advance_ms(GRACE_MS + 100.0)
        out = sync.tick()
        assert out is not None
        return out, sync

    ref, _ = drive(False)
    got, inc_sync = drive(True)
    assert inc_sync.last_merge_used_incremental is True
    assert "rack-0" in got.learners_merged
    # ENG-15 semantics: a learner marked fail-slow AFTER its fragment was
    # accepted stays merged and is NOT listed as excluded; the diagnostic
    # must agree between the incremental and default paths.
    assert got.learners_excluded == ref.learners_excluded == []
    _assert_same_result(ref, got)


def test_multi_round_reuse_resets_fallback_state() -> None:
    """One syncer across three rounds: normal, resubmission-fallback, normal
    again. The fallback must be per-round, not sticky across rounds, and
    every round must match a default-mode reference driven identically."""

    def run_round(
        sync: GraceWindowSyncer,
        clock: FakeClock,
        round_id: int,
        with_resubmission: bool,
    ) -> MergeResult:
        sync.start_round(round_id=round_id)
        for i in range(3):
            clock.advance_ms(10.0)
            sync.submit(_adversarial_fragment(f"rack-{i}", i, round_id, torch.float32))
        if with_resubmission:
            clock.advance_ms(10.0)
            # rack-1 resubmits with the idx-3 payload/tokens (differs from
            # its original idx-1 fragment).
            sync.submit(_adversarial_fragment("rack-1", 3, round_id, torch.float32))
        clock.advance_ms(GRACE_MS + 100.0)
        out = sync.tick()
        assert out is not None
        return out

    ref_clock = FakeClock()
    inc_clock = FakeClock()
    ref_sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=int(GRACE_MS),
        token_weighted=True,
        clock=ref_clock,
    )
    inc_sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=int(GRACE_MS),
        token_weighted=True,
        clock=inc_clock,
        incremental_collect=True,
    )
    plan = [(31, False), (32, True), (33, False)]
    expected_used = [True, False, True]
    for (round_id, with_resub), used in zip(plan, expected_used):
        ref = run_round(ref_sync, ref_clock, round_id, with_resub)
        got = run_round(inc_sync, inc_clock, round_id, with_resub)
        assert inc_sync.last_merge_used_incremental is used
        _assert_same_result(ref, got)


# ----------------------------------------------------------------------
# Default-OFF guarantees and the config knob.
# ----------------------------------------------------------------------


def test_default_off_attribute_and_observability() -> None:
    sync = GraceWindowSyncer(quorum_min_learners=1, grace_window_ms=0)
    assert sync.incremental_collect is False
    assert sync.last_merge_used_incremental is None
    sync.start_round(round_id=1)
    out = sync.submit(_adversarial_fragment("rack-0", 0, 1, torch.float32))
    assert out is not None
    # Mode off: the observability flag stays None (frozen path, no
    # incremental machinery consulted).
    assert sync.last_merge_used_incremental is None


def test_config_knob_defaults_off_and_accepts_true() -> None:
    assert MendConfig().incremental_collect is False
    assert MendConfig(incremental_collect=True).incremental_collect is True
