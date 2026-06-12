"""Ordering-proof tests for opt-in incremental reducer collection."""
from __future__ import annotations

import random

import pytest
import torch

from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment, MergeResult


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _learner_id(index: int) -> str:
    return f"rack-{index}"


def _fragment(index: int, round_id: int, dtype: torch.dtype) -> LearnerFragment:
    sign = -1.0 if index % 2 else 1.0
    tiny = torch.finfo(dtype).smallest_normal
    payload = torch.tensor(
        [
            sign * (10_000.0 + index),
            sign * tiny,
            -0.0 if index == 0 else 0.0,
            float(index + 1) / 7.0,
        ],
        dtype=dtype,
    )
    return LearnerFragment(
        learner_id=_learner_id(index),
        round_id=round_id,
        params_delta=[payload],
        tokens_consumed=11 + 3 * index,
    )


def _arrival_order(kind: str, n: int) -> list[int]:
    if kind == "in_order":
        return list(range(n))
    if kind == "reverse":
        return list(reversed(range(n)))
    if kind == "random":
        order = list(range(n))
        random.Random(1701 + n).shuffle(order)
        return order
    if kind == "laggard_last":
        return list(range(1, n)) + [0]
    if kind == "laggard_first":
        return [n - 1] + list(range(n - 1))
    raise AssertionError(f"unknown order kind {kind!r}")


def _run_round(
    *,
    order: list[int],
    n_learners: int,
    dtype: torch.dtype,
    token_weighted: bool,
    incremental_collect: bool,
) -> MergeResult:
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=max(1, n_learners - 1),
        grace_window_ms=500,
        token_weighted=token_weighted,
        clock=clock,
        incremental_collect=incremental_collect,
    )
    sync.start_round(round_id=23)
    for index in order:
        result = sync.submit(_fragment(index, round_id=23, dtype=dtype))
        assert result is None
        clock.advance_ms(1.0)
    clock.advance_ms(600.0)
    result = sync.tick()
    assert result is not None
    return result


def _assert_same_result(incremental: MergeResult, baseline: MergeResult) -> None:
    assert incremental.round_id == baseline.round_id
    assert incremental.learners_merged == baseline.learners_merged
    assert incremental.learners_excluded == baseline.learners_excluded
    assert incremental.learners_absent == baseline.learners_absent
    assert incremental.elapsed_grace_ms == baseline.elapsed_grace_ms
    assert incremental.reason == baseline.reason
    assert len(incremental.merged_delta) == len(baseline.merged_delta)
    for incremental_tensor, baseline_tensor in zip(
        incremental.merged_delta, baseline.merged_delta
    ):
        assert incremental_tensor.dtype == baseline_tensor.dtype
        assert incremental_tensor.shape == baseline_tensor.shape
        assert torch.equal(incremental_tensor, baseline_tensor)


@pytest.mark.parametrize(
    "order_kind",
    ["in_order", "reverse", "random", "laggard_last", "laggard_first"],
)
@pytest.mark.parametrize("n_learners", [2, 3, 4, 8])
@pytest.mark.parametrize("token_weighted", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_incremental_collect_matches_frozen_merge_order(
    order_kind: str,
    n_learners: int,
    token_weighted: bool,
    dtype: torch.dtype,
) -> None:
    order = _arrival_order(order_kind, n_learners)
    baseline = _run_round(
        order=order,
        n_learners=n_learners,
        dtype=dtype,
        token_weighted=token_weighted,
        incremental_collect=False,
    )
    incremental = _run_round(
        order=order,
        n_learners=n_learners,
        dtype=dtype,
        token_weighted=token_weighted,
        incremental_collect=True,
    )
    _assert_same_result(incremental, baseline)


def test_incremental_collect_duplicate_learner_falls_back_to_frozen_path() -> None:
    clock = FakeClock()
    sync = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=clock,
        incremental_collect=True,
    )
    sync.start_round(round_id=24)
    assert sync.submit(_fragment(0, round_id=24, dtype=torch.float32)) is None
    assert sync.submit(_fragment(1, round_id=24, dtype=torch.float32)) is None
    duplicate = LearnerFragment(
        learner_id=_learner_id(0),
        round_id=24,
        params_delta=[torch.tensor([99.0, 1.0, 0.0, -1.0], dtype=torch.float32)],
        tokens_consumed=101,
    )
    assert sync.submit(duplicate) is None
    clock.advance_ms(600)
    result = sync.tick()
    assert result is not None

    expected_clock = FakeClock()
    expected = GraceWindowSyncer(
        quorum_min_learners=2,
        grace_window_ms=500,
        token_weighted=True,
        clock=expected_clock,
    )
    expected.start_round(round_id=24)
    assert expected.submit(duplicate) is None
    assert expected.submit(_fragment(1, round_id=24, dtype=torch.float32)) is None
    expected_clock.advance_ms(600)
    expected_result = expected.tick()
    assert expected_result is not None
    _assert_same_result(result, expected_result)
