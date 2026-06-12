"""Property tests for GraceWindowSyncer roster and diagnostic invariants."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from hypothesis import given, settings, strategies as st

from tsugi_mend.reducer import GraceWindowSyncer, LearnerFragment, MergeResult


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _learner_id(index: int) -> str:
    return f"learner-{index}"


def _fragment(index: int, round_id: int) -> LearnerFragment:
    value = float(index + 1)
    return LearnerFragment(
        learner_id=_learner_id(index),
        round_id=round_id,
        params_delta=[
            torch.tensor([value, -value, value / 3.0], dtype=torch.float32),
        ],
        tokens_consumed=10 + index,
    )


def _run_syncer(
    *,
    n_learners: int,
    quorum: int,
    submit_order: Sequence[int],
    clock_advances_ms: Sequence[int],
    token_weighted: bool,
    expected_learner_ids: set[str] | None,
    excluded_before_submit: set[int] | None = None,
) -> MergeResult:
    clock = FakeClock()
    grace_window_ms = 1_000
    sync = GraceWindowSyncer(
        quorum_min_learners=quorum,
        grace_window_ms=grace_window_ms,
        token_weighted=token_weighted,
        clock=clock,
    )
    sync.start_round(
        round_id=17,
        expected_learner_ids=expected_learner_ids,
        total_learners=n_learners if expected_learner_ids is None else None,
    )
    for index in sorted(excluded_before_submit or set()):
        sync.mark_failslow(_learner_id(index))

    for index, advance_ms in zip(submit_order, clock_advances_ms, strict=True):
        clock.advance_ms(float(advance_ms))
        result = sync.submit(_fragment(index, round_id=17))
        if result is not None:
            return result

    clock.advance_ms(float(grace_window_ms + 1))
    result = sync.tick()
    if result is None:
        result = sync.finalize_on_timeout()
    return result


def _assert_same_merge(left: MergeResult, right: MergeResult) -> None:
    assert left.learners_merged == right.learners_merged
    assert len(left.merged_delta) == len(right.merged_delta)
    for left_tensor, right_tensor in zip(left.merged_delta, right.merged_delta):
        assert torch.equal(left_tensor, right_tensor)


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    data=st.data(),
    n_learners=st.integers(min_value=1, max_value=6),
    token_weighted=st.booleans(),
)
def test_fuzz_exhaustive_roster_is_bit_identical_to_none(
    data, n_learners: int, token_weighted: bool
) -> None:
    learner_indices = list(range(n_learners))
    submit_order = list(data.draw(st.permutations(learner_indices), label="submit_order"))
    advances = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=n_learners,
            max_size=n_learners,
        ),
        label="advances",
    )
    quorum = data.draw(st.integers(min_value=1, max_value=n_learners), label="quorum")
    expected_ids = {_learner_id(index) for index in learner_indices}

    roster_result = _run_syncer(
        n_learners=n_learners,
        quorum=quorum,
        submit_order=submit_order,
        clock_advances_ms=advances,
        token_weighted=token_weighted,
        expected_learner_ids=expected_ids,
    )
    none_result = _run_syncer(
        n_learners=n_learners,
        quorum=quorum,
        submit_order=submit_order,
        clock_advances_ms=advances,
        token_weighted=token_weighted,
        expected_learner_ids=None,
    )

    _assert_same_merge(roster_result, none_result)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    data=st.data(),
    n_learners=st.integers(min_value=1, max_value=6),
    roster_active=st.booleans(),
    token_weighted=st.booleans(),
)
def test_fuzz_liveness_and_subset_diagnostics(
    data, n_learners: int, roster_active: bool, token_weighted: bool
) -> None:
    learner_indices = list(range(n_learners))
    excluded = set(
        data.draw(
            st.lists(
                st.sampled_from(learner_indices),
                unique=True,
                min_size=0,
                max_size=max(0, n_learners - 1),
            ),
            label="excluded",
        )
    )
    healthy = [index for index in learner_indices if index not in excluded]
    quorum = data.draw(st.integers(min_value=1, max_value=len(healthy)), label="quorum")
    submitters = data.draw(
        st.lists(
            st.sampled_from(healthy),
            unique=True,
            min_size=quorum,
            max_size=len(healthy),
        ),
        label="submitters",
    )
    submit_order = list(data.draw(st.permutations(submitters), label="submit_order"))
    advances = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=len(submit_order),
            max_size=len(submit_order),
        ),
        label="advances",
    )
    expected_ids = (
        {_learner_id(index) for index in learner_indices} if roster_active else None
    )

    result = _run_syncer(
        n_learners=n_learners,
        quorum=quorum,
        submit_order=submit_order,
        clock_advances_ms=advances,
        token_weighted=token_weighted,
        expected_learner_ids=expected_ids,
        excluded_before_submit=excluded,
    )

    submitted_ids = {_learner_id(index) for index in submitters}
    merged_ids = set(result.learners_merged)
    excluded_ids = set(result.learners_excluded)
    absent_ids = set(result.learners_absent)

    assert merged_ids.issubset(submitted_ids)
    assert absent_ids.isdisjoint(merged_ids)
    assert absent_ids.isdisjoint(excluded_ids)
    assert excluded_ids.isdisjoint(merged_ids)
    if expected_ids is None:
        assert result.learners_absent == []
