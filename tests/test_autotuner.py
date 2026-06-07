"""RuntimeAutotuner control-law tests."""
from __future__ import annotations

from tsugi_mend.autotuner import RuntimeAutotuner


def _drive_stream(stream_ms: list[float]):
    tuner = RuntimeAutotuner(
        base_failslow_zscore_threshold=2.0,
        failslow_zscore_min=1.5,
        failslow_zscore_max=5.0,
        base_grace_window_ms=10,
        grace_window_max_ms=200,
        window_steps=8,
        min_samples=4,
    )
    decisions = []
    for step_time_ms, z_score, is_slow in stream_ms:
        decisions.append(
            tuner.observe(
                rank_id="rank-a",
                step_time_ms=step_time_ms,
                detector_z_score=z_score,
                detector_flagged_slow=is_slow,
            )
        )
    return decisions


def test_runtime_autotuner_is_deterministic_for_fixed_stream():
    stream = [
        (100.0, 0.0, False),
        (101.0, 0.0, False),
        (99.0, 0.0, False),
        (100.0, 0.0, False),
        (160.0, 2.5, True),
        (162.0, 2.6, True),
        (100.0, 0.0, False),
        (100.0, 0.0, False),
    ]
    first = _drive_stream(stream)
    second = _drive_stream(stream)

    assert first == second


def test_runtime_autotuner_adapts_threshold_and_grace_in_expected_directions():
    decisions = _drive_stream(
        [
            (100.0, 0.0, False),
            (100.0, 0.0, False),
            (100.0, 0.0, False),
            (100.0, 0.0, False),
            (160.0, 2.5, True),
            (165.0, 2.7, True),
            (100.0, 0.0, False),
            (100.0, 0.0, False),
        ]
    )

    first_detection_step = decisions[3]
    first_slow_step = decisions[4]
    second_slow_step = decisions[5]
    clean_recovery_step = decisions[-1]

    assert first_detection_step.reason == "clean_observation"
    assert first_detection_step.effective_failslow_zscore_threshold == 2.0
    assert first_detection_step.effective_grace_window_ms == 10

    assert first_slow_step.reason == "slow_observation"
    assert first_slow_step.threshold_action == "increase"
    assert first_slow_step.grace_action == "increase"
    assert first_slow_step.effective_failslow_zscore_threshold > 2.0
    assert first_slow_step.effective_grace_window_ms > 10

    assert second_slow_step.effective_grace_window_ms >= (
        first_slow_step.effective_grace_window_ms
    )
    assert clean_recovery_step.grace_action == "decrease"
    assert clean_recovery_step.effective_grace_window_ms < (
        second_slow_step.effective_grace_window_ms
    )
