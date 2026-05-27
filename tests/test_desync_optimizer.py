"""DES-LOC desynchronized-momenta scheduler tests."""
from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from tsugi_mend.desync_optimizer import (
    DesynchronizedSyncSchedule,
    apply_moments,
    extract_moments,
)


def test_schedule_step_zero_is_sync_boundary():
    s = DesynchronizedSyncSchedule(sync_period_steps=4, momentum_sync_period_steps=8)
    d = s.tick(0)
    assert d.should_sync_params is True
    assert d.should_sync_momenta is True


def test_schedule_params_sync_every_N():
    s = DesynchronizedSyncSchedule(sync_period_steps=4, momentum_sync_period_steps=16)
    params_sync_steps = [step for step in range(33) if s.tick(step).should_sync_params]
    # Param sync at 0, 4, 8, 12, 16, 20, 24, 28, 32
    assert params_sync_steps == [0, 4, 8, 12, 16, 20, 24, 28, 32]


def test_schedule_momenta_sync_every_M():
    s = DesynchronizedSyncSchedule(sync_period_steps=4, momentum_sync_period_steps=16)
    momenta_sync_steps = [
        step for step in range(33) if s.tick(step).should_sync_momenta
    ]
    # Momenta sync at 0, 16, 32 only
    assert momenta_sync_steps == [0, 16, 32]


def test_schedule_momenta_sync_implies_params_sync():
    s = DesynchronizedSyncSchedule(sync_period_steps=4, momentum_sync_period_steps=16)
    for step in range(40):
        d = s.tick(step)
        if d.should_sync_momenta:
            assert d.should_sync_params, (
                f"step {step}: momenta sync without params sync violates DES-LOC"
            )


def test_schedule_rejects_M_lt_N():
    with pytest.raises(ValueError, match="DES-LOC requires M >= N"):
        DesynchronizedSyncSchedule(sync_period_steps=8, momentum_sync_period_steps=4)


def test_extract_moments_returns_empty_before_step():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = AdamW([p], lr=0.01)
    moments = extract_moments(opt)
    # AdamW state is created lazily on first step().
    assert moments == {}


def test_extract_and_apply_moments_roundtrip():
    # Build two identical mini-models.
    p1 = torch.nn.Parameter(torch.zeros(4))
    p2 = torch.nn.Parameter(torch.zeros(4))
    opt1 = AdamW([p1], lr=0.01)
    opt2 = AdamW([p2], lr=0.01)
    # Run one step on opt1 so its AdamW state populates.
    loss1 = (p1 * 2).sum()
    loss1.backward()
    opt1.step()
    moments_from_1 = extract_moments(opt1)
    assert id(p1) in moments_from_1
    assert "exp_avg" in moments_from_1[id(p1)]
    assert "exp_avg_sq" in moments_from_1[id(p1)]
    # Run a step on opt2 so its state exists with different values.
    loss2 = (p2 * 5).sum()
    loss2.backward()
    opt2.step()
    moments_from_2_before = extract_moments(opt2)
    assert not torch.allclose(
        moments_from_2_before[id(p2)]["exp_avg"],
        moments_from_1[id(p1)]["exp_avg"],
    )
    # Now copy opt1's moments onto opt2's tensors by id-of-param. Since
    # the two optimizers have different params, key the apply by p2 id
    # using a fresh dict that maps id(p2) to the same tensors.
    transplant = {id(p2): moments_from_1[id(p1)]}
    updated = apply_moments(opt2, transplant)
    assert updated == 2  # exp_avg + exp_avg_sq
    moments_from_2_after = extract_moments(opt2)
    assert torch.allclose(
        moments_from_2_after[id(p2)]["exp_avg"],
        moments_from_1[id(p1)]["exp_avg"],
    )
