"""DES-LOC desynchronized-momenta optimizer wrapper.

Reference: Iacob et al., "DES-LOC: Local Adam with Desynchronized
Synchronization Periods", arXiv:2505.22549 (May 2025; ICLR 2026).

Core idea: in the local-Adam family, parameters are synced every N inner
steps but adaptive-optimizer momenta (first and second moments) can be
synced LESS frequently than parameters without hurting convergence. The
paper reports 170x less communication than DDP and 2x less than prior
Local Adam, with 1.3x-2.1x wall-clock speedup on 100Gb/s links for 1B-13B
models.

This module implements the bookkeeping (when to sync params vs momenta);
the actual cross-rack reduction goes through `reducer.GraceWindowSyncer`
under the hood.

Patent-independence note: DES-LOC is published in May 2025 / ICLR 2026
and is unrelated to TsugiCinema's K-Pool LoRA or Infinity provisional
claims. This implementation is a straight wrapper over `torch.optim`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer


@dataclass
class SyncSchedule:
    """Decision schedule emitted by DesynchronizedSyncSchedule.tick().

    Attributes:
        should_sync_params: True if this step is a param-sync boundary.
        should_sync_momenta: True if this step is a momentum-sync
            boundary. Momentum sync implies param sync as well per the
            DES-LOC paper (the cheap operation is a strict subset of the
            expensive one's cadence).
    """
    should_sync_params: bool
    should_sync_momenta: bool


class DesynchronizedSyncSchedule:
    """Pure-Python scheduler that emits sync decisions at each inner step.

    DES-LOC semantics:
        - Every `sync_period_steps` inner steps: sync parameters.
        - Every `momentum_sync_period_steps` inner steps: sync params
          AND momenta. (M is constrained M >= N at config validation.)
        - Step 0 is treated as a sync boundary so the first round
          starts coherent.

    Usage:
        sched = DesynchronizedSyncSchedule(N=128, M=512)
        for step in range(num_steps):
            decision = sched.tick(step)
            # ... run inner step ...
            if decision.should_sync_momenta:
                reduce_first_and_second_moments(...)
            if decision.should_sync_params:
                reduce_params(...)
    """

    def __init__(self, sync_period_steps: int, momentum_sync_period_steps: int) -> None:
        if sync_period_steps < 1:
            raise ValueError(
                f"sync_period_steps must be >= 1; got {sync_period_steps}"
            )
        if momentum_sync_period_steps < sync_period_steps:
            raise ValueError(
                f"momentum_sync_period_steps ({momentum_sync_period_steps}) must be "
                f">= sync_period_steps ({sync_period_steps}); DES-LOC requires M >= N"
            )
        self.N = sync_period_steps
        self.M = momentum_sync_period_steps

    def tick(self, step: int) -> SyncSchedule:
        # Decoupled DiLoCo + DES-LOC convention: step 0 is the initial
        # sync boundary. Then every N steps after that, params sync;
        # every M steps after that, momenta sync (and implicitly params).
        sync_params = (step % self.N == 0)
        sync_momenta = (step % self.M == 0)
        return SyncSchedule(
            should_sync_params=sync_params or sync_momenta,
            should_sync_momenta=sync_momenta,
        )

    def update_sync_period(self, new_N: int) -> None:
        """Update the parameter sync cadence in place.

        Used by the runtime auto-tuner to lower N after a warmup window
        measures the true per-step compute time. The momentum sync
        cadence M is preserved; the auto-tuner's lower bound is set so
        that M >= N remains true.
        """
        if new_N < 1:
            raise ValueError(f"new_N must be >= 1; got {new_N}")
        if new_N > self.M:
            raise ValueError(
                f"new_N ({new_N}) cannot exceed M ({self.M}); DES-LOC "
                f"requires the param-sync cadence to be at most the "
                f"momentum-sync cadence"
            )
        self.N = new_N


def extract_moments(optimizer: Optimizer) -> dict[int, dict[str, torch.Tensor]]:
    """Best-effort extraction of first / second moments from a
    torch.optim.AdamW-shaped optimizer state.

    Returns a mapping of parameter `id(param)` -> {"exp_avg": tensor,
    "exp_avg_sq": tensor}. Parameters without state (have not been
    stepped yet, or different optimizer family) are skipped.

    This is the inspection surface for the cross-rack momentum reduce.
    The actual reduce uses `reducer.token_weighted_merge` on the lists
    of moments collected from each rack.
    """
    out: dict[int, dict[str, torch.Tensor]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, None)
            if state is None:
                continue
            extracted: dict[str, torch.Tensor] = {}
            if "exp_avg" in state and isinstance(state["exp_avg"], torch.Tensor):
                extracted["exp_avg"] = state["exp_avg"]
            if "exp_avg_sq" in state and isinstance(state["exp_avg_sq"], torch.Tensor):
                extracted["exp_avg_sq"] = state["exp_avg_sq"]
            if extracted:
                out[id(p)] = extracted
    return out


def apply_moments(
    optimizer: Optimizer,
    moments: dict[int, dict[str, torch.Tensor]],
) -> int:
    """Copy averaged moments back into the optimizer state. Returns the
    number of parameters whose state was updated."""
    updated = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, None)
            if state is None:
                continue
            new = moments.get(id(p), None)
            if not new:
                continue
            for k, v in new.items():
                if k in state and isinstance(state[k], torch.Tensor):
                    state[k].copy_(v)
                    updated += 1
    return updated
