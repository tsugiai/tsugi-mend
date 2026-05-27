"""Minimal single-process integration example.

Shows the smallest end-to-end use of tsugiai-mend-sdk on a toy nn.Module.
Equivalent to the Stage A `test_mend_init_and_shutdown_on_toy_model` test,
exposed as a runnable script.

This example does NOT exercise the cross-rack reducer (single rank); it
runs entirely on CPU. Exercising the reducer requires a real multi-node
cross-rack deployment (see docs/benchmark_protocol.md).

Run:
    python examples/minimal_single_process.py
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.runtime import get_runtime


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


def main() -> None:
    model = ToyModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=0,
        sync_period_steps=4,
        momentum_sync_period_steps=16,
        async_tp_enabled=False,
        sideband_peers=(),
        diagnostics_dir="./results/minimal_example_diag",
    )
    mend_init(model, config, rank_id="example/rank-0")
    runtime = get_runtime(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"topology: {runtime.topology.detection_method}, "
          f"n_racks={runtime.topology.n_racks()}")

    for step in range(32):
        runtime.step_begin(step)
        x = torch.randn(4, 16)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        runtime.step_end(step)

        sched = runtime.schedule_for(step)
        if sched.should_sync_params or sched.should_sync_momenta:
            tag = "MOMENTA" if sched.should_sync_momenta else "PARAMS"
            print(f"  step {step}: {tag} sync boundary; loss={loss.item():.4f}")

        time.sleep(0.01)

    mend_shutdown(model)
    print("Done. Diagnostics at ./results/minimal_example_diag/")


if __name__ == "__main__":
    main()
