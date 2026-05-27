"""Concurrent outer-step orchestrator integration example.

Shows how to wire Phase 2 Week 1's ConcurrentOuterStep orchestrator into
a typical PyTorch training loop. The orchestrator overlaps the cross-rack
outer-step grace-window wait with inner-step forward / backward; the
training thread is never blocked on the merge.

In internal measurements the orchestrator recovers most of the cross-rack
grace-window wait as throughput at the canonical 2000ms grace_window_ms
default (see docs/benchmark_protocol.md for the measurement methodology).

This example uses a synthetic single-rank fragment provider and runs on
CPU. In a real multi-rack deployment, the fragment provider would pull
fragments from the cross-rack sideband (`tsugi_mend.sideband`) into the
asyncio queue.

Run:
    python examples/concurrent_orchestrator.py
"""
from __future__ import annotations

import asyncio
import time

import torch
import torch.nn as nn

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.reducer import LearnerFragment
from tsugi_mend.runtime import get_runtime


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(16, 64)
        self.down = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


def make_fragment_provider(rank_id: str, round_id: int, n_tokens: int):
    """Returns a callable that the orchestrator invokes on the asyncio
    thread to obtain an asyncio.Queue of incoming fragments.

    In a real multi-rack deployment this would pull fragments from the
    cross-rack sideband peer connections (`tsugi_mend.sideband`). For this
    example, we drip a single zero-delta fragment representing the local
    rank's contribution."""

    def provider() -> "asyncio.Queue[LearnerFragment]":
        queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()

        async def drip() -> None:
            frag = LearnerFragment(
                learner_id=rank_id,
                round_id=round_id,
                params_delta=[torch.zeros((1,), dtype=torch.float32)],
                tokens_consumed=n_tokens,
            )
            await queue.put(frag)

        asyncio.get_event_loop().create_task(drip())
        return queue

    return provider


def main() -> None:
    model = ToyModel()
    config = MendConfig(
        quorum_min_learners=1,
        grace_window_ms=20,        # short grace window for the example
        sync_period_steps=8,       # outer-step every 8 inner steps
        momentum_sync_period_steps=32,
        async_tp_enabled=False,
        sideband_peers=(),
        # Phase 2 Week 1: enable the orchestrator. Default True.
        concurrent_outer_step=True,
        # Optional: simulate a 100ms cross-rack wait per outer-round so
        # the orchestrator's overlap benefit is visible on this toy
        # single-process example. Default 0 in production.
        simulated_merge_delay_ms=100,
        diagnostics_dir="./results/concurrent_orchestrator_example",
    )
    mend_init(model, config, rank_id="example-orch/rank-0")
    runtime = get_runtime(model)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    n_steps = 32
    n_tokens_per_step = 16 * 4  # batch * sequence_length proxy

    outer_rounds = 0
    print(f"Running {n_steps} steps with concurrent_outer_step={config.concurrent_outer_step}, "
          f"simulated_merge_delay_ms={config.simulated_merge_delay_ms}")
    started = time.monotonic()

    for step in range(n_steps):
        runtime.step_begin(step)
        x = torch.randn(4, 16)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        # Phase 2 Week 1: exercise the orchestrator at sync boundaries.
        # The submit is non-blocking: the asyncio task absorbs the
        # simulated grace-window wait while we keep training.
        sched = runtime.schedule_for(step)
        if sched.should_sync_params and not runtime.outer_step_in_flight():
            runtime.outer_step_begin(
                round_id=step,
                fragment_provider=make_fragment_provider(
                    runtime.rank_id, step, n_tokens_per_step
                ),
            )
        # Non-blocking poll; merge result is delivered 1-3 inner steps later.
        result = runtime.outer_step_collect()
        if result is not None:
            outer_rounds += 1
            print(f"  step {step}: collected outer-round {result.round_id} "
                  f"after {result.elapsed_grace_ms:.1f}ms grace wait "
                  f"(merged {len(result.learners_merged)} learners)")

        runtime.step_end(step)

    # Drain any pending in-flight outer-rounds before shutdown so the
    # asyncio tasks complete cleanly (avoids "Event loop is closed"
    # warnings from torn-down asyncio.wait_for futures).
    drain_deadline = time.monotonic() + 2.0
    while time.monotonic() < drain_deadline:
        result = runtime.outer_step_collect()
        if result is not None:
            outer_rounds += 1
            print(f"  drain: collected outer-round {result.round_id}")
        if not runtime.outer_step_in_flight():
            break
        time.sleep(0.01)

    elapsed = time.monotonic() - started
    print(f"\nElapsed: {elapsed:.2f}s for {n_steps} steps, "
          f"{outer_rounds} outer-rounds collected.")
    print(f"With concurrent_outer_step=True and "
          f"simulated_merge_delay_ms={config.simulated_merge_delay_ms},")
    print(f"the orchestrator absorbed approximately "
          f"{outer_rounds * config.simulated_merge_delay_ms / 1000:.2f}s "
          f"of grace-window wait into inner-step compute time.")

    mend_shutdown(model)


if __name__ == "__main__":
    main()
