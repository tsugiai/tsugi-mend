"""Two-rank torchrun example for tsugi-mend.

Run from the repository root after installing the package:

    torchrun --standalone --nproc-per-node=2 examples/torchrun_two_rank.py

The example uses the gloo backend so it runs on a CPU-only laptop. It starts two
local sideband endpoints, wraps a tiny model in DistributedDataParallel, calls
mend_init after wrapping, trains for a few steps, and submits gathered learner
fragments to the reducer at sync boundaries.
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.reducer import LearnerFragment
from tsugi_mend.runtime import get_runtime


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _prefix(rank: int) -> str:
    return f"[rank {rank}]"


def _snapshot_params(model: DDP) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.module.parameters()]


def _build_fragment(
    model: DDP,
    previous_params: Sequence[torch.Tensor],
    rank: int,
    round_id: int,
    tokens_consumed: int,
) -> LearnerFragment:
    deltas = [
        current.detach().cpu() - previous
        for current, previous in zip(model.module.parameters(), previous_params)
    ]
    return LearnerFragment(
        learner_id=f"local-node/rank-{rank}",
        round_id=round_id,
        params_delta=deltas,
        tokens_consumed=tokens_consumed,
    )


def _provider_from_fragments(
    fragments: Sequence[LearnerFragment],
) -> Callable[[], "asyncio.Queue[LearnerFragment]"]:
    def provider() -> "asyncio.Queue[LearnerFragment]":
        queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()
        for fragment in fragments:
            queue.put_nowait(fragment)
        return queue

    return provider


def _wait_for_sideband_peer(
    runtime: Any,
    expected_rank_id: str,
    timeout_s: float = 5.0,
) -> None:
    if runtime.sideband is None:
        raise RuntimeError("sideband was not started")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if expected_rank_id in runtime.sideband.peer_snapshot():
            return
        time.sleep(0.05)
    raise TimeoutError(f"sideband did not observe peer {expected_rank_id!r}")


def _collect_outer_step(runtime: Any, timeout_s: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = runtime.outer_step_collect()
        if result is not None:
            return result
        time.sleep(0.005)
    raise TimeoutError("outer-step reducer did not complete")


def _merged_norm(tensors: Sequence[torch.Tensor]) -> float:
    total = torch.tensor(0.0)
    for tensor in tensors:
        total = total + tensor.float().pow(2).sum()
    return float(total.sqrt().item())


def main() -> None:
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _env_int("LOCAL_RANK", rank)

    if world_size != 2:
        raise RuntimeError(
            "examples/torchrun_two_rank.py expects exactly 2 ranks; "
            f"got WORLD_SIZE={world_size}"
        )

    sideband_port_base = _env_int("MEND_SIDEBAND_PORT_BASE", 51900)
    local_sideband_port = sideband_port_base + rank
    peer_rank = 1 - rank
    peer_sideband_port = sideband_port_base + peer_rank
    diagnostics_root = os.environ.get(
        "MEND_DIAGNOSTICS_DIR",
        "./results/torchrun_two_rank",
    )

    torch.manual_seed(20240527)
    model = DDP(TinyModel())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    config = MendConfig(
        quorum_min_learners=world_size,
        grace_window_ms=20,
        sync_period_steps=4,
        momentum_sync_period_steps=8,
        async_tp_enabled=False,
        concurrent_outer_step=True,
        sideband_addr=f"tcp://127.0.0.1:{local_sideband_port}",
        sideband_peers=(f"tcp://127.0.0.1:{peer_sideband_port}",),
        sideband_heartbeat_ms=50,
        sideband_connect_timeout_s=0.1,
        diagnostics_dir=f"{diagnostics_root}/rank{rank}",
    )

    mend_started = False
    try:
        mend_init(model, config, rank_id=f"local-node/rank-{rank}")
        mend_started = True
        runtime = get_runtime(model)

        dist.barrier()
        _wait_for_sideband_peer(runtime, f"local-node/rank-{peer_rank}")
        print(
            f"{_prefix(rank)} sideband peer local-node/rank-{peer_rank} observed "
            f"on port {peer_sideband_port}",
            flush=True,
        )

        previous_params = _snapshot_params(model)
        batch = 4
        features = 8
        steps = 8

        for step in range(1, steps + 1):
            runtime.step_begin(step)
            torch.manual_seed(1000 + step + local_rank)
            x = torch.randn(batch, features)
            target = torch.full((batch, 4), fill_value=float(rank))
            loss = (model(x) - target).pow(2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            runtime.step_end(step)

            if step % 2 == 0:
                print(
                    f"{_prefix(rank)} step {step}: loss={loss.item():.4f}",
                    flush=True,
                )

            schedule = runtime.schedule_for(step)
            if schedule.should_sync_params:
                local_fragment = _build_fragment(
                    model=model,
                    previous_params=previous_params,
                    rank=rank,
                    round_id=step,
                    tokens_consumed=batch * features,
                )
                gathered: list[LearnerFragment | None] = [None] * world_size
                dist.all_gather_object(gathered, local_fragment)
                fragments = [fragment for fragment in gathered if fragment is not None]

                runtime.outer_step_begin(
                    round_id=step,
                    fragment_provider=_provider_from_fragments(fragments),
                )
                result = _collect_outer_step(runtime)
                print(
                    f"{_prefix(rank)} step {step}: outer round merged "
                    f"{len(result.learners_merged)} learners, "
                    f"norm={_merged_norm(result.merged_delta):.6f}, "
                    f"reason={result.reason}",
                    flush=True,
                )
                previous_params = _snapshot_params(model)

        dist.barrier()
        print(
            f"{_prefix(rank)} done; diagnostics at {diagnostics_root}/rank{rank}/",
            flush=True,
        )
    finally:
        if mend_started:
            mend_shutdown(model)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
