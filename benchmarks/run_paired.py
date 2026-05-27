"""Run a paired baseline-vs-SDK tsugi-mend benchmark.

Cheap local reproduction cell:

    python -m torch.distributed.run \
        --nproc-per-node=2 \
        --master-addr 127.0.0.1 \
        --master-port 29541 \
        --local-addr 127.0.0.1 \
        benchmarks/run_paired.py \
        --backend gloo \
        --ranks 2 \
        --steps 24 \
        --warmup-steps 4 \
        --output benchmarks/results/cpu_gloo_toy/result.json

The cell uses a deterministic CPU toy workload, local gloo ranks, and no
paid compute. Rank 0 writes a JSON results bundle containing config,
per-step metrics, an exact loss-equivalence check, and a bootstrap 95%
confidence interval for the paired tokens/s uplift.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsugi_mend import MendConfig, mend_init, mend_shutdown  # noqa: E402
from tsugi_mend.reducer import LearnerFragment  # noqa: E402
from tsugi_mend.runtime import get_runtime  # noqa: E402


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    """Config values that define the paired workload."""

    model: str
    steps: int
    warmup_steps: int
    backend: str
    ranks: int
    seed: int
    batch_size: int
    seq_len: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    learning_rate: float
    sync_period_steps: int
    momentum_sync_period_steps: int
    grace_window_ms: int
    simulated_merge_delay_ms: int
    bootstrap_resamples: int
    bootstrap_seed: int
    hardware_label: str
    run_label: str

    @property
    def local_tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def global_tokens_per_step(self) -> int:
        return self.local_tokens_per_step * self.ranks


@dataclass(frozen=True)
class StepMetric:
    step: int
    loss: float
    step_time_s: float
    tokens: int

    @property
    def step_time_ms(self) -> float:
        return self.step_time_s * 1000.0

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.step_time_s


@dataclass(frozen=True)
class BitExactCheck:
    passed: bool
    checked_steps: int
    max_abs_loss_delta: float
    mismatches: list[dict[str, float | int]]


class ToyMLP(nn.Module):
    """Small deterministic CPU model for the $0 gloo cell."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


def percentile(values: Sequence[float], q: float) -> float:
    """Return the linearly interpolated percentile for q in [0, 1]."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def uplift_pct(baseline_tokens_per_s: float, sdk_tokens_per_s: float) -> float:
    """Return percentage uplift from baseline to SDK tokens/s."""

    if baseline_tokens_per_s <= 0:
        raise ValueError("baseline_tokens_per_s must be positive")
    return (sdk_tokens_per_s / baseline_tokens_per_s - 1.0) * 100.0


def bootstrap_uplift_ci(
    baseline_tokens_per_s: Sequence[float],
    sdk_tokens_per_s: Sequence[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a paired 95% CI for the ratio-of-means uplift."""

    if len(baseline_tokens_per_s) != len(sdk_tokens_per_s):
        raise ValueError("baseline and sdk samples must have the same length")
    if not baseline_tokens_per_s:
        raise ValueError("bootstrap requires at least one paired sample")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")

    rng = random.Random(seed)
    n = len(baseline_tokens_per_s)
    estimates: list[float] = []
    for _ in range(resamples):
        baseline_sum = 0.0
        sdk_sum = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            baseline_sum += baseline_tokens_per_s[idx]
            sdk_sum += sdk_tokens_per_s[idx]
        estimates.append(uplift_pct(baseline_sum / n, sdk_sum / n))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_steps(steps: Sequence[StepMetric], warmup_steps: int) -> dict[str, float | int]:
    """Summarize steady-state tokens/s and rank-0 step-time percentiles."""

    steady = [step for step in steps if step.step >= warmup_steps]
    if not steady:
        raise ValueError("no steady-state samples; lower warmup_steps or add steps")
    tps = [step.tokens_per_s for step in steady]
    step_ms = [step.step_time_ms for step in steady]
    return {
        "steady_state_steps": len(steady),
        "mean_tokens_per_s": statistics.fmean(tps),
        "p50_step_ms": percentile(step_ms, 0.50),
        "p95_step_ms": percentile(step_ms, 0.95),
        "p99_step_ms": percentile(step_ms, 0.99),
    }


def check_bit_exact_losses(
    baseline_losses: Sequence[float],
    sdk_losses: Sequence[float],
) -> BitExactCheck:
    """Check exact equality of the rank-0 loss scalar on every paired step."""

    if len(baseline_losses) != len(sdk_losses):
        raise ValueError("baseline and sdk loss lists must have the same length")
    mismatches: list[dict[str, float | int]] = []
    max_abs_delta = 0.0
    for step, (baseline_loss, sdk_loss) in enumerate(zip(baseline_losses, sdk_losses)):
        delta = abs(baseline_loss - sdk_loss)
        max_abs_delta = max(max_abs_delta, delta)
        if baseline_loss != sdk_loss:
            mismatches.append(
                {
                    "step": step,
                    "baseline_loss": baseline_loss,
                    "sdk_loss": sdk_loss,
                    "abs_delta": delta,
                }
            )
    return BitExactCheck(
        passed=not mismatches,
        checked_steps=len(baseline_losses),
        max_abs_loss_delta=max_abs_delta,
        mismatches=mismatches[:10],
    )


def init_distributed(backend: str, expected_ranks: int | None) -> tuple[int, int]:
    """Initialize torch.distributed when launched under torchrun."""

    if "WORLD_SIZE" not in os.environ:
        if expected_ranks not in (None, 1):
            raise ValueError(
                f"--ranks {expected_ranks} requires torchrun; "
                "run with python -m torch.distributed.run"
            )
        return 0, 1

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if expected_ranks is not None and expected_ranks != world_size:
        raise ValueError(
            f"--ranks={expected_ranks} does not match torchrun WORLD_SIZE={world_size}"
        )
    return rank, world_size


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def make_model(config: BenchmarkConfig) -> ToyMLP:
    if config.model != "toy-mlp":
        raise ValueError(f"unsupported model {config.model!r}; only 'toy-mlp' is available")
    torch.manual_seed(config.seed)
    return ToyMLP(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    )


def make_batch(
    *,
    config: BenchmarkConfig,
    rank: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the same deterministic per-rank batch for both modes."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 100_000 * step + rank)
    n_rows = config.batch_size * config.seq_len
    x = torch.randn(n_rows, config.input_dim, generator=generator)
    target = torch.randn(n_rows, config.output_dim, generator=generator)
    return x, target


def average_gradients(model: nn.Module, world_size: int) -> None:
    if world_size == 1:
        return
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def make_zero_fragments(
    model: nn.Module,
    *,
    round_id: int,
    world_size: int,
    tokens_per_rank: int,
) -> list[LearnerFragment]:
    """Build synthetic zero-delta fragments for the local CPU cell.

    The cheap cell exercises the reducer/orchestrator control path without
    altering the optimizer trajectory, which keeps the loss-equivalence
    assertion load-bearing and exact.
    """

    base_delta = [torch.zeros_like(param.detach()) for param in model.parameters()]
    fragments = []
    for learner_idx in range(world_size):
        fragments.append(
            LearnerFragment(
                learner_id=f"local-rank-{learner_idx}",
                round_id=round_id,
                params_delta=[delta.clone() for delta in base_delta],
                tokens_consumed=tokens_per_rank,
            )
        )
    return fragments


def provider_from_fragments(
    fragments: Sequence[LearnerFragment],
) -> "asyncio.Queue[LearnerFragment]":
    queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()

    async def drip() -> None:
        for fragment in fragments:
            await queue.put(fragment)

    asyncio.get_event_loop().create_task(drip())
    return queue


def maybe_start_sdk_outer_round(model: nn.Module, config: BenchmarkConfig, step: int) -> None:
    runtime = get_runtime(model)
    if runtime.outer_step_in_flight():
        return
    fragments = make_zero_fragments(
        model,
        round_id=step,
        world_size=config.ranks,
        tokens_per_rank=config.local_tokens_per_step,
    )
    runtime.outer_step_begin(
        round_id=step,
        fragment_provider=lambda: provider_from_fragments(fragments),
    )


def collect_sdk_outer_round(model: nn.Module) -> None:
    runtime = get_runtime(model)
    runtime.outer_step_collect()


def drain_sdk_outer_round(model: nn.Module) -> float:
    runtime = get_runtime(model)
    started = time.perf_counter()
    deadline = time.monotonic() + 30.0
    while runtime.outer_step_in_flight():
        runtime.outer_step_collect()
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out draining SDK outer round")
        time.sleep(0.001)
    runtime.outer_step_collect()
    return (time.perf_counter() - started) * 1000.0


def run_mode(
    *,
    mode: str,
    config: BenchmarkConfig,
    rank: int,
    world_size: int,
) -> tuple[list[StepMetric], float]:
    """Run one side of the paired benchmark and return rank-local metrics."""

    torch.manual_seed(config.seed)
    model = make_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    metrics: list[StepMetric] = []
    final_drain_ms = 0.0

    if mode == "sdk":
        mend_config = MendConfig(
            quorum_min_learners=world_size,
            grace_window_ms=config.grace_window_ms,
            sync_period_steps=config.sync_period_steps,
            momentum_sync_period_steps=config.momentum_sync_period_steps,
            async_tp_enabled=False,
            concurrent_outer_step=True,
            sideband_peers=(),
            simulated_merge_delay_ms=config.simulated_merge_delay_ms,
            diagnostics_dir=None,
        )
        mend_init(model, mend_config, rank_id=f"benchmark/rank-{rank}")

    try:
        for step in range(config.steps):
            x, target = make_batch(config=config, rank=rank, step=step)
            if mode == "sdk":
                runtime = get_runtime(model)
                runtime.step_begin(step)

            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = F.mse_loss(pred, target)
            loss.backward()
            average_gradients(model, world_size)
            optimizer.step()

            if mode == "sdk":
                runtime = get_runtime(model)
                schedule = runtime.schedule_for(step)
                if schedule.should_sync_params:
                    maybe_start_sdk_outer_round(model, config, step)
                collect_sdk_outer_round(model)
                runtime.step_end(step)

            elapsed_s = time.perf_counter() - started
            metrics.append(
                StepMetric(
                    step=step,
                    loss=float(loss.item()),
                    step_time_s=elapsed_s,
                    tokens=config.global_tokens_per_step,
                )
            )
    finally:
        if mode == "sdk":
            final_drain_ms = drain_sdk_outer_round(model)
            mend_shutdown(model)

    return metrics, final_drain_ms


def paired_step_rows(
    baseline: Sequence[StepMetric],
    sdk: Sequence[StepMetric],
    *,
    warmup_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    for baseline_step, sdk_step in zip(baseline, sdk):
        rows.append(
            {
                "step": baseline_step.step,
                "steady_state": baseline_step.step >= warmup_steps,
                "tokens": baseline_step.tokens,
                "baseline_loss": baseline_step.loss,
                "sdk_loss": sdk_step.loss,
                "baseline_step_ms": baseline_step.step_time_ms,
                "sdk_step_ms": sdk_step.step_time_ms,
                "baseline_tokens_per_s": baseline_step.tokens_per_s,
                "sdk_tokens_per_s": sdk_step.tokens_per_s,
            }
        )
    return rows


def build_result_bundle(
    *,
    config: BenchmarkConfig,
    baseline: Sequence[StepMetric],
    sdk: Sequence[StepMetric],
    sdk_final_drain_ms: float,
) -> dict[str, Any]:
    baseline_summary = summarize_steps(baseline, config.warmup_steps)
    sdk_summary = summarize_steps(sdk, config.warmup_steps)
    baseline_steady_tps = [
        step.tokens_per_s for step in baseline if step.step >= config.warmup_steps
    ]
    sdk_steady_tps = [step.tokens_per_s for step in sdk if step.step >= config.warmup_steps]
    ci_low, ci_high = bootstrap_uplift_ci(
        baseline_steady_tps,
        sdk_steady_tps,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    uplift = uplift_pct(
        baseline_summary["mean_tokens_per_s"],
        sdk_summary["mean_tokens_per_s"],
    )
    bit_exact = check_bit_exact_losses(
        [step.loss for step in baseline],
        [step.loss for step in sdk],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": config.run_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "docs/benchmark_protocol.md",
        "hardware": {
            "label": config.hardware_label,
            "provider": "local",
            "nodes": 1,
            "accelerator": "none",
            "interconnect": "local gloo over loopback",
            "public_cost_usd": 0.0,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "config": asdict(config),
        "bit_exact_loss": asdict(bit_exact),
        "summary": {
            "baseline": baseline_summary,
            "sdk": sdk_summary,
            "uplift_pct": uplift,
            "bootstrap_95_ci_pct": [ci_low, ci_high],
            "sdk_final_drain_ms": sdk_final_drain_ms,
        },
        "per_step": paired_step_rows(
            baseline,
            sdk,
            warmup_steps=config.warmup_steps,
        ),
        "notes": [
            "Cheap $0 CPU/gloo cell; this is not a real cross-rack production run.",
            "Synthetic zero-delta outer fragments exercise the control path without changing the optimizer trajectory.",
            "Do not compare this local toy result to published multi-node headline cells.",
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="toy-mlp")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--sync-period-steps", type=int, default=4)
    parser.add_argument("--momentum-sync-period-steps", type=int, default=8)
    parser.add_argument("--grace-window-ms", type=int, default=0)
    parser.add_argument("--simulated-merge-delay-ms", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260527)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--hardware-label", default="local_cpu_gloo")
    parser.add_argument("--run-label", default="cpu_gloo_toy")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/cpu_gloo_toy/result.json"),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "steps",
        "batch_size",
        "seq_len",
        "input_dim",
        "hidden_dim",
        "output_dim",
        "sync_period_steps",
        "momentum_sync_period_steps",
        "bootstrap_resamples",
        "torch_threads",
    ]
    for field in positive_int_fields:
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be >= 1")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be less than --steps")
    if args.grace_window_ms < 0:
        raise ValueError("--grace-window-ms must be >= 0")
    if args.simulated_merge_delay_ms < 0:
        raise ValueError("--simulated-merge-delay-ms must be >= 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.momentum_sync_period_steps < args.sync_period_steps:
        raise ValueError("--momentum-sync-period-steps must be >= --sync-period-steps")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)

    rank, world_size = init_distributed(args.backend, args.ranks)
    config = BenchmarkConfig(
        model=args.model,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        backend=args.backend,
        ranks=world_size,
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        learning_rate=args.learning_rate,
        sync_period_steps=args.sync_period_steps,
        momentum_sync_period_steps=args.momentum_sync_period_steps,
        grace_window_ms=args.grace_window_ms,
        simulated_merge_delay_ms=args.simulated_merge_delay_ms,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        hardware_label=args.hardware_label,
        run_label=args.run_label,
    )

    try:
        baseline, _ = run_mode(
            mode="baseline",
            config=config,
            rank=rank,
            world_size=world_size,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        sdk, sdk_final_drain_ms = run_mode(
            mode="sdk",
            config=config,
            rank=rank,
            world_size=world_size,
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            bundle = build_result_bundle(
                config=config,
                baseline=baseline,
                sdk=sdk,
                sdk_final_drain_ms=sdk_final_drain_ms,
            )
            if not bundle["bit_exact_loss"]["passed"]:
                raise AssertionError(
                    "baseline and SDK losses are not bit-exact: "
                    f"{bundle['bit_exact_loss']['mismatches']}"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
            summary = bundle["summary"]
            print(f"wrote {args.output}")
            print(
                "bit_exact=PASS "
                f"baseline_tps={summary['baseline']['mean_tokens_per_s']:.2f} "
                f"sdk_tps={summary['sdk']['mean_tokens_per_s']:.2f} "
                f"uplift_pct={summary['uplift_pct']:.2f} "
                "ci95_pct="
                f"[{summary['bootstrap_95_ci_pct'][0]:.2f}, "
                f"{summary['bootstrap_95_ci_pct'][1]:.2f}]"
            )
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
