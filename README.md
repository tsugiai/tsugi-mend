# tsugi-mend

[![PyPI version](https://img.shields.io/pypi/v/tsugi-mend.svg)](https://pypi.org/project/tsugi-mend/)
[![Python versions](https://img.shields.io/pypi/pyversions/tsugi-mend.svg)](https://pypi.org/project/tsugi-mend/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/tsugiai/tsugi-mend/actions/workflows/ci.yml/badge.svg)](https://github.com/tsugiai/tsugi-mend/actions/workflows/ci.yml)

Cross-rack reducer toolkit for PyTorch training loops.

`tsugi-mend` is a software-only component toolkit for wiring
Decoupled-DiLoCo-style periodic merges and concurrent outer-step overlap into
a training loop. It is not a transparent 0.1.x drop-in that intercepts DDP or
FSDP collectives by itself. The caller drives the integration points at
outer-step boundaries, supplies parameter-delta fragments, collects merged
deltas, and applies them at the same lag as the synchronous-reducer path. The
examples and benchmark driver are the worked integrations.

Public-art references and 0.1.x implementation status:

- **Decoupled DiLoCo for Resilient Distributed Pre-training** (Arthur Douillard et al., arXiv:2604.21428, April 2026): the reducer implements minimum quorum, adaptive grace window, and token-weighted merge.
- **Concurrent outer-step overlap**: the `ConcurrentOuterStep` orchestrator is wired when `concurrent_outer_step=True`, so the training thread can overlap the grace-window wait with inner-step compute.
- **DES-LOC / Local Adam** (Iacob et al., arXiv:2505.22549, May 2025; ICLR 2026): desynchronized synchronization-period components are present, but moment synchronization is not automatically wired into `mend_init` in 0.1.x.
- **Async tensor parallelism** (PyTorch / TorchTitan, September 2024): treated as an integration component/configuration point, not automatically installed by `mend_init` in 0.1.x.
- **FALCON fail-slow detection** (arXiv:2410.12588, October 2024): the runtime observes step times and can emit detection diagnostics; FALCON-style quorum exclusion/mitigation is not wired in 0.1.x.
- **Gradient compression** (`none`, `int8`, `powersgd`, `sparse`): primitives and config validation are present; the default path is lossless `none` (`sparse` is also lossless, with a dense fallback), and compression is not invoked by the 0.1.x runtime outer-step path.

The SDK keeps intra-rack TP / CP / PP / FSDP collectives unchanged. In 0.1.x,
the public runtime exercises the reducer plus concurrent outer-step overlap;
the other mechanisms above are components or integration points.

## Install

```bash
pip install tsugi-mend
```

Or install the unified surface that bundles this SDK with the companion patent-aligned SDK:

```bash
pip install tsugi   # exposes tsugi.mend and tsugi.kpool
```

For local development:

```bash
pip install -e ".[dev]"
```

## License and IP posture

This SDK is licensed under **Apache-2.0** with its full automatic patent grant. The SDK is patent-independent by deliberate construction: it does not exercise the K-Pool LoRA (US App. 64/060,315) or Infinity (US App. 64/055,093) patent estates that the companion SDK (`tsugi-kpool`, also Apache-2.0) does. Read the preamble at the top of the `LICENSE` file for the full posture explanation.

The companion patent-aligned SDK at [`github.com/tsugiai/tsugi-kpool`](https://github.com/tsugiai/tsugi-kpool) is the software embodiment of those two TsugiCinema patent estates. The two SDKs share zero code.

## Measurements

The numbers below are first-party internal benchmark measurements taken
under the reproduction contract in [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md)
(same workload / checkpoint / hardware, baseline vs SDK, paired runs,
bootstrap 95% CI). The raw per-run results logs are internal; the
protocol document is the public reproduction pointer, and the headline
cells below can be re-derived by anyone who runs the protocol on the
stated hardware. We report point estimates with their 95% CI where
available and flag single-seed (n=1) cells explicitly.

### Production-grounded results

The robust headline is **bit-exact loss equivalence in default mode**. It is
preserved across every paired run, every seed, every fabric condition we have
measured.
Throughput uplift on real cross-network is **jitter-conditional**: the SDK's
overlap mechanisms hide cross-rack latency when it exists, so the magnitude of
the uplift depends on the fabric jitter present at measurement time.

| Workload | Hardware | Measurement |
|---|---|---|
| **Real cross-network 2-node 8xV100 (synchronous reducer)** | Lambda Labs commodity Ethernet, SmolLM-135M, 500 paired steps × 7 seeds | **Bit-exact loss PASS on every seed** (max\|loss diff\| = 0.0); uplift **mean +3.4%, CI95 [-5%, +12%]**, per-seed range **[-10%, +15%]** (n=7). Details: [`benchmarks/results/real_8xv100_2node/`](benchmarks/results/real_8xv100_2node/) |
| Production-realistic multi-GPU FSDP + 7B model (realistic floor, 3-seed CI) | Modal 8xH100 FSDP FULL_SHARD, Qwen-2.5-7B + simulated 2-rack, 4 delays × 3 seeds | +6.37% ± 1.31% at 2000ms (n=3) |
| H100 Hopper single-instance (synchronous reducer baseline) | Modal 8x H100 SXM5, Llama-3-8B, 2000 paired steps × 3 seeds | -0.97% ± 1.5% (predicted null; Hopper NVLink absorbs the synchronous-path cross-rack tax) |

How to read the production-grounded numbers honestly:

- **Bit-exact loss equivalence is the load-bearing result.** Every cross-network
  paired run preserves loss to IEEE-754 equality vs the synchronous-reducer
  baseline: both paths apply the same Decoupled-DiLoCo-style merged delta at
  the same lag, and the concurrent path only moves the merge wait off the
  training thread. This is not a claim that either path is numerically equal to
  a vanilla DDP/FSDP all-reduce run.
- **Throughput uplift on real cross-network is jitter-conditional, not a fixed
  magnitude.** On the 2-node 8xV100 commodity-Ethernet cell, `n=7` re-measurement
  under [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) shows mean
  **+3.4%** with CI95 **[-5%, +12%]** and a per-seed range of **[-10%, +15%]**.
  Baseline tok/s itself varies ~18% seed-to-seed (1473-1735), and that
  fabric-side variance dominates the SDK signal over the ~3-4 outer rounds a
  500-step run at sync_period 128 contains. A prior single-run measurement on
  the same setup produced **+28.58%** during a higher-jitter Lambda Ethernet
  session; that point estimate sits in the high tail of the measured envelope
  and is **not representative of the mean** under n>=3 protocol. Report any
  cross-network uplift number with a range or CI, per the protocol's "never a
  bare point estimate" rule.
- Production-realistic multi-GPU FSDP yields a smaller honest floor (**+6.37%
  ± 1.31%, n=3**) at injected 2000ms delay because 8-rank NCCL pipelining
  absorbs some of the simulated delay.
- **Protocol-incomplete single-seed note.** The real-fabric Hopper 2-pod
  InfiniBand / RoCE result is not comparable to the n>=3 rows above yet:
  RunPod 2x 8x H100 SXM5 over real InfiniBand / RoCE v2 3.2 Tbps, Llama-3-8B,
  500 paired steps × 1 seed, measured +1.42% tps with +0.18% loss delta. The
  **n=1 caveat is load-bearing** because the point estimate is the same order
  of magnitude as baseline-only seed variance; n=3 CI is pending.

### Ceiling-case / simulated-delay results

Every cell in this subsection uses an injected simulated grace-window
delay on a single instance or simulated two-rack setup, not a real
cross-network measurement. These are ceiling-case stress tests for the
overlap mechanism rather than production numbers.

| Workload | Hardware | Measurement |
|---|---|---|
| Statistical-confidence ceiling case (Hopper 3-seed CI) | Modal H100:1, Qwen-2.5-1.5B, 200 steps × 3 seeds at 2000ms grace window | **+71.49% ± 2.83% (95% CI, n=3)** throughput uplift |
| Cross-rack grace-window overlap on Hopper at 7B scale | Modal H100:1, Qwen-2.5-7B, 200 steps × 5 delays | +76.58% at 2000ms; +39.72% at 1000ms; +19.20% at 500ms |
| Intermediate model-scale (3B) confirmation (Hopper 3-seed CI) | Modal H100:1, Qwen-2.5-3B, 200 steps × 4 delays × 3 seeds | +41.31% ± 0.29% at 2000ms (n=3); +20.40% ± 0.03% at 1000ms; +9.75% ± 0.04% at 500ms |
| Cross-rack grace-window overlap at 1.5B scale | Modal H100:1, Qwen-2.5-1.5B, 200 steps × 7 delays | +70.64% at 2000ms; +34.73% at 1000ms; +16.86% at 500ms |
| Cross-rack grace-window overlap on A10G | Modal A10G, SmolLM-135M, 200 steps × 7 delays | +52.75% at 2000ms (constant); +11.61% at 500ms; -0.06% overhead at 0ms; bit-exact loss preserved across all cells |

How to read the ceiling-case numbers honestly:

- The **+71.49% ± 2.83% (n=3) Hopper result** is a single-instance measurement with an injected simulated grace-window delay, not a real cross-network result. Read it as a ceiling-case for the overlap mechanism.
- The orchestrator's uplift is governed by `N · T_step / G` (sync-period steps × per-step compute time vs grace-window ms). Apparent non-monotonicity with model size (Qwen-3B +41.31% below both Qwen-1.5B and Qwen-7B) is explained by the Qwen-7B measurement using 1/8 the tokens-per-step (seq_len 1024 / mbs 1 vs 2048 / 4); at fixed tokens-per-step, uplift is monotonically decreasing in model size.
- Constant-delay headlines (e.g. +52.75% A10G at 2000ms) are ceiling-case stress tests. The FALCON paper documents cross-rack inter-node RDMA variance (CoV=0.29) but does not characterize the per-iteration latency distribution shape; the delay sweep is a stress test, not a literal FALCON replay.

At every scale the concurrent path's throughput is rock-solid across delays (Qwen-7B single-process: 4,300 ± 80 tok/s; Qwen-3B Hopper: 18,153 ± 4 tok/s; Qwen-1.5B Hopper: 30,840 ± 35 tok/s; SmolLM-135M A10G: 23,610 ± 80 tok/s) while the synchronous baseline collapses linearly with delay.

### Run it multi-node

See [`docs/multinode.md`](docs/multinode.md) for the multi-node launch walkthrough.

## Status

**Pre-Alpha (0.1.3).** APIs are stabilizing and may change before v1.0. Published to PyPI as `tsugi-mend`; also reachable through the unified `tsugi` meta-package as `tsugi.mend`. The staged validation (Stage A unit/integration through cross-network production-fabric runs) all passed under the protocol above; the real-fabric Hopper cross-network result is point-estimate closed (n=1), with an n=3 CI pending.

## Quickstart

```python
from tsugi_mend import MendConfig, mend_init, mend_shutdown
from tsugi_mend.runtime import get_runtime

config = MendConfig(
    quorum_min_learners=4,
    grace_window_ms=2000,
    token_weighted_merge=True,
    sync_period_steps=128,
    # Orchestrator overlaps the cross-rack outer-step wait with inner-step
    # compute. Default True.
    concurrent_outer_step=True,
    diagnostics_dir="./results/mend_diag",
)

mend_init(model, config)
runtime = get_runtime(model)

for step, batch in enumerate(loader):
    runtime.step_begin(step)
    loss = train_one_step(model, optimizer, batch)

    sched = runtime.schedule_for(step)
    if sched.should_sync_params and not runtime.outer_step_in_flight():
        runtime.outer_step_begin(
            round_id=step,
            fragment_provider=make_fragment_provider(...),
        )

    result = runtime.outer_step_collect()
    if result is not None:
        apply_merged_delta(model, result.merged_delta)

    runtime.step_end(step)

mend_shutdown(model)
```

The snippet above is the integration shape, not a complete program. Use the
examples below for runnable wiring, and the benchmark driver for the fuller
fragment gather and merge-application path.

Two runnable, CPU-only integration examples (no GPU or multi-node required):

- [`examples/minimal_single_process.py`](examples/minimal_single_process.py) - smallest end-to-end use on a toy `nn.Module`.
- [`examples/concurrent_orchestrator.py`](examples/concurrent_orchestrator.py) - wiring the `ConcurrentOuterStep` orchestrator into a training loop with a synthetic single-rank fragment provider.

```bash
python examples/minimal_single_process.py
python examples/concurrent_orchestrator.py
```

## Layout

```
src/tsugi_mend/   SDK source
tests/            unit and integration tests (CPU-only)
docs/             architecture, benchmark protocol, convergence-equivalence sketch
examples/         minimal CPU-only training-loop integration examples
```

## Companion SDK

For LoRA-adapter-granularity productization that exercises the K-Pool LoRA and Infinity patent estates, see [`tsugi-kpool`](https://github.com/tsugiai/tsugi-kpool). The two SDKs share zero code and can be installed and used independently, or together via the unified [`tsugi`](https://github.com/tsugiai/tsugi) meta-package.
