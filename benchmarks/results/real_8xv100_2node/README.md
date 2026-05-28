# Result bundle: `real_8xv100_2node`

This is the **real-hardware multi-node cell**. Unlike the `$0` `cpu_gloo_2rank_mlp`
cell, running it requires provisioning a real 2-node GPU cluster + the optional
`real-cell` (or `benchmark`) extra (`pip install 'tsugi-mend[real-cell]'`). The
harness does not provision compute; the cell config below documents the shape
of the run + the cross-seed aggregate from the measurements committed to this
folder.

## Protocol

Follows [`docs/benchmark_protocol.md`](../../../docs/benchmark_protocol.md): same
workload, same checkpoint, same seed, same data, different synchronization;
bit-exact loss equivalence asserted in default mode; tokens/s reported with a
paired-bootstrap 95% CI per run plus the cross-seed CI across `n` seeds.

- **Cell config** (`benchmarks/run_paired.py` `CELLS["real_8xv100_2node"]`):
  backend `nccl`, launch `torchrun`, ranks 16, steps 500, warmup 50,
  sync_period_steps 128, apply_lag_steps 4, simulated_merge_delay_ms 0,
  model_id `HuggingFaceTB/SmolLM-135M`, batch 8, seq_len 512.
- **Reproduction (one paired run)** — launch the harness across the cluster
  with the brief in [`docs/multinode.md`](../../../docs/multinode.md):

  ```bash
  torchrun --nnodes=2 --nproc-per-node=8 --node-rank={0,1} \
    --master-addr=<node0-priv-ip> --master-port=29500 \
    benchmarks/run_paired.py --launch torchrun --cell real_8xv100_2node \
    --backend nccl --ranks 16 --steps 500 --warmup-steps 50 --seed <s> \
    --hardware-label "<provider>, 2x 8xV100 (16GB), commodity Ethernet"
  ```

- **For an `n`-seed cross-seed CI**, alternate baseline/sdk paired runs across
  `n` distinct seeds (the protocol's drift-interleaving rule).

## Measurement (n=7, Lambda Labs us-south-2, commodity Ethernet)

`n=7` paired seeds. Each run: 500 paired steps, sync_period 128, apply_lag 4,
warmup 50. Per-run CIs are paired-bootstrap (10000 resamples) over the
steady-state paired step times.

| seed | bit-exact | max\|loss diff\| | baseline tok/s | sdk tok/s | per-run uplift | per-run CI95 |
|---:|:---:|---:|---:|---:|---:|---:|
| 20240527 | **PASS** | 0.0 | 1472.9 | 1690.9 | **+14.80%** | [-5.81, +41.73] |
| 20240528 | **PASS** | 0.0 | 1734.7 | 1662.1 | **-4.19%**  | [-9.76, +2.97]  |
| 20240529 | **PASS** | 0.0 | 1689.6 | 1777.4 | **+5.20%**  | [-0.81, +12.82] |
| 20240530 | **PASS** | 0.0 | 1562.4 | 1753.1 | **+12.21%** | [+6.51, +19.20] |
| 20240531 | **PASS** | 0.0 | 1644.8 | 1595.7 | **-2.98%**  | [-13.33, +11.18] |
| 20240601 | **PASS** | 0.0 | 1713.1 | 1537.5 | **-10.25%** | [-24.70, +11.86] |
| 20240602 | **PASS** | 0.0 | 1536.1 | 1670.7 | **+8.76%**  | [-11.20, +39.36] |

### Cross-seed (n=7)

- **Bit-exact loss equivalence: PASS on every seed.** `max|loss diff| = 0.0`
  over 500 steps × 7 seeds. The load-bearing invariant is robust.
- **Throughput uplift: mean +3.36%, stdev 9.35pp, range [-10.25%, +14.80%].**
  Cross-seed `t`-CI95 (df=6) = **[-5.28%, +12.01%]**; bootstrap CI95 over the
  per-seed point estimates (10000 resamples) = [-3.20%, +9.61%]. **Both CI95
  bands straddle zero.**
- **Baseline tok/s itself varies materially seed-to-seed** (1472.9 - 1734.7
  tok/s, ~18% spread), indicating fabric-side noise that dominates the SDK
  signal over the ~3-4 outer rounds a 500-step run at sync_period 128 contains.

## How to read these numbers honestly

- **Bit-exact loss equivalence is the robust headline.** The SDK's default mode
  preserves per-step loss to IEEE-754 equality across every paired run, every
  seed, every fabric condition we've measured. This is what the SDK guarantees.
- **Throughput uplift on real cross-network is jitter-conditional.** The SDK's
  overlap mechanisms (Decoupled-DiLoCo sync at sync_period boundaries plus the
  grace-window aggregator and sideband control plane) deliver speedup *only when
  there is cross-network jitter to hide*. When the fabric is calm and the
  synchronous-reducer baseline isn't bottlenecked, the SDK's coordination is
  small overhead and the uplift trends toward zero (or slightly negative for
  random-token workloads where the data path itself isn't a bottleneck).
- **A previously-reported single-run measurement on this same hardware setup
  produced +28.58%.** Re-measurement under the protocol above shows the
  +28.58% point estimate is **inside the high tail of the measured envelope but
  not representative of the mean**. The conditional-on-jitter interpretation
  matches the per-seed spread (-10% to +15%) and the baseline-tok/s variance: on
  a high-jitter day the SDK preserves throughput while the baseline collapses;
  on a calm day the gap closes.
- **Treat the bit-exact PASS as the invariant**; treat any specific tps uplift
  number as a point estimate that should be reported with a range or CI, never
  as a bare magnitude. See [`docs/benchmark_protocol.md`](../../../docs/benchmark_protocol.md)
  for the "report mean + CI, never bare point estimate" rule.

## Reproducing the n=7 measurement

The reproduction contract is the protocol above. Cumulative spend on Lambda
Labs `gpu_8x_v100_n` at the listed config is ~$0.5-1 per paired run + setup;
n=7 + interleaving alternation runs in ~30-60 min wall-clock on two on-demand
instances in the same region.
