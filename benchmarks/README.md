# tsugi-mend benchmark harness

A public-safe, runnable implementation of the paired-run protocol in
[`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md). It closes the
"can't go from `pip install` to a reproduced benchmark" gap with a **$0
cheap-reproducible** paired-run cell that anyone can run locally, plus a
results-bundle format that a future real-cluster run drops straight into.

The harness is **additive**: it does not change SDK runtime behavior, the
public API, default modes, or CI.

## What it does

For a given config the driver runs the SAME workload (same model, seed, data,
optimizer, sync cadence) under two synchronization paths and compares them:

| Path | Cross-rack merge | Who pays the merge delay |
|---|---|---|
| `baseline` | `GraceWindowSyncer` driven synchronously | the training thread BLOCKS on it |
| `sdk` | `ConcurrentOuterStep` orchestrator (`mend_init`, `concurrent_outer_step=True`) | the asyncio loop thread; the training thread OVERLAPS it |

Both paths apply the SAME Decoupled-DiLoCo `token_weighted_merge` delta at the
SAME inner step, so the parameter trajectory -- and therefore the per-step
loss -- is identical. This bit-exact check is relative to the synchronous
reducer path in this harness, not to a vanilla DDP/FSDP all-reduce run. The
driver then:

1. **asserts bit-exact loss equivalence** in the default (lossless) mode --
   the load-bearing invariant. It verifies this (elementwise IEEE-754 equality
   of the per-step loss trajectories); it does not assume it. If the default
   path ever stops being bit-exact, the cell FAILS and the process exits
   non-zero.
2. measures steady-state tokens/s for both paths (excluding warmup, per the
   protocol) and reports the SDK-over-baseline uplift with a **paired-bootstrap
   95% CI** (10000 resamples).
3. writes a public-safe result bundle to `benchmarks/results/<cell>/result.json`.

## Files

```
benchmarks/
  metrics.py                       pure helpers: bit_exact_equal, steady_state,
                                   bootstrap_uplift_ci, aggregate_seeded_uplift
                                   (unit-tested)
  run_paired.py                    config-driven paired-run driver
  run_stall_sweep.py               seeded peer-straggler sweep driver
  results/<cell>/result.json       committed result bundle (schema below)
  results/<cell>/README.md         per-cell reproduction notes
```

## Run the cheap cell ($0)

```bash
pip install -e ".[dev]"          # torch is the only runtime dep
python benchmarks/run_paired.py --cell cpu_gloo_2rank_mlp
```

Expected output shape (the exact uplift number varies run-to-run on a CPU;
bit-exact is invariant):

```
================================================================
cell: cpu_gloo_2rank_mlp  (cheap)
hardware: local CPU (gloo); $0 cheap reproducible cell
bit-exact loss equivalence (default mode): PASS  (max |loss diff| = 0.000e+00 over 160 steps)
baseline tokens/s: <...>  (mean step <...> ms)
sdk      tokens/s: <...>  (mean step <...> ms)
uplift: +<...>%  (95% CI [+<...>%, +<...>%], n=140 paired steps, 10000 resamples)
================================================================
wrote bundle: benchmarks/results/cpu_gloo_2rank_mlp/result.json
```

The process exits `0` on a bit-exact PASS and `1` on a FAIL. Pass `--no-write`
to print the summary without overwriting the committed bundle.

The same cheap cell can also run under `torchrun` through the env:// launch
path that real multi-node jobs use:

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 \
  benchmarks/run_paired.py --launch torchrun --cell cpu_gloo_2rank_mlp --no-write
```

This is still a $0 CPU/gloo run. It validates that each torchrun process acts
as one benchmark rank, that rank 0 writes the bundle, and that fragment object
gather uses a dedicated gloo process group rather than the data-plane backend.

## Run the stall sweep ($0 quick smoke)

The stall sweep injects a fixed extra wait on selected peer ranks at the
outer-step fragment-exchange boundary. This is distinct from
`simulated_merge_delay_ms`: the simulated merge delay is symmetric transport
cost inside the reducer finalize path, while `straggler_delay_ms` models one or
more slow peers reaching the merge boundary late. Both paths receive the same
injected wait. The baseline blocks on it through synchronous fragment exchange;
the SDK path submits the exchange to the concurrent outer-step provider so the
training thread can overlap it.

Quick smoke:

```bash
python benchmarks/run_stall_sweep.py --quick
```

Expected output shape:

```
======================================================================================
cell: cpu_gloo_stall_sweep_quick  (quick stall sweep)
bit-exact loss equivalence: PASS (max |loss diff| = 0.000e+00)
delay=   0 ms  stragglers=0  bit_exact=PASS  uplift=<...>% CI95 [<...>%, <...>%]  p95 baseline/sdk=<...>/<...> ms
delay=   0 ms  stragglers=1  bit_exact=PASS  uplift=<...>% CI95 [<...>%, <...>%]  p95 baseline/sdk=<...>/<...> ms
delay=  50 ms  stragglers=0  bit_exact=PASS  uplift=<...>% CI95 [<...>%, <...>%]  p95 baseline/sdk=<...>/<...> ms
delay=  50 ms  stragglers=1  bit_exact=PASS  uplift=<...>% CI95 [<...>%, <...>%]  p95 baseline/sdk=<...>/<...> ms
======================================================================================
wrote bundle: benchmarks/results/cpu_gloo_stall_sweep_quick/result.json
```

The quick mode still uses `n_seeds=5`, so every grid point exercises the
run-level aggregation rule: compute each seed's paired tokens/s uplift, drop
the single lowest and highest uplift, then report mean, sample variance, and a
bootstrap 95% CI across the surviving seed-level uplifts. It uses a reduced
model shape, fewer steps, the delay grid `{0, 50}` ms, and straggler counts
`{0, 1}` so it is suitable as a local smoke.

Full local grid:

```bash
python benchmarks/run_stall_sweep.py
```

The full grid uses `straggler_delay_ms` in `{0, 50, 100, 250, 500, 1000}` and
peer-straggler counts in `{0, 1, 2, 4}`. The default full sweep uses five local
CPU/gloo ranks so rank 0 can remain the reporting observer while ranks 1..4
serve as the maximum peer-straggler set. The full run is intentionally not a CI
gate; it is a local measurement artifact.

The sweep writes `benchmarks/results/<cell>/result.json` with one row per grid
point. Each row includes:

- `delay_ms`, `straggler_count`, and the concrete `straggler_ranks`.
- `mean_uplift_pct`, `sample_variance_pct2`, and `ci95_low_pct` /
  `ci95_high_pct` after dropping the fastest and slowest seed-level uplifts.
- `bit_exact_pass` and `max_abs_loss_diff`; any seed that fails bit-exact loss
  equivalence invalidates the grid point and exits non-zero.
- `p50/p95/p99` step time means for both paths and mean tokens/s for both paths.
- an observe-only `FailSlowDetector` report. The sweep feeds per-rank timing
  observations into the detector and records flagged ranks and z-scores, but it
  never calls rank exclusion or quorum-dropping mitigation.

## Cells: cheap-reproducible vs real hardware

| Cell | Backend | Reproducible at | Notes |
|---|---|---|---|
| `cpu_gloo_2rank_mlp` | `gloo` (CPU) | **$0, any laptop** | the cheap cell; synthetic MLP, 2 local ranks, injected merge delay |
| `cpu_gloo_stall_sweep_quick` | `gloo` (CPU) | **$0, any laptop** | quick seeded peer-straggler sweep result bundle |
| `real_8xv100_2node` | `nccl` (GPU) | a real GPU cluster (**out of scope here**) | Hugging Face causal LM, per-node FSDP sharding, deterministic token stream, no simulated merge delay |

### Real multi-node Hugging Face/FSDP cell

The real cell is implemented but not cheap-reproducible. It requires CUDA, an
`nccl` data-plane process group, torchrun/env:// launch, and the optional
real-cell dependencies:

```bash
pip install -e ".[real-cell]"
```

Example two-node shape, with eight local processes per node:

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id=tsugi-mend-real-cell-001 \
  benchmarks/run_paired.py \
    --launch torchrun \
    --cell real_8xv100_2node \
    --hardware-label "Provider, 2 nodes x 8 GPUs, fabric, pinned CUDA/NCCL/PyTorch"
```

The real cell loads `HuggingFaceTB/SmolLM-135M` and its tokenizer lazily only
inside that path, wraps the model in per-node FSDP groups, and exchanges
same-local-rank shard deltas across nodes through a dedicated gloo object
gather group. With `dataset_id` unset, the workload uses a deterministic
synthetic token stream from the tokenizer vocab, so no dataset download is
required. If `--dataset-id` is supplied, the real path lazily loads a small
deterministic `train` slice through the optional `datasets` dependency and
tokenizes that text instead. `simulated_merge_delay_ms` is `0`; real
cross-network latency is the measured delay.

Running this cell requires provisioning a GPU cluster and is a separate
maintainer task. This harness does NOT provision compute and does NOT run it.
The bundle format already carries the hardware / workload / fabric fields the
protocol requires, so a real run drops into the same
`benchmarks/results/<cell>/result.json` shape.

## Result-bundle format (`result.json`)

```jsonc
{
  "schema_version": "1.0",
  "cell": "<cell name>",
  "reproducible": "cheap" | "real-hardware (requires a GPU cluster)",
  "protocol": "docs/benchmark_protocol.md",
  "hardware": { "label", "platform", "machine", "python", "torch", "backend",
                "ranks", "launch" },
  "workload": { "kind", "model_id", "tokenizer_id", "dataset_id",
                "batch", "in_dim", "hidden", "out_dim", "sequence_length",
                "tokens_per_step", "optimizer", "lr", "seed" },
  "sdk_config": { "quorum_min_learners", "grace_window_ms", "sync_period_steps",
                  "apply_lag_steps", "simulated_merge_delay_ms",
                  "outer_step_compression_mode", "concurrent_outer_step",
                  "token_weighted_merge" },
  "run": { "steps", "warmup_steps", "n_steady_steps" },
  "bit_exact_loss_equivalence": { "passed", "method", "n_steps_compared",
                                  "first_baseline_loss", "last_baseline_loss",
                                  "max_abs_loss_diff" },
  "metrics": {
    "baseline": { "tokens_per_second", "mean_step_time_ms", "p50/p95/p99_step_time_ms" },
    "sdk":      { "tokens_per_second", "mean_step_time_ms", "p50/p95/p99_step_time_ms" },
    "uplift":   { "tokens_per_second_pct", "ci95_low_pct", "ci95_high_pct",
                  "n_paired_steps", "n_bootstrap_resamples", "confidence" }
  }
}
```

The bundle is **public-safe**: `hardware.label` and `platform` are generic OS
/ library strings; no hostnames, IPs, internal paths, or spend figures.

## How to read the cheap-cell numbers honestly

- **Bit-exact is the robust result.** `max_abs_loss_diff` is exactly `0.0` and
  reproduces every run. That is the invariant the harness exists to guard.
- **The CPU uplift is a real but noisy point estimate.** It measures the
  overlap mechanism (the baseline blocks on the merge delay; the SDK overlaps
  it), but absolute CPU step times vary with OS scheduling, so report it with
  its 95% CI, never as a bare point estimate. Across repeated local runs the
  estimate stayed positive with a CI above zero.
- **It is a NEW cheap result, not a restatement of any headline number.** It
  does not reproduce, stand in for, or extrapolate to a production cross-rack
  measurement; it demonstrates the harness and the bit-exact invariant at $0.
- **The uplift is conditional on there being a merge delay to overlap.** With
  `--simulated-merge-delay-ms 0` the orchestrator has nothing to overlap, so its
  asyncio coordination is pure overhead and the SDK path measures *slower* than
  the baseline (a small negative uplift) -- still bit-exact. The SDK helps only
  when a real cross-rack merge wait exists for the inner-step compute to hide
  behind; the cheap cell injects that delay to exhibit the mechanism. Run it at
  `--simulated-merge-delay-ms 0` yourself to see the no-delay overhead floor.
