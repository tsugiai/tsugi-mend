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
loss -- is identical. The driver then:

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
                                   bootstrap_uplift_ci (unit-tested)
  run_paired.py                    config-driven paired-run driver
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

## Cells: cheap-reproducible vs real hardware

| Cell | Backend | Reproducible at | Notes |
|---|---|---|---|
| `cpu_gloo_2rank_mlp` | `gloo` (CPU) | **$0, any laptop** | the cheap cell; synthetic MLP, 2 local ranks, injected merge delay |
| real multi-node cross-rack | `nccl` (GPU) | a real GPU cluster (**out of scope here**) | the harness is READY for it -- see below |

### Scaling to a real multi-node config (args only)

The same driver scales by command-line args; nothing in the measurement code
changes. For example, a real cell would pass a real backend / rank count /
hardware label and (in a follow-up that wires a Hugging Face workload) a real
`model_id` / `tokenizer_id` / `dataset_id`:

```bash
python benchmarks/run_paired.py \
    --cell real_8xgpu_2node \
    --backend nccl --ranks 2 \
    --steps 500 --warmup-steps 50 \
    --sync-period-steps 128 \
    --hardware-label "Provider X, 2x 8xGPU, <fabric>, <pinned PyTorch/NCCL/CUDA>"
```

Running a real multi-node cell requires provisioning a GPU cluster and is a
separate task that is out of scope for this harness. This harness does NOT
provision compute and does NOT run it. The bundle format already carries the
hardware / workload / fabric fields the protocol requires, so a real run drops
into the same `benchmarks/results/<cell>/result.json` shape.

## Result-bundle format (`result.json`)

```jsonc
{
  "schema_version": "1.0",
  "cell": "<cell name>",
  "reproducible": "cheap" | "real-hardware (requires a GPU cluster)",
  "protocol": "docs/benchmark_protocol.md",
  "hardware": { "label", "platform", "machine", "python", "torch", "backend", "ranks" },
  "workload": { "kind", "model_id", "tokenizer_id", "dataset_id",
                "batch", "in_dim", "hidden", "out_dim", "tokens_per_step",
                "optimizer", "lr", "seed" },
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
