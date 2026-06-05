# Benchmark protocol

This document is the reproduction contract for any tsugi-mend measurement that ends up in a published artifact (paper, blog post, results doc).

## Cardinal rule

**Same workload, same checkpoint, same hardware, different synchronization.** Every paired run must hold model, tokenizer, dataset, sequence length, global batch, micro-batch, seed set, optimizer, learning-rate schedule, and any other hyperparameters constant between `--mode baseline` and `--mode sdk`. Only `MendConfig` (and the implicit FSDP / NCCL behavior it modifies) varies.

## Primary measurement

Tokens per second, observed at rank 0, averaged over the steady-state portion of the run (excluding warmup steps 0-50). Sample n = step count - warmup.

## Bit-exact referent

When a tsugi-mend artifact reports bit-exact loss equivalence, the referent is
the benchmark's synchronous-reducer path. In the current harness, both
`baseline` and `sdk` use the same Decoupled-DiLoCo-style local-step plus
periodic-merge construction and apply the same merged delta at the same
`apply_lag_steps` offset. The bit-exact check proves that moving the merge wait
onto the concurrent outer-step path does not change numerics. It is not a claim
that either path is numerically equal to a vanilla DDP/FSDP all-reduce run.

## Secondary measurements (always report)

- p50, p95, p99 step time in milliseconds (rank 0, steady state).
- Time to target validation loss (when validation set is held out).
- Final loss at fixed token budget (use this for short-run Stage C / D / E paired runs where time-to-target is impractical).
- Recovery behavior under injected slowdown / fault (Stage D, optional Stage E).
- Cost per run using a public cloud-price proxy.

## Repeated-runs protocol

- Stage B: 1 run (smoke test; no statistics).
- Stage C: 1 paired run (baseline + sdk).
- Stage D: 3 paired runs minimum. Alternate baseline and sdk to interleave drift in cluster performance over time.
- Stage E: 3 paired runs minimum. Same alternation rule.

## Stall-sweep protocol

The stall sweep is the local, reproducible harness for answering "when does the
concurrent outer-step path help under injected peer slowdown?" It is a
measurement tool only. It does not change SDK runtime behavior, enable rank
exclusion, or alter the default lossless path.

For each grid point:

- Hold workload, seed-specific data, optimizer, sync cadence, and all SDK
  settings fixed between the `baseline` and `sdk` paths.
- Inject a deterministic fixed `straggler_delay_ms` on the configured peer-rank
  set at the outer-step fragment-exchange boundary. This knob is distinct from
  `simulated_merge_delay_ms`, which remains the symmetric merge-transport delay
  inside the reducer finalize path.
- Use `straggler_delay_ms` in `{0, 50, 100, 250, 500, 1000}` and peer-straggler
  counts in `{0, 1, 2, 4}` for the full local grid. Rank 0 remains the reporting
  observer; peer-straggler count 4 therefore requires at least 5 local CPU/gloo
  ranks.
- Run at least `n_seeds=5` paired trials per grid point. Alternate path order
  across seeds (`baseline_sdk`, then `sdk_baseline`) to interleave local
  performance drift.
- Assert bit-exact loss equivalence on every seed. A bit-exact failure makes the
  grid point invalid and the sweep exits non-zero. It is not treated as a slow
  run.
- Compute one SDK-over-baseline tokens/s uplift per seed. Drop the single
  lowest and single highest seed-level uplift, then report mean, sample
  variance, and a bootstrap 95% CI across the surviving seed-level uplifts.
- Report p50, p95, and p99 step time for both paths, plus the mean tokens/s for
  both paths.

The expected public artifact is the uplift-vs-injected-stall curve, emitted as
`benchmarks/results/<cell>/result.json`. The quick smoke
(`python benchmarks/run_stall_sweep.py --quick`) uses the same n>=5/drop rule
on a smaller `{0, 50}` ms x `{0, 1}` peer-straggler grid.

The sweep may feed per-rank timing observations into `FailSlowDetector` in
observe-only mode and record flagged ranks plus z-scores. This validates the
detector signal under a known injected slowdown. The sweep must not call
`mark_failslow`, exclude a learner from the merge quorum, or otherwise invoke
mitigation, because that changes the numerical contract.

## 95% confidence intervals

Use a bootstrap CI (10000 resamples). Report the CI alongside the point estimate every time the uplift is quoted.

## Hardware reporting

For every external artifact:

- Provider name (Lambda Labs, CoreWeave, Modal, etc.).
- Instance type (e.g., `gpu_8x_h100_sxm5`).
- Number of nodes.
- Inter-node fabric description (commodity Ethernet, IB, RoCE; measured effective bandwidth via NCCL bench).
- PyTorch / NCCL / CUDA / driver versions (from the pinned container).
- Total measurement cost.

## Workload reporting

- Model name + commit hash (HF revision).
- Tokenizer name + revision.
- Dataset name + revision + subset spec.
- Global batch, micro-batch, sequence length.
- Optimizer + LR + LR schedule + warmup.
- Seed set.

## SDK configuration reporting

Dump `MendConfig` as JSON at `mend_init` time into the diagnostics file (this happens via the `mend_init` event payload). The "SDK configuration" section of any results writeup is a direct read from that JSON.

## Expected delta envelope

Pre-registered expectations for cross-rack software-only distributed training:

Each artifact must name its baseline. The current public harness uses the
synchronous-reducer path as its paired baseline; a vanilla DDP/FSDP comparison
must be labeled separately if a future artifact runs one.

| Stage | Expected throughput delta vs the named synchronous baseline | Interpretation |
|---|---|---|
| C (intra-node simulated two-rack) | -5% to +5% | NO uplift expected; checking for non-regression |
| D (2x A100 cross-Ethernet) | +5% to +15% | First realistic cross-rack measurement |
| E (2x 8xH100 cross-network) | +15% to +30%, with +40% upside on straggler-heavy jobs | production-fabric measurement |

Any extrapolation table should anchor on the MEASURED Stage E delta, not the predicted envelope.

## Negative-result discipline

If Stage D or Stage E measures uplift below the PASS threshold (negative, or positive but with CI crossing 0):

- Document the result honestly. Report it as "modest" or "inconclusive" rather than reframing.
- Do not retry with different seeds hoping for a different number; that is forking-path territory.
- Review before any further cloud spend.

This applies to PASSED results too: the absolute number must be reported with its 95% CI, not just a point estimate.

## Citation verification

The measurements in any published artifact are first-party benchmark numbers, not literature citations. The public-art references that frame the measurement (Decoupled DiLoCo, DES-LOC, async-TP, FALCON) should be verified against their sources before any artifact ships.
