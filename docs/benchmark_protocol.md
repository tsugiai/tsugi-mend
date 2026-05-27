# Benchmark protocol

This document is the reproduction contract for any tsugi-mend measurement that ends up in a published artifact (paper, blog post, results doc).

## Cardinal rule

**Same workload, same checkpoint, same hardware, different synchronization.** Every paired run must hold model, tokenizer, dataset, sequence length, global batch, micro-batch, seed set, optimizer, learning-rate schedule, and any other hyperparameters constant between `--mode baseline` and `--mode sdk`. Only `MendConfig` (and the implicit FSDP / NCCL behavior it modifies) varies.

## Primary measurement

Tokens per second, observed at rank 0, averaged over the steady-state portion of the run (excluding warmup steps 0-50). Sample n = step count - warmup.

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

| Stage | Expected delta vs vanilla DDP / FSDP | Interpretation |
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
