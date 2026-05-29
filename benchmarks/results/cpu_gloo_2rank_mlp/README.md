# Result bundle: `cpu_gloo_2rank_mlp`

This is the **$0 cheap-reproducible** paired-run cell. Anyone with the repo
and `pip install -e ".[dev]"` can re-run it on a laptop CPU at no cost and
get a real bit-exact PASS plus an overlap-driven tokens/s uplift with a
bootstrap 95% CI.

## Reproduce

```bash
python benchmarks/run_paired.py --cell cpu_gloo_2rank_mlp
```

This spawns two `gloo` (CPU) worker processes that each run the SAME tiny
MLP on the SAME seed and synthetic data, syncing parameters across the two
ranks every `sync_period_steps` inner steps through the SDK's
Decoupled-DiLoCo `token_weighted_merge` reducer, under two synchronization
paths:

- **baseline** drives the `GraceWindowSyncer` synchronously, so the
  training thread blocks across the simulated cross-rack merge delay;
- **sdk** drives the `ConcurrentOuterStep` orchestrator (via `mend_init`
  with `concurrent_outer_step=True`), so the merge runs on the asyncio loop
  thread and the training thread overlaps the delay with inner-step compute.

Both paths apply the SAME merged delta at the SAME inner step
(`apply_lag_steps` after the round opens), so the parameter trajectories
coincide and the per-step loss is identical. The only difference is whether
the merge delay is **blocked on** (baseline) or **overlapped** (SDK).

## What the bundle records

`result.json` follows the bundle schema (see `benchmarks/README.md`):

- `hardware` -- generic platform / torch / backend labels (no hostnames,
  IPs, or paths).
- `workload` -- the synthetic-MLP shape, optimizer, seed.
- `sdk_config` -- the relevant `MendConfig` fields.
- `bit_exact_loss_equivalence` -- the load-bearing PASS: elementwise
  IEEE-754 equality of the two per-step loss trajectories. `max_abs_loss_diff`
  is `0.0`.
- `metrics` -- steady-state tokens/s for both paths plus the SDK-over-baseline
  uplift with a paired-bootstrap 95% CI (10000 resamples).

## How to read this honestly

- **Bit-exact is the robust, load-bearing result.** `max_abs_loss_diff` is
  exactly `0.0` and reproduces on every run. If a future change to the
  default (lossless) path broke bit-exactness, this cell would FAIL and exit
  non-zero.
- **The tokens/s uplift point estimate is noisy on a CPU.** It is a real
  measurement of the overlap mechanism, but absolute CPU step times vary with
  OS scheduling, so the committed number is one observed run, not a stable
  population mean. The committed `result.json` records +14.52% with CI95
  [+3.41%, +25.26%]. Re-runs may move within that band or outside it on a
  different CPU, OS scheduler state, or torch build. Report the CPU number with
  its CI, never as a bare point estimate.
- **This is a NEW cheap result, not a restatement of any headline number.**
  It is a synthetic MLP on CPU under an injected merge delay. It demonstrates
  that the overlap mechanism produces a measurable, bit-exact-preserving
  speedup; it is not a production cross-rack measurement and does not stand in
  for one.

The committed `result.json` is one such run. Re-running overwrites it with a
fresh measurement of the same shape.
