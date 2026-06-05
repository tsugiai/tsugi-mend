# Result bundle: `cpu_gloo_stall_sweep_quick`

This is the `$0` quick smoke for the seeded peer-straggler sweep. It runs a
small synthetic MLP on two local CPU/gloo ranks and emits an
uplift-vs-injected-stall result bundle. The quick grid is intentionally small:

- injected peer-straggler delay: `{0, 50}` ms
- peer-straggler count: `{0, 1}`
- `n_seeds=5` per grid point
- run-level aggregation: drop the single lowest and single highest seed-level
  uplift, then report mean, sample variance, and bootstrap CI95 across the
  surviving three seed-level uplifts

## Reproduce

```bash
python benchmarks/run_stall_sweep.py --quick
```

This command writes:

```text
benchmarks/results/cpu_gloo_stall_sweep_quick/result.json
```

## Observed output

This committed bundle was produced by:

```bash
uv run python benchmarks/run_stall_sweep.py --quick
```

Output:

```text
======================================================================================
cell: cpu_gloo_stall_sweep_quick  (quick stall sweep)
bit-exact loss equivalence: PASS (max |loss diff| = 0.000e+00)
delay=   0 ms  stragglers=0  bit_exact=PASS  uplift= -11.74% CI95 [ -42.46%,  +13.80%]  p95 baseline/sdk=0.94/1.30 ms
delay=   0 ms  stragglers=1  bit_exact=PASS  uplift=  -5.71% CI95 [ -41.39%,  +17.24%]  p95 baseline/sdk=0.84/1.20 ms
delay=  50 ms  stragglers=0  bit_exact=PASS  uplift=  -5.66% CI95 [ -38.33%,  +12.81%]  p95 baseline/sdk=0.84/1.22 ms
delay=  50 ms  stragglers=1  bit_exact=PASS  uplift=  +3.96% CI95 [  -3.72%,   +7.83%]  p95 baseline/sdk=14.57/14.48 ms
======================================================================================
wrote bundle: benchmarks/results/cpu_gloo_stall_sweep_quick/result.json
```

## How to read this honestly

- Bit-exact loss equivalence is the load-bearing result. Every seeded paired
  run in this quick grid passed with `max_abs_loss_diff = 0.0`.
- The quick CPU uplift is a smoke measurement, not a production claim. The
  quick model is deliberately small, and the zero-delay rows show local
  scheduling noise and orchestration overhead.
- The `50 ms, 1 straggler` row is the smoke check that the injected peer wait is
  present and that the SDK path can overlap part of it while preserving
  bit-exact loss.
- The `failslow_observe_only` blocks record `FailSlowDetector` observations for
  the per-rank timing series. They do not call rank exclusion, alter quorum, or
  change any numerical behavior.
