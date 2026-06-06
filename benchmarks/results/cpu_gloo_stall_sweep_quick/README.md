# Result bundle: `cpu_gloo_stall_sweep_quick`

This is the `$0` quick smoke for the seeded peer-straggler sweep. It runs a
small synthetic MLP on two local CPU/gloo ranks and emits an
uplift-vs-injected-stall result bundle. The quick grid is intentionally small
but spans a no-stall point and a meaningful stall point so the smoke shows the
honest curve shape in one cheap run:

- injected peer-straggler delay: `{0, 100}` ms
- peer-straggler count: `{0, 1}`
- `n_seeds=5` per grid point (baseline/sdk order alternated across seeds to
  interleave drift)
- run-level aggregation: drop the single lowest and single highest seed-level
  uplift, then report mean, sample variance, and bootstrap CI95 across the
  surviving three seed-level uplifts

The workload is sized so a single inner step costs a few milliseconds on CPU.
That keeps the uplift numbers interpretable: in a sub-millisecond-step regime
the measurement is dominated by orchestration overhead and the smoke prints
noisy negative uplift, which is not informative.

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

Output (wall-clock timing varies run to run; bit-exact is invariant):

```text
======================================================================================
cell: cpu_gloo_stall_sweep_quick  (quick stall sweep)
bit-exact loss equivalence: PASS (max |loss diff| = 0.000e+00)
delay=   0 ms  stragglers=0  bit_exact=PASS  uplift= +12.03% CI95 [  +5.89%,  +17.97%]  p95 baseline/sdk=8.78/4.01 ms
delay=   0 ms  stragglers=1  bit_exact=PASS  uplift=  +9.52% CI95 [  +5.26%,  +16.86%]  p95 baseline/sdk=8.98/4.16 ms
delay= 100 ms  stragglers=0  bit_exact=PASS  uplift=  +8.05% CI95 [  -1.96%,  +14.14%]  p95 baseline/sdk=8.95/4.11 ms
delay= 100 ms  stragglers=1  bit_exact=PASS  uplift= +10.37% CI95 [ +10.14%,  +10.64%]  p95 baseline/sdk=116.69/100.94 ms
======================================================================================
wrote bundle: benchmarks/results/cpu_gloo_stall_sweep_quick/result.json
```

## How to read this honestly

- Bit-exact loss equivalence is the load-bearing result. Every seeded paired
  run in this quick grid passed with `max_abs_loss_diff = 0.0`.
- The quick CPU uplift is a smoke measurement, not a production claim. It is a
  small synthetic model on two CPU ranks; production-relevant numbers come from
  the GPU/InfiniBand cells, not this smoke.
- The `0 ms` rows are NOT a zero-work case here: even with no injected
  straggler, the SDK overlaps the real cross-rack merge collective with inner
  compute, so the uplift is modestly positive. That is a merge-overlap benefit,
  not a stall-recovery benefit.
- The `100 ms, 1 straggler` row is the straggler check: the baseline p95 jumps
  (here ~117 ms) because the fast ranks block in the collective on the slow
  peer, while the SDK path (here ~101 ms) overlaps part of the wait with inner
  steps. The overlap can only hide as much stall as there is concurrent inner
  compute, so the recovery is bounded, not unlimited. The full grid
  (`straggler_delay_ms` in `{0, 50, 100, 250, 500, 1000}` x peer-straggler count
  `{0, 1, 2, 4}`) maps the whole uplift-vs-stall curve.
- The `failslow_observe_only` blocks record `FailSlowDetector` observations for
  the per-rank timing series. The detector observes ALL step-time anomalies,
  including the natural step-time spike at each merge boundary, so it can flag
  ranks even at zero injected delay (that flag reflects the merge-boundary
  spike, not an injected straggler). It never calls rank exclusion, alters
  quorum, or changes any numerical behavior.
