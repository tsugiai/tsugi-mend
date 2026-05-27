# Benchmarks

This directory contains public-safe reproducibility tooling for tsugi-mend.
It implements the paired-run contract in
[`docs/benchmark_protocol.md`](../docs/benchmark_protocol.md): same workload,
same seed, same model, same data, and different synchronization/control path.

The harness is additive and does not change SDK runtime behavior.

## Cheap reproducible cell

The committed `$0` cell runs on local CPU with the gloo backend:

```bash
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
```

Expected output shape:

```text
wrote benchmarks/results/cpu_gloo_toy/result.json
bit_exact=PASS baseline_tps=<float> sdk_tps=<float> uplift_pct=<float> ci95_pct=[<float>, <float>]
```

The cell uses a deterministic toy MLP and synthetic CPU batches. The baseline
path performs vanilla synchronous gradient averaging over gloo ranks. The SDK
path runs the same optimizer trajectory while wrapping the loop with
`MendConfig`, runtime hooks, and the concurrent outer-step control path using
synthetic zero-delta fragments. The zero-delta fragments exercise the control
path without changing the optimizer trajectory, so bit-exact loss equivalence is
checked directly rather than assumed.

This local result is useful for reproducibility plumbing and invariant checks.
It is not a production cross-rack measurement and should not be compared to the
multi-node headline cells in the README.

## Real-cluster cells

Future real-cluster cells should use the same driver shape and the same results
bundle schema, but with hardware-specific arguments such as rank count, model,
step count, backend, sync cadence, and hardware label. Real cross-rack results
must follow the protocol document: report steady-state tokens/s, rank-0 step
time percentiles, bit-exact loss status, and a bootstrap 95% confidence interval
over paired steady-state steps.

Do not commit internal hostnames, private paths, raw provider logs, or paid
cloud outputs into this public repository. Commit only public-safe summaries and
bundles.

## Result bundles

Result bundles live under `benchmarks/results/<cell>/result.json` and include:

- schema version and run label
- public-safe hardware/software metadata
- complete harness config
- bit-exact loss check
- baseline and SDK steady-state summaries
- bootstrap 95% CI for uplift
- per-step rank-0 metrics
