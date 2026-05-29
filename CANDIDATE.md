# Task 15 candidate evidence

## Self-spawn cheap cell

Command:

```bash
uv run --extra dev python benchmarks/run_paired.py --cell cpu_gloo_2rank_mlp --no-write
```

Output:

```text
================================================================
cell: cpu_gloo_2rank_mlp  (cheap)
hardware: local CPU (gloo); $0 cheap reproducible cell
bit-exact loss equivalence (default mode): PASS  (max |loss diff| = 0.000e+00 over 160 steps)
baseline tokens/s: 13290110.7  (mean step 9.86 ms)
sdk      tokens/s: 17954079.6  (mean step 7.30 ms)
uplift: +35.09%  (95% CI [+16.23%, +55.14%], n=140 paired steps, 10000 resamples)
================================================================
```

## Torchrun cheap cell

Command:

```bash
uv run --extra dev torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 benchmarks/run_paired.py --launch torchrun --cell cpu_gloo_2rank_mlp --no-write
```

Output:

```text
W0528 23:53:31.764000 19562 torch/distributed/elastic/multiprocessing/redirects.py:35] NOTE: Redirects are currently not supported in MacOs.
W0528 23:53:31.802000 19562 torch/distributed/run.py:862]
W0528 23:53:31.802000 19562 torch/distributed/run.py:862] *****************************************
W0528 23:53:31.802000 19562 torch/distributed/run.py:862] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0528 23:53:31.802000 19562 torch/distributed/run.py:862] *****************************************
================================================================
cell: cpu_gloo_2rank_mlp  (cheap)
hardware: local CPU (gloo); $0 cheap reproducible cell
bit-exact loss equivalence (default mode): PASS  (max |loss diff| = 0.000e+00 over 160 steps)
baseline tokens/s: 13866690.5  (mean step 9.45 ms)
sdk      tokens/s: 19351712.5  (mean step 6.77 ms)
uplift: +39.56%  (95% CI [+17.93%, +63.45%], n=140 paired steps, 10000 resamples)
================================================================
```
