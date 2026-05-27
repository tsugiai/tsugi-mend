# Benchmark Results

Each subdirectory is a public-safe result bundle produced by
`benchmarks/run_paired.py`.

The committed `cpu_gloo_toy` bundle is the `$0` reproducibility cell. It runs on
local CPU with gloo and verifies the benchmark harness, result schema, and
bit-exact loss assertion without provisioning paid hardware.

Real cross-rack bundles can be added later only after a founder-authorized run
and public-safe scrub. They should use the same JSON schema so cheap local and
real-cluster cells can be compared by tooling without changing formats.
