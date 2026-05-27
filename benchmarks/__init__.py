"""Public-safe benchmark harness for tsugi-mend.

This package implements the paired-run protocol documented in
``docs/benchmark_protocol.md``:

- ``metrics``   pure helper functions (bit-exact check, bootstrap CI,
                steady-state tokens/s summary). Unit-tested.
- ``run_paired`` config-driven paired-run driver (baseline vs SDK) that
                scales from a $0 CPU/gloo cheap cell up to a real
                multi-node config by command-line args only.

The harness is additive: it does not change SDK runtime behavior, the
public API, or CI. See ``benchmarks/README.md`` for how to run the cheap
cell and how the result bundles are shaped.
"""
from __future__ import annotations

__all__ = ["metrics", "run_paired"]
