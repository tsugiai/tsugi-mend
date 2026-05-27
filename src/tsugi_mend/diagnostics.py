"""JSONL diagnostics writer for the runtime.

One file per process under `diagnostics_dir`. Benchmark tables and plots
are generated from these files by the end-of-run analysis script.

Schema (one JSON object per line):
    {
      "ts": <unix seconds, float>,
      "event": <string>,
      ... <event-specific fields> ...
    }

Standard event names emitted by the runtime:
    "mend_init"           runtime started
    "mend_shutdown"       runtime stopped
    "outer_round"        a Decoupled DiLoCo outer round completed
    "failslow_decision"  per-rank fail-slow decision after a step
    "sideband_snapshot"  periodic peer-progress snapshot
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DiagnosticsWriter:
    """Append-only JSONL writer. One line per event."""

    def __init__(self, diagnostics_dir: str | None) -> None:
        self._enabled = diagnostics_dir is not None
        if not self._enabled:
            return
        path = Path(diagnostics_dir) if diagnostics_dir else None
        assert path is not None
        path.mkdir(parents=True, exist_ok=True)
        self._fh = open(path / f"max_sdk_pid{os.getpid()}.jsonl", "a")

    def emit(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        record = {"ts": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._enabled:
            self._fh.close()
