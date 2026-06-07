# Architecture

## Layers

```
+--------------------------------------------------------------+
| user training script (transformers / accelerate / torchrun)  |
+--------------------------------------------------------------+
| MendConfig + mend_init / mend_shutdown                          |  <- public API
+--------------------------------------------------------------+
| ConcurrentOuterStep      DesynchronizedSyncSchedule          |
|  (Phase 2 wk1: orchestrator on top of GraceWindowSyncer)     |
| GraceWindowSyncer          (DES-LOC)                         |
|  (Decoupled DiLoCo)                                          |
| FailSlowDetector         RuntimeAutotuner                    |
|  (FALCON)                  (online scheduling controls)      |
| async_tp.enable_async_tp  Topology classifier                 |
|  (TorchTitan probe)        (rack-aware DP-last)              |
| Sideband (asyncio + TCP)  DiagnosticsWriter                  |
|  (control plane)           (JSONL event log)                 |
+--------------------------------------------------------------+
| torch.distributed (NCCL) | OS TCP stack                      |
+--------------------------------------------------------------+
```

## Public-art mapping (NO patent claim exercised)

| Component | Public reference | What it implements |
|---|---|---|
| `concurrent.ConcurrentOuterStep` | Orchestration layer above GraceWindowSyncer; asyncio-task-based; lets the cross-rack outer-step run concurrently with inner-step async-TP. Wraps the published Decoupled DiLoCo control law without modifying it. Patent-independent. | submit_async / collect / state-machine IDLE/PENDING/READY/FAILED |
| `reducer.GraceWindowSyncer` | Decoupled DiLoCo, arXiv:2604.21428 | minimum quorum, adaptive grace window, token-weighted merge |
| `desync_optimizer.DesynchronizedSyncSchedule` | DES-LOC, arXiv:2505.22549 | params sync every N inner steps; momenta sync every M >= N |
| `async_tp.enable_async_tp` | PyTorch / TorchTitan async-TP, September 2024 | best-effort enabling of TorchTitan's intra-node async-TP path |
| `failslow.FailSlowDetector` | FALCON, arXiv:2410.12588 | sliding-window z-score detection of slow ranks |
| `autotuner.RuntimeAutotuner` | Guard, arXiv:2605.17879; operational recovery analysis, arXiv:2605.09370 | optional, default-off adaptation of fail-slow detector threshold and grace-window wait |
| `topology.detect` | generic engineering | rack classification from NCCL_TOPO_FILE or hostname grouping |
| `sideband.Sideband` | generic engineering | low-bandwidth TCP heartbeat carrying step-id / vector-clock / queue-depth / health metadata |
| `diagnostics.DiagnosticsWriter` | generic engineering | append-only JSONL event log |

The runtime composes public-art and generic engineering components end-to-end:

1. Intra-rack TP/CP/PP/FSDP traffic is unmodified vanilla NCCL.
2. The DES-LOC schedule decides each step whether to sync params, momenta, or neither.
3. Cross-rack syncs go through the GraceWindowSyncer (Decoupled DiLoCo Algorithm 2).
4. The fail-slow detector observes step times and emits diagnostics. In 0.1.x the detector is observe-only; rank exclusion is not wired into the runtime.
5. When `MendConfig.auto_tune_runtime=True`, the RuntimeAutotuner continuously adapts the effective fail-slow z-score threshold and the GraceWindowSyncer wall-clock grace wait from observed step-time statistics. It never changes merge cadence, merge math, quorum membership, or rank exclusion.
6. The sideband carries rack-level progress metadata as a control plane separate from the NCCL data plane.

None of these mechanisms exercise TsugiCinema's K-Pool LoRA (App. 64/060,315) or Infinity (App. 64/055,093) patents. See the `LICENSE` preamble.

## What is intentionally not here

- LoRA-adapter-granularity reduction. The companion `tsugi-kpool` covers that; this SDK operates at full-parameter granularity.
- Variance-threshold convergence triggers. That belongs to the Infinity patent estate; this SDK uses the grace-window trigger from Decoupled DiLoCo instead.
- K-of-N adapter routing. That belongs to the K-Pool LoRA patent estate; this SDK does not select a subset of model components per step.
- Runtime-driven fail-slow rank exclusion. The detector and autotuner can report slow-rank diagnostics, but mitigation that drops a rank from quorum is intentionally deferred.
- Custom C++ NCCL ProcessGroup. Python-level integration is sufficient for the current roadmap.
- Multi-rack 3+ rack reducer optimization. The current GraceWindowSyncer handles N >= 2 racks; production 4+ rack support is future work.

## File-by-file reading order

For someone joining this codebase, read in this order:

1. `src/tsugi_mend/config.py` -- every knob and what it does.
2. `src/tsugi_mend/reducer.py` -- the Decoupled DiLoCo control law.
3. `src/tsugi_mend/desync_optimizer.py` -- the DES-LOC schedule.
4. `src/tsugi_mend/failslow.py` -- the FALCON detector.
5. `src/tsugi_mend/autotuner.py` -- optional runtime tuning of detector sensitivity and grace wait.
6. `src/tsugi_mend/topology.py` -- rack classification.
7. `src/tsugi_mend/sideband.py` -- control-plane metadata channel.
8. `src/tsugi_mend/runtime.py` -- how the layers compose at runtime.
9. `docs/benchmark_protocol.md` -- the cross-rack benchmark protocol.

## Companion SDK boundary

The patent-aligned `tsugi-kpool` exists at a different layer:

| Concept | tsugi-mend | tsugi-kpool |
|---|---|---|
| Granularity | full-model parameters | LoRA adapter parameters only |
| Convergence trigger | Decoupled DiLoCo grace window | variance threshold on adapter snapshots |
| Aggregator | GraceWindowSyncer | BufferConvergenceAggregator |
| Sideband payload | rack-level progress | per-adapter buffer fill |
| License | Apache-2.0 (full patent grant) | Apache-2.0 (patent grant extends to K-Pool LoRA + Infinity as practiced by SDK code) |
| Patent estate | none exercised | K-Pool LoRA + Infinity |

The SDKs share zero code. They can be installed side by side under the same Python environment if needed.
