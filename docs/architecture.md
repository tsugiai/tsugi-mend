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
| FailSlowDetector         async_tp.enable_async_tp            |
|  (FALCON)                  (TorchTitan probe)                |
| Topology classifier      Sideband (asyncio + TCP)            |
|  (rack-aware DP-last)      (control plane)                   |
+--------------------------------------------------------------+
| torch.distributed (NCCL) | OS TCP stack                      |
+--------------------------------------------------------------+
```

## Public-art mapping (NO patent claim exercised)

| Component | Public reference | What it implements |
|---|---|---|
| `concurrent.ConcurrentOuterStep` | Phase 2 Week 1 (2026-05-22): orchestration layer above GraceWindowSyncer; asyncio-task-based; lets cross-rack outer-step run concurrently with inner-step async-TP. Wraps the published Decoupled DiLoCo control law without modifying it. Patent-independent. | submit_async / collect / state-machine IDLE/PENDING/READY/FAILED |
| `reducer.GraceWindowSyncer` | Decoupled DiLoCo, arXiv:2604.21428 | minimum quorum, adaptive grace window, token-weighted merge |
| `desync_optimizer.DesynchronizedSyncSchedule` | DES-LOC, arXiv:2505.22549 | params sync every N inner steps; momenta sync every M >= N |
| `async_tp.enable_async_tp` | PyTorch / TorchTitan async-TP, September 2024 | best-effort enabling of TorchTitan's intra-node async-TP path |
| `failslow.FailSlowDetector` | FALCON, arXiv:2410.12588 | sliding-window z-score detection of slow ranks |
| `topology.detect` | generic engineering | rack classification from NCCL_TOPO_FILE or hostname grouping |
| `sideband.Sideband` | generic engineering | low-bandwidth TCP heartbeat carrying step-id / vector-clock / queue-depth / health metadata |
| `diagnostics.DiagnosticsWriter` | generic engineering | append-only JSONL event log |

The control law composes the four public-art techniques end-to-end:

1. Intra-rack TP/CP/PP/FSDP traffic is unmodified vanilla NCCL.
2. The DES-LOC schedule decides each step whether to sync params, momenta, or neither.
3. Cross-rack syncs go through the GraceWindowSyncer (Decoupled DiLoCo Algorithm 2).
4. The fail-slow detector observes step times and marks slow ranks for exclusion from the current quorum round.
5. The sideband carries rack-level progress metadata as a control plane separate from the NCCL data plane.

None of these mechanisms exercise TsugiCinema's K-Pool LoRA (App. 64/060,315) or Infinity (App. 64/055,093) patents. See the `LICENSE` preamble.

## What is intentionally not here

- LoRA-adapter-granularity reduction. The companion `tsugiai-kpool-sdk` covers that; this SDK operates at full-parameter granularity.
- Variance-threshold convergence triggers. That belongs to the Infinity patent estate; this SDK uses the grace-window trigger from Decoupled DiLoCo instead.
- K-of-N adapter routing. That belongs to the K-Pool LoRA patent estate; this SDK does not select a subset of model components per step.
- Custom C++ NCCL ProcessGroup. Python-level integration is sufficient for the Stages B-E roadmap.
- Multi-rack 3+ rack reducer optimization. The current GraceWindowSyncer handles N >= 2 racks; Stage E targets 2 racks; production 4+ rack support is a Phase 2 productization track.

## File-by-file reading order

For someone joining this codebase, read in this order:

1. `src/tsugi_mend/config.py` -- every knob and what it does.
2. `src/tsugi_mend/reducer.py` -- the Decoupled DiLoCo control law.
3. `src/tsugi_mend/desync_optimizer.py` -- the DES-LOC schedule.
4. `src/tsugi_mend/failslow.py` -- the FALCON detector.
5. `src/tsugi_mend/topology.py` -- rack classification.
6. `src/tsugi_mend/sideband.py` -- control-plane metadata channel.
7. `src/tsugi_mend/runtime.py` -- how the layers compose at runtime.
8. `docs/benchmark_protocol.md` -- the cross-rack benchmark protocol.

## Companion SDK boundary

The patent-aligned `tsugiai-kpool-sdk` exists at a different layer:

| Concept | tsugiai-mend-sdk | tsugiai-kpool-sdk |
|---|---|---|
| Granularity | full-model parameters | LoRA adapter parameters only |
| Convergence trigger | Decoupled DiLoCo grace window | variance threshold on adapter snapshots |
| Aggregator | GraceWindowSyncer | BufferConvergenceAggregator |
| Sideband payload | rack-level progress | per-adapter buffer fill |
| License | Apache-2.0 (full patent grant) | Apache-2.0 (patent grant extends to K-Pool LoRA + Infinity as practiced by SDK code) |
| Patent estate | none exercised | K-Pool LoRA + Infinity |

The SDKs share zero code. They can be installed side by side under the same Python environment if needed.
