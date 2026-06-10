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
| `concurrent.ConcurrentOuterStep` | Orchestration layer above GraceWindowSyncer; asyncio-task-based; lets the cross-rack outer-step run concurrently with inner-step async-TP. Wraps the published Decoupled DiLoCo control law without modifying it. Patent-independent. | submit_async / collect / state-machine IDLE/PENDING/READY/FAILED |
| `reducer.GraceWindowSyncer` | Decoupled DiLoCo, arXiv:2604.21428 | minimum quorum, adaptive grace window, token-weighted merge |
| `desync_optimizer.DesynchronizedSyncSchedule` | DES-LOC, arXiv:2505.22549 | params sync every N inner steps; momenta sync every M >= N |
| `async_tp.enable_async_tp` | PyTorch / TorchTitan async-TP, September 2024 | best-effort enabling of TorchTitan's intra-node async-TP path |
| `failslow.FailSlowDetector` | FALCON, arXiv:2410.12588 | sliding-window z-score detection of slow ranks |
| `autotuner.RuntimeAutotuner` | Guard (arXiv:2605.17879) online performance monitoring; "From Detection to Recovery" (arXiv:2605.09370) operational recovery-wait analysis | online (continuous) adaptation of the fail-slow detection threshold and the grace-window wall-clock wait from the observed step-time stream |
| `compression.apply_compression` | INT8 quantization, PowerSGD (arXiv:1905.13727), SparseRL-Sync (arXiv:2605.07330) | optional outer-step delta transforms; `none` and `sparse` are lossless, while `int8` and `powersgd` are lossy opt-in modes |
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

- LoRA-adapter-granularity reduction. The companion `tsugi-kpool` covers that; this SDK operates at full-parameter granularity.
- Variance-threshold convergence triggers. That belongs to the Infinity patent estate; this SDK uses the grace-window trigger from Decoupled DiLoCo instead.
- K-of-N adapter routing. That belongs to the K-Pool LoRA patent estate; this SDK does not select a subset of model components per step.
- Custom C++ NCCL ProcessGroup. Python-level integration is sufficient for the current roadmap.
- Custom multi-rack reducer *topology* (hierarchical / tree all-reduce across many racks). The GraceWindowSyncer is world-size-aware: pass `start_round(round_id, expected_learner_ids=...)` and it finalizes early the moment every expected, non-fail-slow learner has reported (`MergeResult.reason == "all_present"`) instead of always waiting out the grace window, and it reports which expected learners were absent at finalize (`MergeResult.learners_absent`). What remains future work is a *hierarchical* aggregation topology (e.g. per-rack pre-reduction feeding a top-level merge) for very large rack counts; the current merge is still a single flat token-weighted aggregation.

The live `mend_init` runtime path can now use that world-size awareness through
`MendConfig.expected_learner_ids`. When set, the tuple is passed from
the runtime container into `ConcurrentOuterStep`, then into
`GraceWindowSyncer.start_round(expected_learner_ids=...)` for each submitted
outer round. The roster identifiers are the same strings carried on incoming
`LearnerFragment.learner_id` values from the fragment provider. Leave the config
field as `None` when the runtime cannot name the complete learner set. If an
unexpected learner id is observed before a round completes, or if the declared
roster cannot satisfy the configured quorum, the orchestrator restarts that
round in roster-unaware mode so the result matches the historical
quorum-then-full-grace path for the same fragment arrivals.

## Online runtime autotuner (opt-in, default OFF)

`autotuner.RuntimeAutotuner` continuously adapts two operational knobs from the runtime's own observed per-rank step-time stream, past a short warmup window. It is gated behind `MendConfig.auto_tune_runtime` (default `False`), so default-mode behavior is byte-for-byte unchanged.

- **Fail-slow detection sensitivity.** The effective z-score threshold tracks the cluster's live jitter: a jittery cluster gets a higher threshold (fewer false straggler flags under benign jitter), a clean cluster relaxes the threshold toward its floor (a real fail-slow is still caught). The jitter estimate is computed robustly (the single largest sample is excluded from the coefficient-of-variation) so a lone transient straggler does not inflate the threshold high enough to mask itself. This is the lightweight online performance-monitoring idea from Guard.
- **Grace-window wait.** The effective wall-clock `grace_window_ms` widens when a sustained straggler is observed (recent peak/median step-time ratio high) so the syncer can absorb the laggard, and narrows back toward the static baseline when the cluster is clean. This is the operational recovery-wait heuristic from "From Detection to Recovery".

The effective values and the steps on which they changed are surfaced as `auto_tune_runtime_decision` diagnostic events (and `auto_tune_runtime_active` on `mend_init`), consistent with the existing `failslow_decision` / `auto_tune_sync_period_decided` surfacing.

**Why this preserves bit-exact loss equivalence (both OFF and ON).** The detection threshold is observe-only: the detector's decision is a diagnostic flag, no mitigation/exclusion is wired off it here, so adapting the threshold changes only which steps are flagged, never any tensor value. The grace window is a wall-clock wait only: in default (lossless) mode the syncer waits for the same fragments regardless of the wait length, and the merged delta is the token-weighted merge of the same fragment set applied at the same logical boundary, so adapting the wait changes timing/overlap, never which fragments merge or the apply boundary. The autotuner deliberately does **not** online-adapt the merge cadence (`sync_period_steps` / momentum cadence / apply lag): a paired baseline-vs-sdk run has different step times on the two paths (the SDK overlaps the merge), so adapting cadence from measured step times would make the two paths choose different cadences and break bit-exactness. (Rank exclusion / fail-slow mitigation and outer-momentum restarting are separate, deferred concerns and are not part of this autotuner.)

## Optional compression modes

`MendConfig.outer_step_compression_mode` defaults to `none`, which clones the dense delta and preserves the established bit-exact path. `int8` and `powersgd` remain lossy experimental modes and are off by default.

The `sparse` mode is different: it is lossless. Each tensor is encoded as flattened int64 indices plus exact values only when that sparse payload is estimated to be smaller than the dense tensor payload; otherwise it falls back to dense. Decoding reconstructs the original dense tensor bit-for-bit before merge, including negative zero and non-finite values.

This mode only helps when the transmitted delta is genuinely element-sparse, such as adapter-heavy or sparse-update regimes. The default DiLoCo-style `params_delta` after many inner optimizer steps is typically dense, so the sparse codec is expected to choose dense fallback there rather than reduce communication.

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
