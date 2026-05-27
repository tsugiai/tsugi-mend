# Phase 2 Week 1: Concurrent async-TP overlap with cross-rack reducer

Design note for the concurrent outer-step orchestrator.

## Goal

Today, the cross-rack reducer (`reducer.GraceWindowSyncer`) blocks the training loop during its grace window. While the syncer waits for slower learners (up to `grace_window_ms`, default 2000ms), the local rank's GPUs sit idle. On Hopper-class racks this idle time is the largest remaining single-stage source of throughput loss in the cross-rack pipeline.

Concurrent async-TP overlap closes that gap: the runtime continues issuing inner-step forward / backward / async-TP-overlapped collectives while the outer-step cross-rack reduce-scatter runs in parallel on a separate asyncio task (and, on real GPU clusters, a separate CUDA stream). The merged outer-step delta lands at the next sync-period boundary instead of immediately, which is the same convergence guarantee Decoupled DiLoCo Algorithm 2 already gives (the merge is between two staggered inner-step blocks; the merge does not have to apply mid-block).

Expected uplift on top of the Stage D-proper +28.58% baseline: 5-12 percent on TP-using models (OpenAI analyst estimate), 15-20 percent compounding on GB200 NVL72 (Gemini estimate). Patent-independent.

## Non-goals

- This week does not introduce gradient compression (PowerSGD or INT8 quantized all-reduce). That is the weeks 10-12 stretch item and is feature-flag-gated to preserve the bit-exact-default-mode marketing anchor.
- This week does not change the GraceWindowSyncer Algorithm 2 control law (quorum + grace + token-weighted merge). The concurrency is a new orchestration layer ABOVE the existing syncer.
- This week does not exercise the K-Pool LoRA or Infinity patent estates. Variance-threshold triggers, K-of-N adapter routing, and elastic gradient buffers belong to the kpool-sdk.

## Design

### Where the concurrency lives

```
Vanilla today:                  Phase 2 Week 1:
  inner_step()                    inner_step()
  if outer_sync_boundary:          inner_step()
    BLOCK on syncer.tick()         inner_step()
    apply(merged_delta)            if outer_sync_boundary:
  inner_step()                       outer_step_begin_async()  // returns immediately
  inner_step()                     inner_step()                  // GPU stays busy
                                   inner_step()                  // GPU stays busy
                                   inner_step()                  // GPU stays busy
                                   if outer_step_collect() is not None:
                                     apply(merged_delta)         // applied late by 1-3 inner steps
                                   inner_step()
```

The new orchestrator submits the outer-step into an asyncio task on the same event loop the sideband already uses. It exposes two methods:

```python
class ConcurrentOuterStep:
    def submit_async(round_id, local_fragment_provider) -> None: ...
    def collect() -> Optional[MergeResult]: ...
```

The orchestrator does not block. `collect()` is non-blocking and returns `None` while the merge is in flight; `submit_async()` returns immediately and schedules the syncer.tick() loop on the asyncio event loop.

The runtime calls `outer_step_begin()` at each outer-sync boundary and `outer_step_collect()` at each inner step. The merged delta is applied to the local model when `collect()` returns a non-None result; this is at most `grace_window_ms` of inner-step latency late.

### Convergence guarantee

Decoupled DiLoCo Algorithm 2 already merges between staggered inner-step blocks. The orchestrator's late-apply by D inner steps (D typically in {1..8} in measured workloads, bounded by ceil(grace_window_ms / T_step) in the worst case) is structurally equivalent to running the syncer with offset t_p and the learner with offset t_p + D, both within the valid {0, ..., H-1} offset range that Algorithm 2 explicitly allows. The merge operation itself is identical (same learner snapshots, same w_mp weights, same Theta_p^(t-H) base); only the broadcast-receive timing differs.

The full structural argument plus empirical-validation table (loss equivalence preserved across measured D range) is at `docs/convergence_equivalence_sketch.md` (2026-05-23). The DiLoCo paper itself does NOT contain a closed-form convergence theorem with explicit staleness bound; its convergence evidence is empirical (Tables 2 and 3 simulate hardware-failure rates and show maintained ML performance). The orchestrator's late-apply lies inside the same empirical regime the paper validates, plus extends slightly beyond it in a direction the algorithm is designed to tolerate.

The orchestrator should refuse to operate in the corner case where D >= H (delta applied beyond the syncer's next sync boundary); the proposed `auto_tune_sync_period_min` MendConfig knob is the load-bearing safety bound.

### CUDA stream design (Stage B+; out of scope for Stage A)

On real GPUs, the outer-step reduce-scatter must run on a CUDA stream distinct from the inner-step's forward/backward so the device actually overlaps the two. The Stage A unit tests do not exercise CUDA streams (CPU-only); the Stage B+ benchmarks validate the device-level overlap. Add a `outer_step_cuda_stream_priority: int = 0` config knob for tuning; default 0 (no priority preference). Stage C and D-proper benchmark sweeps will probe the priority effect.

### Failure handling

- If `submit_async()` is called before the previous outer-step has been collected, raise `RuntimeError` (orchestration error; the caller must `collect()` first or wait).
- If the orchestrator's asyncio task raises (e.g., the GraceWindowSyncer rejects a fragment via FailSlow), the exception is captured and re-raised on the next `collect()` call. This gives the caller deterministic exception ordering (the exception is visible at the merge boundary, not the submit boundary).
- If the orchestrator detects that the asyncio event loop has gone away (sideband shutdown raced), the next `collect()` returns a `MergeResult` with `reason="event_loop_lost"` and `learners_merged=[]`. The runtime treats this as a fail-open and continues training with the local rank's params.

### Config knobs added in Week 1

```python
@dataclass
class MendConfig:
    # ... existing fields ...

    # Phase 2 Week 1: concurrent async-TP overlap with cross-rack reducer.
    concurrent_outer_step: bool = True
    # CUDA stream priority for the outer-step reduce-scatter when running
    # concurrently. 0 is "default" priority; negative values are higher
    # priority. Stage B+ benchmarks validate the device-level overlap.
    outer_step_cuda_stream_priority: int = 0
```

### Test plan

- **Stage A (CPU, this week):** new `test_concurrent_outer_step.py` exercising the orchestrator with a deterministic clock. Drives 4 learners, submits fragments with controlled timing, verifies that:
  - `submit_async()` returns immediately
  - `collect()` returns `None` before quorum + grace elapsed
  - `collect()` returns the correct `MergeResult` after the grace window
  - Double-submit raises `RuntimeError`
  - Asyncio task exceptions are deterministically re-raised on `collect()`
- **Stage B (single A10, week 1 day 5-7):** smoke test on Lambda Labs with one A10 and a synthetic two-learner topology. Confirm the orchestrator does not introduce regression vs vanilla outer-step.
- **Stage C (8x V100, week 2 day 3-4):** paired-runs benchmark comparing concurrent-outer-step ON vs OFF on the existing Stage C protocol. PASS criterion: zero loss-equivalence regression; tokens-per-second uplift >=3% (small expected lift because Stage C is single-instance and cross-rack is simulated).
- **Stage D-proper rerun (week 3):** paired-runs on real 2-node 8x V100 cross-network with concurrent-outer-step ON vs OFF. PASS criterion: tokens-per-second uplift >=5% over the existing +28.58% baseline. Bit-exact loss equivalence preserved.

## Risk register

- **Asyncio race conditions:** the orchestrator and the sideband share an event loop. Wrong-order operations (collect-before-submit, submit-before-start) must raise clear errors, not hang. Mitigation: explicit state machine in `ConcurrentOuterStep` with assertions at every transition.
- **CUDA stream priority misconfiguration:** Stage A does not exercise streams. If Stage C reveals that priority-0 is the wrong default on H100 / GB200, the config knob can be tuned without code change.
- **DES-LOC interaction:** the DES-LOC schedule already has the property that momenta sync less frequently than parameters. The concurrent outer-step must preserve this; specifically, the orchestrator must not accidentally cause a parameter-sync to occur without a momentum-sync at the same boundary. Mitigation: the schedule is queried at `outer_step_begin()`, not at `collect()`; the late-apply preserves the ordering.
- **Convergence drift on long runs:** the late-apply means the rank's params during inner-steps N+1, N+2, N+3 are slightly stale (have not yet received the merged outer-step delta). This is the expected DiLoCo behavior; the runs validate convergence empirically. PASS criterion: bit-exact loss equivalence at the Stage D-proper run boundaries.
- **Stage A.6-style FSDP gotcha:** the `tsugiai-kpool-sdk` Stage A.6 work surfaced six real production-path bugs around FSDP bucket-view gradients and per-parameter hooks. The mend-sdk does not use per-parameter hooks (it operates on full-parameter granularity), but FSDP+NCCL composition is still a place where the unified product surface has historically had unexpected interactions. Mitigation: re-run the existing 51-test pytest suite at each commit; add the concurrent-outer-step coverage incrementally.

## Day-by-day plan

- **Day 1 (2026-05-22):** design doc (this file). New `concurrent_outer_step` config knob plus first test stub. Commit.
- **Day 2-3:** implement `ConcurrentOuterStep` orchestrator in `concurrent.py`. Wire into `_MaxRuntime`. Update existing tests where they touch the runtime hooks. Commit at end of each day.
- **Day 4:** Stage A pass on M1 Max local CPU; pytest 51 + new tests. Commit when green.
- **Day 5-7:** Stage B smoke test on Lambda Labs 1x A10 (~$1-2 spend). Validate no regression vs vanilla outer-step. Commit results.
- **Week 2 day 3-4:** Stage C paired-runs benchmark. Commit results.
- **Week 3:** Stage D-proper paired-runs. Commit results plus final Phase 2 Week 1-3 summary.

## Companion changes outside this module

- `docs/architecture.md`: add a new entry under "Layers" for the concurrent-outer-step orchestrator. Not a deep change.
- `docs/benchmark_protocol.md`: add the concurrent-outer-step ON vs OFF paired-runs protocol for Stage C and Stage D-proper.

No changes required to:
- `desync_optimizer.py` (DES-LOC schedule is consulted at `outer_step_begin()` boundary; behavior unchanged)
- `failslow.py` (FALCON detector continues to mark stragglers per-rank; the orchestrator queries `failslow.is_failslow(learner_id)` at submission)
- `topology.py` (rack-aware classification is read-only here)
- `sideband.py` (the orchestrator uses the existing sideband event loop; no new sideband fields)
- `reducer.py` (GraceWindowSyncer is the inner state machine; orchestrator wraps it without modification; preserves the patent-independence note about the variance-threshold trigger)

End of Phase 2 Week 1 design.
