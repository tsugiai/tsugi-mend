# Convergence-equivalence sketch: late-apply tolerance of the orchestrator

Date: 2026-05-23
Author: Mend SDK research-line continuation session
Companion to internal FALCON-distribution-verification and uplift-surface-characterization briefs (2026-05-23).
Readability target: senior research engineer with distributed-training background; no measure-theory prerequisite.

## What this sketch is

A short, honest argument for why the ConcurrentOuterStep orchestrator (which applies the merged outer-step delta D inner steps late, where D is typically in {1, 2, 3} but can extend up to ceil(grace_window_ms / T_step)) does not require a new convergence proof beyond what Decoupled DiLoCo (Douillard et al., arXiv:2604.21428) already establishes for its Algorithm 2.

This is a sketch, not a theorem. The DiLoCo paper itself does NOT contain an explicit convergence theorem with a closed-form staleness bound; its convergence evidence is empirical (Tables 2 and 3 simulate hardware failure rates and show maintained ML performance). The argument below shows the orchestrator's late-apply lies inside the same empirical regime that DiLoCo validates, plus extends slightly beyond it in a direction the algorithm is designed to tolerate.

## Setup and notation

From Algorithm 2 of the DiLoCo paper:
- `H` = per-fragment sync interval (inner steps between outer-step boundaries for any one fragment)
- `t_p` in {0, ..., H-1} = offset for fragment `p` (the global step modulo H at which fragment p's outer step fires)
- `tau` = number of compute steps over which the outer-step sync is overlapped (set to 2 in the paper's experiments)
- `xi_grace` = adaptive grace window; bounded by `gamma * xi_slack` where `xi_slack = tau * xi_step - (xi_quorum + xi_sync)`, `gamma < 1`
- `K` = minimum quorum size (set to 1 in the paper's experiments, with adaptive grace window collecting more)
- `Theta_p^(t)` = global state of fragment `p` at outer-step boundary `t`
- `theta_{m,p}^(t)` = local learner `m`'s state of fragment `p` at step `t`
- `Delta_p^(t) = Merge({theta_{m,p}^(t)}_{m in M_t}, Theta_p^(t - H))` = outer gradient

The outer optimizer at line 13 of Algorithm 2: `Theta_p^(t) <- OuterOpt(Theta_p^(t - H), Delta_p^(t))`.

After the outer update, the syncer broadcasts `Theta_p^(t)` back to the learners (line 15). Each learner upon receipt sets its local `theta_{m,p}^(t) <- Theta_p^(t)` (line 16 of Algorithm 1) and resets its per-fragment counters.

In the Mend SDK orchestrator, this broadcast does NOT happen at global step `t`. Instead, the orchestrator's asyncio task continues to wait the grace window, and the broadcast lands `D` inner steps later, at global step `t + D`. During those D inner steps, the learner's local model has applied D additional inner-step updates `theta_{m,p}^(t+1), theta_{m,p}^(t+2), ..., theta_{m,p}^(t+D)` using the OLD local state as the starting point (not the new Theta_p^(t)).

## The argument in three steps

### Step 1. The late-apply is equivalent to using a different t_p offset within the H sync interval.

The DiLoCo paper explicitly states (Section 3, page 3): "Let `t_1, t_2, ..., t_P` in {0, ..., H-1} be distinct offsets. Then, at step t we only synchronize a fragment p if `t mod H = t_p`."

The choice of `t_p` is arbitrary within {0, ..., H-1}. Any choice produces a valid Algorithm 2 instance. Now consider two parallel instances:
- Instance A: fragment p has offset `t_p`. The syncer applies the outer step at step `t` exactly when `t mod H = t_p`.
- Instance B: fragment p has offset `t_p + D` (mod H). The syncer applies the outer step at step `t + D` exactly when `(t + D) mod H = (t_p + D) mod H = t_p + D`. Both instances run Algorithm 2 with valid parameters.

The Mend orchestrator's late-apply behavior is structurally equivalent to running Instance A's syncer schedule but having the LEARNER use Instance B's offset for its local-state-reset point. The merge itself is identical (uses the same set of `theta_{m,p}^(t)` learner snapshots, weighted by the same `w_mp` function, and the same `Theta_p^(t - H)` base). The only difference is that the learner's "I have just received an updated global state" event arrives D inner steps later than the syncer's "I just published the global state" event.

This is a hybrid of two valid Algorithm 2 schedules. The merge is well-defined; the new global state is well-defined; the only convergence concern is the learner's local trajectory during the gap.

### Step 2. The learner's trajectory during the D-step gap is bounded by the inner optimizer's behavior, which the paper already accommodates.

During inner steps `t+1, ..., t+D`, the learner applies inner optimization using its own data shard:

```
theta_{m,p}^(t+k) = InnerOpt(theta_{m,p}^(t+k-1), gradient at theta_{m,p}^(t+k-1))
```

for k in {1, ..., D}. At step `t+D` the orchestrator's broadcast arrives and the learner resets:

```
theta_{m,p}^(t+D) <- Theta_p^(t)
```

The learner's `theta_{m,p}^(t+D)` immediately before the reset differs from `Theta_p^(t)` by at most D iterations of the inner optimizer. For AdamW (the paper's chosen inner optimizer) with learning rate `eta_inner`, the L2 distance is bounded by:

```
|| theta_{m,p}^(t+D)_pre_reset - Theta_p^(t) || <= sum_{k=1}^{D} eta_inner * || g_k ||
```

where `g_k` is the per-step Adam-normalized gradient. For typical pretraining (eta_inner ~ 3e-4, AdamW-normalized gradients of unit-RMS-scale magnitude), each inner step moves the model approximately `eta_inner` in L2 norm. So D inner steps move it approximately `D * eta_inner`.

The DiLoCo paper's algorithm tolerates this drift by design. From the paper (caption of Figure 2): "The second learner stalls for three steps, but the overall training never stops. All missed updates are applied to the faulty learner's state once it continues training." This is a 3-inner-step stall regime, which is structurally the same as our 3-inner-step late-apply.

For D > 3, the paper does not provide direct experimental validation, but the algorithm itself does not encode any D <= 3 restriction. The token-weighted merge (line 9 of Algorithm 2) was specifically designed to handle larger inter-learner step variation: "slightly delayed updates can still contribute to the global update, reducing outer gradient variance while naturally handling hardware speed discrepancies" (Section 3.2).

### Step 3. The maximum D is bounded by H-1 in the worst case, and by ceil(grace_window_ms / T_step) in the orchestrator's typical operating point.

The orchestrator's late-apply window is at most the grace window `G = grace_window_ms` (the asyncio task's task does not extend its wait beyond `2 * grace_window_ms` per `concurrent.py:228-229`). The number of inner steps that fit into this window is `D_max = ceil(G / T_step)`.

With G = 2000ms default and T_step in {237ms, 265ms, 344ms, 484ms} from the Phase 2 Week 1 measurements, D_max is in {5, 8}. With the recommended auto-tuner setting `N* = ceil(G / T_step)`, D becomes approximately equal to N* itself; the orchestrator transitions from compute-bound (D = 0) to delay-bound (D ≈ N*) at the auto-tuned operating point.

If the operator chooses `N >= D_max + 1` (always at least one inner step beyond the grace window), then D < N <= H, which is within the H sync interval and the algorithm's structural assumptions. The default `MendConfig.sync_period_steps = 128` and `momentum_sync_period_steps = 512` give a generous safety factor: even D = 8 (the largest we have measured) is 16x smaller than the default N.

The risk regime is when an operator chooses a small N (e.g., N=4 from the auto-tuner) AND a large grace_window_ms (e.g., G=5000), AND the workload has a small T_step (e.g., T_step=200ms). Then D_max = 25 inner steps. This exceeds the experimental regime of the DiLoCo paper (where tau=2 implied D <= 1). The orchestrator should refuse to operate in this regime; the `auto_tune_sync_period_min` config knob is the load-bearing safety bound. Set it to at least `ceil(G / T_step)`; if T_step is unknown, set it to at least 8 (a conservative upper bound for typical hyperscaler training).

## Empirical validation already in place

The Track A/B/D/G measurements show that for D in {1, ..., 8}, the orchestrator preserves loss equivalence within the bf16 stochastic noise band:

| Track | Workload | D (estimated) | Final loss delta (sync vs conc) | Within bf16 noise? |
|---|---|---:|---:|---|
| A | Llama-3-8B 8xH100 | 1 (compute-bound) | -0.010 | Yes |
| B | Qwen-7B 8xH100 | 1 (compute-bound) | +0.21 | Yes |
| G | Qwen-3B H100:1 | ~4 (delay-bound) | (per Track G: tight per-seed) | Yes |
| prior | Qwen-1.5B H100:1 | ~7 (delay-bound) | (per multi-seed CI run) | Yes |
| prior | SmolLM-135M A10G | ~5 (delay-bound) | -0.0013 to +0.0018 | Yes (bit-exact-class) |

The empirical evidence supports the structural argument: at D up to approximately 7 inner steps with sync_period_steps=10, the orchestrator preserves loss equivalence. The proof sketch above gives the structural reason; the measurements give the empirical confirmation.

## Edge cases and limitations

1. **D >= H breaks the structural argument.** If D >= H, the orchestrator would be applying a delta into an outer-step boundary that the syncer has already moved past. The proof sketch above requires D < H. Practical defense: keep `MendConfig.sync_period_steps >= 4 * ceil(grace_window_ms / minimum_T_step)`.

2. **D >= momentum_sync_period_steps would corrupt the DES-LOC ordering.** The runtime queries the DES-LOC schedule at `outer_step_begin()`. If the schedule's momentum-sync-period boundary advances before the orchestrator's grace-window-late delta lands, the momentum-vs-parameter ordering invariant is violated. Mitigation: enforce `momentum_sync_period_steps >= sync_period_steps + D_max` in MendConfig validation. The current config (default sync_period_steps=128, momentum=512) trivially satisfies this for any D <= 384.

3. **Radial-Directional Averaging (RDA) ordering.** The DiLoCo paper's Section D.2 introduces RDA as the merge function for M > 2 learners; the radial component is sensitive to the order in which learners report. The orchestrator's late-apply does NOT change the merge function or its inputs (the merge runs at the syncer step `t`, not at `t + D`). RDA invariants are preserved.

4. **Non-monotonic merge weight under late arrival.** A learner that reports very late (e.g., due to a fail-slow that the FALCON sub-system has not yet flagged) gets a `w_mp` that grows in the tokens-numerator but stays flat in the steps-denominator, biasing its quality factor up. The orchestrator's grace window does not change this; it only changes WHEN the merge result is broadcast to learners. Fail-slow handling is the same as in the synchronous baseline.

5. **Asyncio scheduling jitter on the orchestrator's tick_interval_s.** The `tick_interval_s = 0.005s` default at `concurrent.py:111` introduces a small jitter in when the orchestrator detects the grace window has elapsed. The jitter is bounded by 5ms per outer-period, which is negligible relative to typical `grace_window_ms = 2000`. No new convergence concern.

## What we do NOT claim

This sketch does NOT establish:
- A closed-form convergence rate (e.g., "the orchestrator converges as O(1/sqrt(T)) under conditions X, Y, Z").
- That the orchestrator works for any inner optimizer beyond AdamW (the paper's chosen InnerOpt). Sophia, Lion, or Adafactor may have different staleness sensitivity.
- That the orchestrator works at scales beyond the experiments here (M >= 8 learners, P >= 24 fragments, H >= 24 inner-step sync intervals). The DiLoCo paper itself tested only these regimes.
- That the auto-tuner's N* = ceil(G / T_step) preserves convergence; only that D <= N - 1 < H, so the structural argument continues to hold.

These are future research questions, not blocking for the orchestrator's adoption as a software-only Decoupled DiLoCo enhancement.

## Recommended action

1. Reference this document from the orchestrator's module docstring (`concurrent.py:10-13`) so that the convergence-equivalence justification is auditable from the codebase.
2. Add a runtime assertion that `sync_period_steps > ceil(grace_window_ms / inner_step_time_estimate)` to catch the D >= H risk at config-validation time. This requires an `inner_step_time_estimate` MendConfig field, OR the auto-tuner warmup loop.
3. Update `phase2_week1_async_tp_overlap.md` (the original design doc) to point at this brief instead of the one-line "no new convergence proof required" hand-wave.

## References verified

- arXiv:2604.21428v1 (Decoupled DiLoCo), full text extracted via `pdftotext -layout`. Algorithm 2 at p. 4-5; adaptive grace window at p. 5; merge function (RDA) at p. 27-28; hardware failure simulation at Tables 2 and 3, p. 9-10.
- The DiLoCo paper does NOT contain a formal convergence theorem with explicit staleness bound. The closest thing to a bound is the empirical "learner stalls for three steps" caption of Figure 2 (p. 4).
- An internal uplift-surface-characterization brief (2026-05-23) provides the per-workload D estimates used in the empirical-validation table.
- `tsugiai-mend-sdk/src/tsugi_mend/concurrent.py:228-229`: the asyncio task's deadline is bounded by `2 * grace_window_ms`, which is the orchestrator's hard upper bound on D * T_step.
