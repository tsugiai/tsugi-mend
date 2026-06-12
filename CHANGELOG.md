# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **RuntimeAutotuner v2 observe-only signals** (default behavior unchanged;
  autotuner OFF path untouched, ON path with default knobs byte-identical):
  - **Per-learner EWMA/CUSUM drift flag.** New
    `autotuner.EwmaCusumDriftClassifier`: an exponentially weighted moving
    average of each learner's step latency plus a one-sided CUSUM
    accumulator against the peer baseline (median of the other learners'
    EWMAs when peer streams are fed via the new
    `RuntimeAutotuner.observe_peer()`; the learner's own slow-EWMA
    reference for a single local stream). The design pattern is classical
    statistical process control (EWMA and one-sided CUSUM control charts);
    O(1) state and time per observation. It catches a slow intra-window
    latency drift that a static sliding-window z-score misses (the window
    mean ramps along with the samples). Surfaced additively as
    `AutotuneDecision.drift_flag` / `drift_cusum` and, when a diagnostics
    writer is attached to the autotuner, as a new `auto_tune_drift_flag`
    JSONL event emitted on the flag's rising edge. **FLAG-ONLY by
    construction:** the signal never feeds learner exclusion, merge
    cadence, the effective threshold/grace values, or any tensor; merge
    math is frozen and default-mode bit-exactness is unaffected.
  - **Sustained peer-relative gate** on the existing adaptation moves: a
    sensitivity (z-score threshold) or grace-window move now applies only
    after its triggering condition has held for
    `(sustain_windows - 1) * window_steps + 1` consecutive evaluations,
    i.e. `sustain_windows` consecutive windows' worth (the detection-side
    multi-window sustained-deviation pattern from Guard, arXiv:2605.17879;
    its mitigation half, job restart, is deliberately not adopted). The
    default `sustain_windows = 1` reproduces the previous control law
    byte-for-byte; any `K >= 2` provably suppresses single-sample blips (a
    lone sample deviates the rolling-window statistics for at most its
    window residence). Relaxation back toward baseline is never delayed.
  - New defaulted `MendConfig` knobs, validated at config init:
    `auto_tune_sustain_windows`, `auto_tune_drift_ewma_alpha`,
    `auto_tune_drift_baseline_alpha`, `auto_tune_drift_cusum_slack`,
    `auto_tune_drift_cusum_threshold`. Note: the runtime does not yet
    thread these knobs into the `RuntimeAutotuner` it constructs
    internally (defaults apply on that path, preserving current behavior);
    the one-line runtime wiring plus peer-stream feeding from the sideband
    is tracked as follow-up work.

- New opt-in outer-step compression mode `outer_step_compression_mode="quant4"`
  at the existing `compression.apply_compression` transform seam: symmetric
  4-bit block-wise quantization of the outer delta (blocks of 128 elements,
  one fp32 scale per block computed as max-abs / 7, integer codes in [-7, 7],
  two codes packed per byte) with a per-key error-feedback residual
  (`Quant4State`, mirroring how `PowerSGDState` carries its state; exactly one
  residual tensor per key; cleared by `reset()`). New public symbols:
  `Quant4State`, `Quant4Payload`, `quant4_encode`, `quant4_decode`,
  `quant4_compress_delta`, `DEFAULT_QUANT4_BLOCK_SIZE`.
  **quant4 is LOSSY when enabled.** The default mode remains `none`, which is
  bit-exact and is byte-for-byte untouched by this change (asserted with
  `torch.equal` in the new off-path test); `int8` / `powersgd` / `sparse`
  behavior is unchanged. Pinned non-finite policy: NaN and +/-inf inputs
  encode as 0 and are excluded from the block scale, and they are sanitized
  before the error-feedback update so a transient non-finite value cannot
  poison the persistent residual; IEEE-754 negative zero decodes as +0.0.
  Motivated by Streaming DiLoCo (arXiv:2501.18512), which reports that 4-bit
  quantization of outer gradients holds loss at iso-quality while reducing
  cross-rack bandwidth; that is the paper's claim on the paper's workloads,
  not a result re-validated here. No new runtime callers are wired; like the
  other modes, quant4 is exercised at the transform seam only.

### Changed

- `MendConfig` construction is now keyword-only. This is a public-API tightening
  for hypothetical external positional callers; positional field order was never
  a stable contract, so no deprecation shim is warranted.
- Optimized the lossless sparse-delta bitwise nonzero mask while preserving
  exact raw-byte semantics for negative zero, non-finite values, and subnormals.
- Under-declared expected learner rosters now disable roster-aware
  early-finalize in place when an unexpected non-fail-slow learner arrives.
  Already-accepted fragments, fail-slow state, and grace-window timing are
  preserved, and `outer_step_collect` reports the fallback as
  `roster_fallback`.

### Fixed

- Corrected `MergeResult.learners_excluded` so it reports only fragments
  actually excluded from the merge when a learner is marked fail-slow after
  its fragment was already accepted.

### Tests

- Added deterministic Hypothesis coverage for `GraceWindowSyncer` roster
  bit-identity, liveness, and diagnostic subset invariants.
## [0.1.4] - 2026-06-10

### Added

- World-size-aware multi-rack reducer is now **wired live through the runtime**.
  New `MendConfig.expected_learner_ids: tuple[str, ...] | None` (default `None`)
  declares the round's expected learner (rack) roster; the runtime threads it
  into `ConcurrentOuterStep`, which passes it to
  `GraceWindowSyncer.start_round(round_id, expected_learner_ids=...)` on every
  outer round, so the early-finalize (`reason == "all_present"`) and absentee
  diagnostic (`MergeResult.learners_absent`) added in the previous entry are no
  longer dormant. `ConcurrentOuterStep.__init__` / `submit_async` and
  `mend`'s `outer_step_begin` gained an optional `expected_learner_ids:
  frozenset[str] | None` (constructor default plus per-round override) for
  clusters whose live world size changes round-to-round. The
  `outer_step_collect` diagnostic event now also carries `learners_absent`.
  **Roster-id contract:** each declared id must equal a
  `LearnerFragment.learner_id` the round's fragment provider delivers (not
  necessarily this process's own `rank_id`), and the roster must be
  **exhaustive**: name every learner expected to report this round. **Safe
  fallback:** a roster too small to satisfy quorum is rejected at config init
  (config-level roster) or dropped to the `None` path with a warning
  (per-round override); a roster naming learners that never arrive is inert
  (early-finalize simply never fires, the round finalizes via grace once
  quorum is met by the learners that do arrive, absentees show up in
  `learners_absent`). Neither case hangs the round or changes the merged
  result. An **under-declared** roster (one that omits a live learner) is not
  detected: the round early-finalizes once the declared set is present, which
  can exclude a late straggler that a roster-unaware round would have merged
  within the remaining grace window; a stricter detect-and-replay fallback is
  tracked as future work. **Bit-exact:** with the default `None`, the merge
  control law and merged tensors are unchanged (the only observable
  default-mode difference is the additive `learners_absent` field/diagnostic
  key, always `[]`); when on, early-finalize merges the same fragment set
  grace-expiry would for the same arrivals, so the merged delta is
  bit-identical (asserted with `torch.equal` in the new
  `concurrent`/`runtime` integration tests).
- World-size-aware multi-rack reducer for `reducer.GraceWindowSyncer`.
  `start_round` now accepts optional `expected_learner_ids: set[str] | None`
  (and `total_learners: int | None`). When the expected learner set is known,
  the syncer **early-finalizes** the moment every expected, non-fail-slow
  learner has reported and quorum is met, emitting a new
  `MergeResult.reason == "all_present"` instead of always waiting out the full
  grace window. It also records `MergeResult.learners_absent: list[str]` -- the
  expected learners neither received nor fail-slow-excluded at finalize
  (disjoint from `learners_merged` and `learners_excluded`). Supplying a known
  total now rejects `quorum_min_learners > total` with a clear `ValueError`.
  This closes the documented 3+/4+ rack gap (needless post-quorum latency and
  no absentee diagnostic) at flat-merge granularity.

### Changed

- `start_round(round_id)` gained two optional keyword parameters
  (`expected_learner_ids`, `total_learners`), both defaulting to `None`. With
  both omitted, the control law is unchanged: quorum, then the full grace
  window, and an empty `learners_absent`. No public symbol was renamed or
  removed; the change is additive.
- `ConcurrentOuterStep.__init__`, `ConcurrentOuterStep.submit_async`, and the
  runtime's `outer_step_begin` gained an optional `expected_learner_ids`
  keyword (default `None`). All additive; the default path is unchanged.

## [0.1.3] - 2026-06-09

### Added

- Lossless sparse-delta compression mode (`outer_step_compression_mode="sparse"`).
  Encodes a parameter delta as flattened int64 indices plus exact values when
  that payload would be smaller than the dense tensor, and falls back to dense
  otherwise, so it never grows the wire and preserves exact tensor bits
  (including IEEE-754 negative zero, via a raw-byte non-zero mask). Opt-in; the
  default stays `"none"`. The communication benefit is conditional on genuinely
  element-sparse deltas (typical dense DiLoCo deltas select the dense fallback).
  New helpers in `tsugi_mend.compression`: `sparse_delta_encode`,
  `sparse_delta_decode`, `sparse_compress_delta`, and the `SparseDeltaPayload`
  dataclass.
- Online runtime autotuner (`MendConfig.auto_tune_runtime`, default OFF). When
  enabled, a deterministic, stdlib-only control law adapts the effective
  fail-slow z-score threshold (from the observed step-time coefficient of
  variation) and the wall-clock grace-window wait (from the recent peak/median
  ratio), each clamped to configurable min/max bounds. Bit-exact-safe by
  construction in default mode: the detector is observe-only and the grace
  window is a wall-clock wait, so adapting either changes timing/diagnostics
  only, never which fragments merge or any tensor value. New `tsugi_mend.autotuner`
  module (`RuntimeAutotuner`, `AutotuneDecision`) plus the `auto_tune_*`
  `MendConfig` fields (all defaulted).
- Seeded stall-sweep benchmark harness (`benchmarks/run_stall_sweep.py`) with a
  per-rank deterministic straggler injector, n>=5 paired trials using a
  drop-extremes + bootstrap-CI reporting rule, and a per-seed bit-exact loss
  assertion. Benchmark and docs tooling only; not shipped in the installed wheel.

### Changed

- `outer_step_compression_mode` now accepts `"sparse"` in addition to
  `"none" | "int8" | "powersgd"`.
- README version marker updated to 0.1.3; the compression-modes list now
  includes `sparse`.

## [0.1.2] - 2026-05-29

### Added

- Real multi-node cell for `benchmarks/run_paired.py`. New `--launch
  {selfspawn,torchrun,auto}` flag adds a `torchrun`/`env://` launch path so the
  same driver scales from the `$0` `cpu_gloo_2rank_mlp` cell up to a real 2-node
  GPU cluster. Object (`LearnerFragment`) gather rides a dedicated `gloo`
  process group regardless of the data-plane backend (`nccl` over flaky
  Python-object collectives is unreliable). Static rendezvous keeps the
  bundle-writer on global rank 0 deterministically.
- Pre-baked `real_8xv100_2node` cell consuming the bundle's HF identifiers
  (`HuggingFaceTB/SmolLM-135M` by default), wrapping the model in **per-node**
  FSDP, with cross-node same-local-rank `gloo` gather across the two learners.
  CUDA + the optional `real-cell` extra (`pip install 'tsugi-mend[real-cell]'`)
  are required; the path raises a clear early error on a non-CUDA host rather
  than silently falling back to the synthetic MLP.
- Optional `real-cell` and `benchmark` extras for `transformers`, `accelerate`,
  `datasets` (and friends). The core install, the `$0` cell, `import
  tsugi_mend`, the `mypy src` CI gate, and CI's default job do **not** require
  any of them.
- Result documentation for the `real_8xv100_2node` cell at
  `benchmarks/results/real_8xv100_2node/README.md`, including a `n=7` Lambda
  Labs 2x8xV100 commodity-Ethernet measurement under
  [`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).
- Bounded process-group timeouts in the benchmark harness and the torchrun
  example. An explicit `timeout=` is now passed to every `init_process_group`
  / `new_group` site (`BenchConfig.process_group_timeout_s`, default 180s;
  `--process-group-timeout-s`; `MEND_PROCESS_GROUP_TIMEOUT_S` in the example).
  A peer failure now raises a bounded backend error instead of an indefinite
  hang; happy-path numerics are unchanged.
- Multi-node runbook completion in `docs/multinode.md`: a "Failure contract and
  timeout" section, a "Multi-NIC interface selection" subsection
  (`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`, `ip -br addr` discovery, the
  wrong-interface failure mode), and filled example addresses.

### Changed

- README headline-row framing: bit-exact loss equivalence is now the
  load-bearing headline; cross-network throughput uplift is reframed as
  **jitter-conditional** and reported with a range / CI on the
  `real_8xv100_2node` cell. The prior single-run **+28.58%** point estimate is
  contextualized as a high-tail event from a higher-jitter session — `n=7`
  re-measurement under the protocol shows mean **+3.4%**, CI95 **[-5%, +12%]**,
  per-seed range **[-10%, +15%]**, with bit-exact loss on every seed. The
  protocol's "never a bare point estimate" rule applies.
- `mypy benchmarks` is now clean as well as `mypy src` (the CI gate). The
  optional `transformers` and `datasets` extras are silenced via a per-module
  override in `pyproject.toml`; the two `loss.backward()` call sites and the
  synthetic-path `_loss_for_step` return get scoped `type: ignore` codes (the
  HF forward is `Any` under `--strict`; the underlying values are real
  `Tensor`s at runtime).
- `benchmarks/README.md` and `docs/multinode.md` updated to describe the
  `torchrun` launch path and the real-hardware cell.
- Documentation now describes `tsugi-mend` honestly as a component toolkit
  rather than a transparent drop-in. `mend_init` wires the reducer plus the
  concurrent outer-step orchestrator; async-TP, DES-LOC moment synchronization,
  FALCON mitigation, and gradient compression are present as components or
  integration points and are not automatically invoked by `mend_init` in 0.1.x.
  The README quickstart now shows the real outer-step integration loop.
- NOTICE attribution scoped to what the runtime actually exercises; the
  Decoupled DiLoCo citation corrected to the published arXiv:2604.21428 title
  and author list and made consistent across `README.md`, `NOTICE`, and
  `reducer.py`.
- `real_8xv100_2node` result README reconciled to the committed cell config
  (`apply_lag_steps` 8, batch 1, `sequence_length` 256, lr 1e-5) with every
  reproduction flag pinned; the single-seed `n=1` production-floor row moved out
  of the comparative table into a labeled protocol-incomplete note; the cpu-cell
  prose aligned to the committed CI; `docs/benchmark_protocol.md` now states the
  bit-exact referent (the SDK overlap path equals the synchronous-reducer path,
  not a vanilla DDP/FSDP all-reduce).
- `SECURITY.md` updated to note that opt-in HMAC and TLS are available in 0.1.x
  (secure-by-default is planned for 0.2.0).

### Security

- Sideband opt-in controls hardened (control-plane only; the default
  trusted-network behavior is unchanged). HMAC frames are now replay-protected
  via a bounded per-rank monotonic nonce floor. TLS requires a CA file and
  verifies peer identity (`check_hostname` plus `CERT_REQUIRED`) instead of
  silently accepting any certificate. Inbound connections enforce a read timeout
  and a concurrent-connection cap (slow-reader / socket-exhaustion mitigation).
  The insecure non-loopback bind warning now fires per bind instead of once per
  process.

## [0.1.1] - 2026-05-27

### Added

- `tsugi-mend` console-script CLI (also runnable as `python -m tsugi_mend`)
  with three subcommands:
  - `version`: print the installed package version.
  - `info`: print the `MendConfig` defaults plus a one-line description
    (torch-free, like `MendConfig` itself).
  - `doctor`: environment preflight: torch importability + version, CUDA
    availability, NCCL availability, visible CUDA device count, and optional
    TCP reachability of sideband peers via `--peers tcp://host:port,...`.
    Prints a clear PASS/WARN per check and exits non-zero only on hard
    failures (an explicitly-requested peer being unreachable or malformed).
    torch is lazy-imported so `version`/`info` work without torch installed.
- PEP 561 `py.typed` marker so downstream type checkers treat the package as
  typed; included in the built wheel.
- Tokenless PyPI Trusted Publishing release workflow documentation and
  packaging checks for the first OIDC publish.
- Community support documents: contributing guide, code of conduct, security
  policy, issue templates, PR template, and Dependabot configuration.
- Multi-node getting-started runbook plus a CPU-only two-rank `torchrun`
  example covering sideband setup and diagnostics.

### Changed

- Bumped version to `0.1.1` (the `0.1.0` wheel on PyPI is frozen).
- README measurement framing now leads with the real cross-network V100 result
  and labels simulated-delay numbers as ceiling-case stress tests.

### Security

- Sideband hardening: optional HMAC pre-shared-key authentication, optional TLS
  plumbing, peer allow-listing, max-line enforcement, stricter payload
  validation, and a warning for unauthenticated non-loopback binds. These
  controls remain opt-in for 0.1.x so the default trusted-network UX is
  unchanged.

## [0.1.0] - 2026-05-27

### Added

- Initial public release of `tsugi-mend`.
