"""Configuration dataclass for the tsugiai-mend-sdk runtime.

Fields are grouped by the public-art technique they parameterize. Defaults
are deliberately conservative; the canonical cross-rack configuration is
documented in `docs/benchmark_protocol.md`.

This module is patent-independent. There is no variance-threshold trigger,
no K-of-N adapter routing, no elastic-adapter-buffer concept. Those belong
to the companion patent-aligned SDK at github.com/tsugiai/tsugi-kpool
and are not present here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tsugi_mend.sideband import (
    DEFAULT_SIDEBAND_INBOUND_READ_TIMEOUT_S,
    DEFAULT_SIDEBAND_MAX_INBOUND_CONNECTIONS,
    DEFAULT_SIDEBAND_MAX_LINE_BYTES,
)


@dataclass(kw_only=True)
class MendConfig:
    """tsugiai-mend-sdk configuration.

    The configuration is grouped by mechanism:

    1. Cross-rack reducer (Decoupled DiLoCo).
    2. Desynchronized optimizer momenta (DES-LOC).
    3. Async tensor parallelism (TorchTitan).
    4. Fail-slow detection (FALCON).
    5. Rack-aware topology classification.
    6. Sideband control plane (heartbeat metadata channel).
    7. Diagnostics output.

    Defaults are chosen so a developer running on a single node sees
    behavior close to vanilla FSDP. Cross-rack effects only activate when
    `sideband_peers` lists more than one peer and the topology classifier
    identifies cross-rack ranks.
    """

    # ------------------------------------------------------------------
    # Cross-rack reducer (Decoupled DiLoCo) -- arXiv:2604.21428
    # ------------------------------------------------------------------
    # Minimum number of learners (racks) whose fragments must be present
    # before the syncer may begin aggregation. Decoupled DiLoCo
    # Algorithm 2.
    quorum_min_learners: int = 4
    # After the K-th learner arrives, the syncer waits up to this many
    # milliseconds for additional learners before performing the merge.
    # Algorithm 2 "adaptive grace window".
    grace_window_ms: int = 2000
    # Per-learner contribution weighting. Decoupled DiLoCo paper merges
    # by the number of tokens consumed by that learner since the last
    # outer step (Algorithm 2 line 11).
    token_weighted_merge: bool = True
    # Operator-declared expected learner (rack) roster for each outer
    # round. When set, the runtime threads it to
    # GraceWindowSyncer.start_round(round_id, expected_learner_ids=...),
    # which lets the syncer EARLY-FINALIZE the moment every expected,
    # non-fail-slow learner has reported and quorum is met (instead of
    # always waiting out the full grace window), and surface the absentee
    # diagnostic (MergeResult.learners_absent). Default None preserves the
    # historical quorum-then-full-grace control law byte-for-byte and an
    # empty learners_absent.
    #
    # ROSTER-ID CONTRACT (load-bearing): each id here MUST equal a
    # `LearnerFragment.learner_id` the round's fragment_provider delivers
    # (e.g. "rack-0", "rack-1"), NOT necessarily this process's own
    # `rank_id`, and the roster must be EXHAUSTIVE: name every learner
    # expected to report this round. The expected set is matched against
    # the arriving fragments' learner_id by the syncer. Ids that never
    # arrive are safe: early-finalize never fires, the round falls back to
    # quorum-then-grace, and the absentees are reported. A quorum the
    # roster cannot satisfy is rejected at config init. An UNDER-DECLARED
    # roster (one that omits a live learner) is NOT detected: the round
    # early-finalizes once the declared set is present, which can exclude
    # a late straggler that a roster-unaware round would have merged
    # within the remaining grace window. Declare the full roster.
    expected_learner_ids: Optional[tuple[str, ...]] = None
    # Outer-optimizer momentum (Nesterov). DiLoCo / Decoupled DiLoCo
    # default.
    outer_optimizer_momentum: float = 0.9

    # ------------------------------------------------------------------
    # Desynchronized optimizer (DES-LOC) -- arXiv:2505.22549
    # ------------------------------------------------------------------
    # Number of inner steps between parameter syncs.
    sync_period_steps: int = 128
    # Number of inner steps between adaptive-momenta syncs. DES-LOC's
    # core insight is that momenta can sync less frequently than
    # parameters; the paper reports 170x less communication than DDP
    # at M=4*N.
    momentum_sync_period_steps: int = 512

    # ------------------------------------------------------------------
    # Async tensor parallelism (TorchTitan)
    # ------------------------------------------------------------------
    # Whether to install async-TP overlap hooks for intra-node
    # collectives. The TorchTitan note says ~29% forward / ~8% E2E on
    # Llama 3 7B at 64 H100.
    async_tp_enabled: bool = True
    # Phase 2 Week 1 (2026-05-22): run the cross-rack outer-step
    # (GraceWindowSyncer) concurrently with inner-step async-TP so the
    # local rank's GPUs stay busy through the grace window instead of
    # blocking. The merged outer-step delta is applied at the next
    # inner-sync boundary (typically 1-3 inner steps later). This is
    # convergence-equivalent to Decoupled DiLoCo Algorithm 2 because
    # the algorithm was designed for staggered inner-step blocks
    # between learners. Expected uplift on top of the Stage D-proper
    # +28.58% baseline: 5-12% (OpenAI estimate). Patent-independent.
    concurrent_outer_step: bool = True
    # CUDA stream priority for the outer-step reduce-scatter when
    # running concurrently. 0 is default priority; negative values
    # are higher priority. Stage A does not exercise CUDA streams;
    # Stage B+ benchmarks validate the device-level overlap.
    outer_step_cuda_stream_priority: int = 0
    # Phase 2 Week 1 Day 4-7: optional simulated grace-window wait,
    # in milliseconds, injected inside GraceWindowSyncer._finalize.
    # Models real-world cross-rack grace-window wait (FALCON
    # arXiv:2410.12588 documents bimodal tail latencies as the
    # dominant inner-step idle source on commodity Ethernet). Applied
    # to BOTH synchronous and orchestrator paths so an apples-to-
    # apples comparison isolates the orchestrator's overlap benefit.
    # Default 0 = no synthetic delay (production behavior).
    simulated_merge_delay_ms: int = 0
    # Phase 2 Week 1 Day 4-7: distribution shape for the simulated
    # merge delay. "constant" injects exactly `simulated_merge_delay_ms`
    # on every outer-round; "bimodal" matches the FALCON observation
    # of 80% short rounds (50 ms) + 20% long rounds (capped at the
    # `simulated_merge_delay_ms` value); "long_tail" draws from a
    # log-normal distribution with mean = simulated_merge_delay_ms / 2.
    # Default "constant" preserves the existing single-seed and 3-seed
    # CI measurements. Bimodal / long_tail are non-default exploratory
    # paths for stretch validation.
    simulated_merge_delay_distribution: str = "constant"
    # Opt-in compression transform applied to the params_delta inside the
    # fragment provider. "none" (default) is a no-op pass-through that
    # preserves bit-exact loss equivalence. "int8" applies per-tensor
    # symmetric INT8 quantization (lossy; not convergence-preserving
    # without error feedback). "powersgd" applies rank-r low-rank
    # approximation with persistent error-feedback residual (Vogels 2019,
    # arXiv:1905.13727); convergence-preserving by design. "sparse"
    # applies a lossless index+value delta codec with dense fallback, so
    # it preserves exact tensor values but only reduces communication when
    # the delta is genuinely element-sparse. Compression modes other than
    # "none" are experimental and off by default.
    outer_step_compression_mode: str = "none"
    # PowerSGD low-rank approximation rank. Higher r reduces compression
    # but improves accuracy. PowerSGD paper recommends r=4 as default.
    outer_step_compression_powersgd_rank: int = 4

    # ------------------------------------------------------------------
    # Auto-tuner for sync_period_steps (uplift-surface characterization).
    # When enabled, the runtime measures
    # per-step compute time T_step over a warmup window and sets the
    # effective sync_period_steps to ceil(grace_window_ms / T_step),
    # clamped to [auto_tune_sync_period_min, sync_period_steps]. The
    # static sync_period_steps field becomes the upper bound on the
    # auto-tuned N; the static value is preserved when the auto-tuner
    # is disabled.
    #
    # Default disabled because: (a) it changes the convergence trajectory
    # at the outer-step cadence; (b) it requires measured T_step which
    # is not available at initialization; (c) the optimal N depends on
    # the late-apply lag D, which is a separate convergence-equivalence
    # consideration (see docs/convergence_equivalence_sketch.md). Should
    # be enabled by hyperscaler workloads after a short workload-specific
    # validation pass.
    auto_tune_sync_period: bool = False
    # Number of warmup steps over which to measure T_step. The runtime
    # uses the median of the last 25 of these steps to dodge cudagraph-
    # capture artifacts that inflate the earliest step times.
    auto_tune_sync_period_warmup_steps: int = 50
    # Lower bound on the auto-tuned N. Bounds the worst-case late-apply
    # lag D = ceil(G / T_step) - N when the orchestrator is delay-bound.
    # The Decoupled DiLoCo convergence-equivalence argument (see
    # docs/convergence_equivalence_sketch.md) is well-established for
    # D in {1, 2, 3}; clamping N so D stays within that range is the
    # load-bearing safety bound.
    auto_tune_sync_period_min: int = 4

    # ------------------------------------------------------------------
    # Fail-slow detection (FALCON) -- arXiv:2410.12588
    # ------------------------------------------------------------------
    # Sliding window of recent step-time samples used to compute the
    # z-score for the current step.
    failslow_window_steps: int = 50
    # If a rank's recent step time exceeds this many standard deviations
    # above the rolling mean of its window, mark it stragglering and
    # exclude from this round's quorum.
    failslow_zscore_threshold: float = 3.0
    # Minimum window fill before z-score detection is permitted. Avoids
    # spurious flags at startup.
    failslow_min_samples: int = 10

    # ------------------------------------------------------------------
    # Online runtime autotuner (fail-slow detection sensitivity +
    # grace-window wait). Published basis: Guard (arXiv:2605.17879,
    # online performance monitoring / node health) for the
    # detection-threshold adaptation, and "From Detection to Recovery"
    # (arXiv:2605.09370, operational analysis) for the recovery-wait
    # heuristic. Engineering only; the upstream papers' headline numbers
    # are NOT reproduced here as tsugi-mend results.
    # ------------------------------------------------------------------
    # Master switch. When True, the runtime continuously adapts (past a
    # warmup window) the effective fail-slow z-score threshold and the
    # effective grace-window wall-clock wait from the observed per-rank
    # step-time stream. Default OFF so default-mode behavior is
    # byte-for-byte unchanged.
    #
    # Bit-exact-safe by construction: (a) the detection threshold is
    # observe-only (no mitigation/exclusion is wired off the detector
    # here), so adapting it changes only which steps are FLAGGED in the
    # diagnostic stream, never any tensor value; (b) the grace window is a
    # wall-clock wait only -- in default mode the syncer waits for the same
    # fragments regardless of the wait length, so adapting it changes
    # timing/overlap, never WHICH fragments merge or the apply boundary.
    # The autotuner deliberately does NOT online-adapt the merge cadence
    # (sync_period_steps / momentum cadence / apply lag); doing so would
    # make a paired baseline-vs-sdk run diverge (their step times differ).
    auto_tune_runtime: bool = False
    # Rolling window of recent step-time samples the autotuner's control
    # law statistics (CoV, peak ratio) are computed over.
    auto_tune_runtime_window_steps: int = 50
    # Minimum window fill before the autotuner starts adapting. Until this
    # many samples accumulate the effective values stay at the static
    # config baseline (failslow_zscore_threshold / grace_window_ms).
    auto_tune_runtime_min_samples: int = 10
    # Bounds on the adapted z-score threshold. The detector's static
    # failslow_zscore_threshold is the floor the clean-cluster case relaxes
    # toward; the autotuner raises the threshold up to the max under high
    # observed jitter.
    auto_tune_zscore_min: float = 2.0
    auto_tune_zscore_max: float = 8.0
    # Bounds (ms) on the adapted grace-window wait. The static
    # grace_window_ms sits inside this range; the autotuner widens up to
    # the max under a sustained straggler and narrows back toward baseline
    # when clean.
    auto_tune_grace_window_min_ms: int = 0
    auto_tune_grace_window_max_ms: int = 10_000
    # Control-law gains. cov_gain scales how strongly observed jitter
    # (coefficient of variation) raises the z-score threshold; grace_gain
    # scales how strongly a sustained straggler (recent peak/median ratio)
    # widens the grace window. Both must be >= 0.
    auto_tune_cov_gain: float = 4.0
    auto_tune_grace_gain: float = 1.0
    # Number of consecutive windows a candidate sensitivity or grace-window
    # move must persist before it is applied. Default 1 preserves the existing
    # immediate adaptation behavior; larger values suppress one-window blips.
    auto_tune_runtime_sustained_windows: int = 1
    # Observe-only EWMA/CUSUM drift signal. The CUSUM accumulates positive
    # EWMA latency excess relative to a peer baseline. It emits diagnostics
    # only and never feeds exclusion, cadence, tensors, or merge selection.
    auto_tune_drift_ewma_alpha: float = 0.2
    auto_tune_drift_cusum_threshold: float = 2.0
    auto_tune_drift_cusum_slack: float = 0.05

    # ------------------------------------------------------------------
    # Rack-aware topology
    # ------------------------------------------------------------------
    # Whether the runtime should attempt rack-aware DP-last mapping.
    # Reads NCCL_TOPO_FILE if present; otherwise falls back to
    # heuristics on hostname / rank groupings.
    rack_aware: bool = True
    # Override path to NCCL_TOPO_FILE. None means "use env var".
    nccl_topo_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Sideband (control-plane metadata channel)
    # ------------------------------------------------------------------
    # The sideband carries step-id, vector-clock, queue-depth, and
    # health-bit metadata between racks. It is deliberately separate
    # from the NCCL data plane. NOTE: this is the engineering
    # control-plane / data-plane split common to most distributed
    # systems (see Decoupled DiLoCo Section 3.2) and does not exercise
    # TsugiCinema's Infinity patent claims.
    sideband_addr: str = "tcp://0.0.0.0:51900"
    sideband_peers: tuple[str, ...] = field(default_factory=tuple)
    sideband_heartbeat_ms: int = 100
    # Sub-second connect timeout for sideband peer dial.
    sideband_connect_timeout_s: float = 0.5
    # Sideband security remains opt-in for 0.1.x to preserve the
    # zero-config trusted-network UX. Secure-by-default auth is planned
    # for 0.2.0.
    sideband_psk: Optional[str] = None
    sideband_peer_allowlist: Optional[tuple[str, ...]] = None
    sideband_max_line_bytes: int = DEFAULT_SIDEBAND_MAX_LINE_BYTES
    # Bound incomplete or slow inbound control-plane frames.
    sideband_inbound_read_timeout_s: float = DEFAULT_SIDEBAND_INBOUND_READ_TIMEOUT_S
    # Cap concurrent inbound handler work to limit socket-exhaustion pressure.
    sideband_max_inbound_connections: int = DEFAULT_SIDEBAND_MAX_INBOUND_CONNECTIONS
    sideband_tls: bool = False
    sideband_tls_certfile: Optional[str] = None
    sideband_tls_keyfile: Optional[str] = None
    sideband_tls_ca_file: Optional[str] = None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.quorum_min_learners < 1:
            raise ValueError(
                f"quorum_min_learners must be >= 1; got {self.quorum_min_learners}"
            )
        if self.grace_window_ms < 0:
            raise ValueError(f"grace_window_ms must be >= 0; got {self.grace_window_ms}")
        if self.expected_learner_ids is not None:
            roster = self.expected_learner_ids
            for lid in roster:
                if not isinstance(lid, str) or lid == "":
                    raise ValueError(
                        "expected_learner_ids entries must be non-empty strings; "
                        f"got {lid!r}"
                    )
            if len(set(roster)) != len(roster):
                raise ValueError(
                    f"expected_learner_ids must not contain duplicates; got {roster!r}"
                )
            # Consistent with GraceWindowSyncer.start_round's own validation:
            # quorum can never be met if it exceeds the declared roster size.
            # Reject at config time so the operator gets a clear error rather
            # than a runtime fallback that silently ignores the roster.
            if self.quorum_min_learners > len(roster):
                raise ValueError(
                    f"quorum_min_learners ({self.quorum_min_learners}) cannot exceed "
                    f"len(expected_learner_ids) ({len(roster)}); quorum could never "
                    f"be met"
                )
        if not (0.0 <= self.outer_optimizer_momentum < 1.0):
            raise ValueError(
                f"outer_optimizer_momentum must be in [0, 1); "
                f"got {self.outer_optimizer_momentum}"
            )
        if self.sync_period_steps < 1:
            raise ValueError(f"sync_period_steps must be >= 1; got {self.sync_period_steps}")
        if self.momentum_sync_period_steps < self.sync_period_steps:
            raise ValueError(
                f"momentum_sync_period_steps ({self.momentum_sync_period_steps}) "
                f"must be >= sync_period_steps ({self.sync_period_steps}). "
                f"DES-LOC requires M >= N."
            )
        if self.failslow_window_steps < 2:
            raise ValueError(
                f"failslow_window_steps must be >= 2; got {self.failslow_window_steps}"
            )
        if self.failslow_zscore_threshold <= 0:
            raise ValueError(
                f"failslow_zscore_threshold must be > 0; "
                f"got {self.failslow_zscore_threshold}"
            )
        if self.failslow_min_samples < 2:
            raise ValueError(
                f"failslow_min_samples must be >= 2; got {self.failslow_min_samples}"
            )
        if self.failslow_min_samples > self.failslow_window_steps:
            raise ValueError(
                f"failslow_min_samples ({self.failslow_min_samples}) cannot exceed "
                f"failslow_window_steps ({self.failslow_window_steps})"
            )
        if self.sideband_heartbeat_ms < 1:
            raise ValueError(
                f"sideband_heartbeat_ms must be >= 1; got {self.sideband_heartbeat_ms}"
            )
        if not self.sideband_addr.startswith("tcp://"):
            raise ValueError(
                f"sideband_addr must start with tcp://; got {self.sideband_addr!r}"
            )
        for peer in self.sideband_peers:
            if not peer.startswith("tcp://"):
                raise ValueError(
                    f"sideband_peers entries must start with tcp://; got {peer!r}"
                )
        if self.sideband_psk == "":
            raise ValueError("sideband_psk must be non-empty when configured")
        if self.sideband_peer_allowlist is not None:
            for rank_id in self.sideband_peer_allowlist:
                if not isinstance(rank_id, str) or rank_id == "":
                    raise ValueError(
                        "sideband_peer_allowlist entries must be non-empty strings"
                    )
        if self.sideband_max_line_bytes < 1:
            raise ValueError(
                f"sideband_max_line_bytes must be >= 1; got {self.sideband_max_line_bytes}"
            )
        if self.sideband_inbound_read_timeout_s <= 0:
            raise ValueError(
                "sideband_inbound_read_timeout_s must be > 0; "
                f"got {self.sideband_inbound_read_timeout_s}"
            )
        if self.sideband_max_inbound_connections < 1:
            raise ValueError(
                "sideband_max_inbound_connections must be >= 1; "
                f"got {self.sideband_max_inbound_connections}"
            )
        if self.sideband_tls and (
            self.sideband_tls_certfile is None or self.sideband_tls_keyfile is None
            or self.sideband_tls_ca_file is None
        ):
            raise ValueError(
                "sideband_tls=True requires sideband_tls_certfile and "
                "sideband_tls_keyfile and sideband_tls_ca_file"
            )
        if self.simulated_merge_delay_distribution not in (
            "constant", "bimodal", "long_tail",
        ):
            raise ValueError(
                f"simulated_merge_delay_distribution must be one of "
                f"'constant', 'bimodal', 'long_tail'; "
                f"got {self.simulated_merge_delay_distribution!r}"
            )
        if self.outer_step_compression_mode not in ("none", "int8", "powersgd", "sparse"):
            raise ValueError(
                f"outer_step_compression_mode must be one of "
                f"'none', 'int8', 'powersgd', 'sparse'; "
                f"got {self.outer_step_compression_mode!r}"
            )
        if self.outer_step_compression_powersgd_rank < 1:
            raise ValueError(
                f"outer_step_compression_powersgd_rank must be >= 1; "
                f"got {self.outer_step_compression_powersgd_rank}"
            )
        if self.auto_tune_sync_period_warmup_steps < 2:
            raise ValueError(
                f"auto_tune_sync_period_warmup_steps must be >= 2; "
                f"got {self.auto_tune_sync_period_warmup_steps}"
            )
        if self.auto_tune_sync_period_min < 1:
            raise ValueError(
                f"auto_tune_sync_period_min must be >= 1; "
                f"got {self.auto_tune_sync_period_min}"
            )
        if self.auto_tune_sync_period_min > self.sync_period_steps:
            raise ValueError(
                f"auto_tune_sync_period_min ({self.auto_tune_sync_period_min}) "
                f"must be <= sync_period_steps ({self.sync_period_steps}). "
                f"The static sync_period_steps is the upper bound on the "
                f"auto-tuned N; the lower bound cannot exceed it."
            )
        if self.auto_tune_runtime_window_steps < 2:
            raise ValueError(
                f"auto_tune_runtime_window_steps must be >= 2; "
                f"got {self.auto_tune_runtime_window_steps}"
            )
        if self.auto_tune_runtime_min_samples < 2:
            raise ValueError(
                f"auto_tune_runtime_min_samples must be >= 2; "
                f"got {self.auto_tune_runtime_min_samples}"
            )
        if self.auto_tune_runtime_min_samples > self.auto_tune_runtime_window_steps:
            raise ValueError(
                f"auto_tune_runtime_min_samples ({self.auto_tune_runtime_min_samples}) "
                f"cannot exceed auto_tune_runtime_window_steps "
                f"({self.auto_tune_runtime_window_steps})"
            )
        if self.auto_tune_zscore_min <= 0:
            raise ValueError(
                f"auto_tune_zscore_min must be > 0; got {self.auto_tune_zscore_min}"
            )
        if self.auto_tune_zscore_max < self.auto_tune_zscore_min:
            raise ValueError(
                f"auto_tune_zscore_max ({self.auto_tune_zscore_max}) must be >= "
                f"auto_tune_zscore_min ({self.auto_tune_zscore_min})"
            )
        if self.auto_tune_grace_window_min_ms < 0:
            raise ValueError(
                f"auto_tune_grace_window_min_ms must be >= 0; "
                f"got {self.auto_tune_grace_window_min_ms}"
            )
        if self.auto_tune_grace_window_max_ms < self.auto_tune_grace_window_min_ms:
            raise ValueError(
                f"auto_tune_grace_window_max_ms ({self.auto_tune_grace_window_max_ms}) "
                f"must be >= auto_tune_grace_window_min_ms "
                f"({self.auto_tune_grace_window_min_ms})"
            )
        if self.auto_tune_cov_gain < 0:
            raise ValueError(
                f"auto_tune_cov_gain must be >= 0; got {self.auto_tune_cov_gain}"
            )
        if self.auto_tune_grace_gain < 0:
            raise ValueError(
                f"auto_tune_grace_gain must be >= 0; got {self.auto_tune_grace_gain}"
            )
        if self.auto_tune_runtime_sustained_windows < 1:
            raise ValueError(
                f"auto_tune_runtime_sustained_windows must be >= 1; "
                f"got {self.auto_tune_runtime_sustained_windows}"
            )
        if not (0.0 < self.auto_tune_drift_ewma_alpha <= 1.0):
            raise ValueError(
                f"auto_tune_drift_ewma_alpha must be in (0, 1]; "
                f"got {self.auto_tune_drift_ewma_alpha}"
            )
        if self.auto_tune_drift_cusum_threshold <= 0:
            raise ValueError(
                f"auto_tune_drift_cusum_threshold must be > 0; "
                f"got {self.auto_tune_drift_cusum_threshold}"
            )
        if self.auto_tune_drift_cusum_slack < 0:
            raise ValueError(
                f"auto_tune_drift_cusum_slack must be >= 0; "
                f"got {self.auto_tune_drift_cusum_slack}"
            )
