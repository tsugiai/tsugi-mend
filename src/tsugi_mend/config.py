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


@dataclass
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
    # arXiv:1905.13727); convergence-preserving by design. Compression
    # modes other than "none" are experimental and off by default.
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
        if self.simulated_merge_delay_distribution not in (
            "constant", "bimodal", "long_tail",
        ):
            raise ValueError(
                f"simulated_merge_delay_distribution must be one of "
                f"'constant', 'bimodal', 'long_tail'; "
                f"got {self.simulated_merge_delay_distribution!r}"
            )
        if self.outer_step_compression_mode not in ("none", "int8", "powersgd"):
            raise ValueError(
                f"outer_step_compression_mode must be one of "
                f"'none', 'int8', 'powersgd'; "
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
