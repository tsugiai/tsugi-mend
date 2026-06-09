"""Decoupled DiLoCo cross-rack reducer.

Reference: Douillard, Rush, Donchev, Charles, ..., Ranzato, Dean (Google
DeepMind), "Decoupled DiLoCo for Resilient Distributed Pre-training",
arXiv:2604.21428, April 2026.

Algorithm summary (Algorithm 2 in the paper, pseudocode form):

    on each outer round t:
        syncer.collect_fragments_until_quorum(K)
        syncer.wait_for_grace_window(grace_ms)
        merged = token_weighted_merge(fragments_received)
        outer_optimizer.step(merged)
        broadcast(updated_params)

This module is a pure-Python implementation of the merge / quorum / grace
logic so the unit tests can drive it on synthetic tensors. The actual
NCCL / RPC plumbing is in `runtime.py`.

Patent-independence note: the quorum + grace-window + token-weighted merge
triple is the published Decoupled DiLoCo control law; it is deliberately
different from TsugiCinema's K-Pool LoRA + Infinity variance-threshold
trigger. Do not introduce a variance-threshold-fire-decision concept into
this module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from random import Random


@dataclass
class LearnerFragment:
    """A parameter-delta fragment contributed by one learner (rack) for
    one outer round.

    Attributes:
        learner_id: stable identifier (e.g., rack-0, rack-1, ...).
        round_id: monotonic outer-round counter.
        params_delta: list of CPU tensors, one per model parameter in
            the canonical parameter order known to the syncer.
        tokens_consumed: number of tokens the learner processed during
            its inner-step block. Used as the token-weighted-merge weight.
        arrival_time_s: monotonic clock at which the syncer received
            this fragment. Set by the syncer, not the learner.
    """
    learner_id: str
    round_id: int
    params_delta: list[Tensor]
    tokens_consumed: int
    arrival_time_s: float = 0.0


@dataclass
class MergeResult:
    """Output of one outer-round merge.

    Attributes:
        round_id: the outer-round counter this merge belongs to.
        merged_delta: the token-weighted (or uniform) merged parameter delta.
        learners_merged: sorted ids of the learners whose fragments merged.
        learners_excluded: sorted ids of learners fail-slow-excluded this round.
        elapsed_grace_ms: wall-clock ms elapsed in the grace window since the
            quorum-satisfaction stamp (0.0 when quorum was forced or the round
            finalized before the stamp).
        reason: why the round finalized. One of:
            - "quorum_satisfied": forced finalize via finalize_on_timeout().
            - "grace_expired": the grace window elapsed after quorum.
            - "all_present": every expected, non-fail-slow learner submitted
              and quorum was met, so the round finalized early without waiting
              out the remaining grace window. Only possible when the round was
              started with expected-learner awareness.
        learners_absent: sorted ids of expected learners that were neither
            received nor fail-slow-excluded at finalize time. Always empty when
            the round was started without expected-learner awareness (the
            default), or when the finalize reason is "all_present". Disjoint
            from both learners_merged and learners_excluded.
    """
    round_id: int
    merged_delta: list[Tensor]
    learners_merged: list[str]
    learners_excluded: list[str]
    elapsed_grace_ms: float
    reason: str  # "quorum_satisfied", "grace_expired", "all_present"
    learners_absent: list[str] = field(default_factory=list)


def token_weighted_merge(fragments: Iterable[LearnerFragment]) -> list[Tensor]:
    """Token-weighted parameter-delta merge per Decoupled DiLoCo Algorithm 2.

    merged_p = sum_i (tokens_i * delta_i_p) / sum_i tokens_i

    All fragments must share the same parameter shape list. Caller is
    responsible for ensuring this.
    """
    frags = list(fragments)
    if not frags:
        raise ValueError("token_weighted_merge requires at least one fragment")
    total_tokens = sum(f.tokens_consumed for f in frags)
    if total_tokens <= 0:
        raise ValueError(
            f"token_weighted_merge requires positive total tokens; got {total_tokens}"
        )
    # Validate same number of parameters across fragments.
    n_params = len(frags[0].params_delta)
    for f in frags[1:]:
        if len(f.params_delta) != n_params:
            raise ValueError(
                f"fragment from {f.learner_id} has {len(f.params_delta)} params; "
                f"expected {n_params}"
            )
    merged: list[Tensor] = []
    for p_idx in range(n_params):
        # Use float64 accumulator to avoid drift on long sums; cast back at end.
        first = frags[0].params_delta[p_idx]
        acc = torch.zeros_like(first, dtype=torch.float64)
        for f in frags:
            acc = acc + f.tokens_consumed * f.params_delta[p_idx].to(torch.float64)
        merged.append((acc / float(total_tokens)).to(first.dtype))
    return merged


def uniform_merge(fragments: Iterable[LearnerFragment]) -> list[Tensor]:
    """Token-blind uniform-average merge. Used when
    `config.token_weighted_merge=False`."""
    frags = list(fragments)
    if not frags:
        raise ValueError("uniform_merge requires at least one fragment")
    n_params = len(frags[0].params_delta)
    merged: list[Tensor] = []
    for p_idx in range(n_params):
        first = frags[0].params_delta[p_idx]
        acc = torch.zeros_like(first, dtype=torch.float64)
        for f in frags:
            acc = acc + f.params_delta[p_idx].to(torch.float64)
        merged.append((acc / float(len(frags))).to(first.dtype))
    return merged


@dataclass
class _SyncerState:
    """Mutable syncer state for one outer round in progress."""
    round_id: int
    fragments: dict[str, LearnerFragment] = field(default_factory=dict)
    failslow_excluded: set[str] = field(default_factory=set)
    round_start_s: float = 0.0
    k_satisfied_at_s: Optional[float] = None
    # Expected-learner awareness for this round (Multi-rack 3+/4+ support).
    # None means "expected set unknown" -> the syncer falls back to the
    # quorum-then-full-grace control law (the historical default, preserved
    # byte-for-byte). When a set is supplied, the syncer can finalize early as
    # soon as every expected, non-fail-slow learner has reported AND quorum is
    # met, and it can report which expected learners were absent at finalize.
    expected_learner_ids: Optional[set[str]] = None


class GraceWindowSyncer:
    """In-process state machine that implements Decoupled DiLoCo Algorithm 2.

    This is the syncer-side state machine; it is driven by a runtime that
    receives fragments off the wire (or off a shared queue in unit tests).

    Usage pattern:

        syncer = GraceWindowSyncer(config)
        syncer.start_round(round_id=42)
        for f in incoming_fragments:
            done = syncer.submit(f)
            if done is not None:
                # done is a MergeResult
                break
        else:
            # exhausted incoming without quorum; force decision
            done = syncer.finalize_on_timeout()

    Multi-rack 3+/4+ awareness (optional): pass the round's expected learner
    ids to start_round so the syncer can finalize early once they have all
    reported, instead of always waiting out the full grace window, and report
    which were absent:

        syncer.start_round(round_id=42, expected_learner_ids={"r0", "r1", "r2"})
        # ... as soon as r0, r1, r2 all submit (and quorum is met), submit()/
        # tick() returns a MergeResult with reason == "all_present".

    Default (no expected set) behavior is unchanged: quorum, then the full
    grace window, with an empty `learners_absent`.

    The state machine does not block. A real runtime polls
    `tick()` on a clock; the unit tests drive it deterministically by calling
    `set_clock()`.
    """

    def __init__(
        self,
        quorum_min_learners: int,
        grace_window_ms: int,
        token_weighted: bool = True,
        clock: Callable[[], float] | None = None,
        simulated_merge_delay_ms: int = 0,
        simulated_merge_delay_distribution: str = "constant",
    ) -> None:
        self.quorum_min = quorum_min_learners
        self.grace_window_ms = grace_window_ms
        self.token_weighted = token_weighted
        self._clock: Callable[[], float] = clock or time.monotonic
        self._state: Optional[_SyncerState] = None
        # Phase 2 Week 1 Day 4-7: optional injectable delay inside _finalize.
        # Stress-test for the orchestrator's overlap budget against a
        # cross-rack grace-window wait. The FALCON paper (arXiv:2410.12588)
        # documents inter-node RDMA latency CoV=0.29 as the largest variance
        # among communication paths (Table 2); FALCON does NOT report
        # per-iteration latency percentiles, so the specific base delay value
        # is a stress-test target rather than a literal P99 measurement.
        # See docs/convergence_equivalence_sketch.md (2026-05-23) for the
        # convergence-equivalence proof; an internal FALCON citation audit
        # records the published-variance-anchored parameters.
        # Both synchronous and orchestrator paths invoke _finalize, so the
        # delay applies apples-to-apples: the synchronous reducer blocks for
        # the delay, the orchestrator overlaps it with inner-step compute.
        self.simulated_merge_delay_ms = max(0, int(simulated_merge_delay_ms))
        if simulated_merge_delay_distribution not in ("constant", "bimodal", "long_tail"):
            raise ValueError(
                f"simulated_merge_delay_distribution must be one of "
                f"'constant', 'bimodal', 'long_tail'; "
                f"got {simulated_merge_delay_distribution!r}"
            )
        self.simulated_merge_delay_distribution = simulated_merge_delay_distribution
        # Dedicated RNG for delay sampling. Seeded from os.urandom by
        # default so successive rounds get different bimodal / long-tail
        # samples; tests can replace via set_delay_rng().
        import random as _random
        self._delay_rng: _random.Random = _random.Random()

    def start_round(
        self,
        round_id: int,
        expected_learner_ids: set[str] | None = None,
        total_learners: int | None = None,
    ) -> None:
        """Begin a new outer round.

        Args:
            round_id: monotonic outer-round counter.
            expected_learner_ids: the set of learner (rack) ids this round
                expects to hear from. When provided, the syncer can early-
                finalize the moment every expected, non-fail-slow learner has
                reported and quorum is met (reason "all_present"), instead of
                always waiting out the full grace window. It also enables the
                absentee diagnostic (`MergeResult.learners_absent`). When None
                (the default), behavior is byte-for-byte identical to before:
                quorum, then the full grace window, and an empty
                `learners_absent`.
            total_learners: the expected total learner count for this round.
                Used only to validate the quorum against the round's world
                size. A bare count does NOT enable early-finalize on its own:
                a count cannot name the expected ids, so when
                `expected_learner_ids` is omitted the expected set stays
                unknown and early-finalize stays disabled. If both are given,
                the count must equal the set size.

        Raises:
            ValueError: if `quorum_min_learners` exceeds the round's known
                total (`total_learners`, or `len(expected_learner_ids)` when
                no count is given), since quorum could then never be met; or
                if both `total_learners` and `expected_learner_ids` are given
                and disagree.
        """
        expected: Optional[set[str]] = (
            set(expected_learner_ids) if expected_learner_ids is not None else None
        )
        known_total: Optional[int] = total_learners
        if expected is not None:
            if total_learners is not None and total_learners != len(expected):
                raise ValueError(
                    f"total_learners ({total_learners}) must equal "
                    f"len(expected_learner_ids) ({len(expected)}) when both "
                    f"are provided"
                )
            if known_total is None:
                known_total = len(expected)
        if known_total is not None and self.quorum_min > known_total:
            raise ValueError(
                f"quorum_min_learners ({self.quorum_min}) cannot exceed the "
                f"round's total learners ({known_total}); quorum could never "
                f"be met"
            )
        self._state = _SyncerState(
            round_id=round_id,
            round_start_s=self._clock(),
            expected_learner_ids=expected,
        )

    def mark_failslow(self, learner_id: str) -> None:
        """Exclude a learner from this round (FALCON integration point).
        Idempotent. A failslow-excluded learner's submit() is rejected."""
        if self._state is None:
            raise RuntimeError("mark_failslow called before start_round")
        self._state.failslow_excluded.add(learner_id)

    def submit(self, fragment: LearnerFragment) -> Optional[MergeResult]:
        """Submit a fragment for the current round.

        Returns a MergeResult if this submission completes the round
        (either because we passed quorum AND the grace window has
        already elapsed, or because all known learners reported).
        Otherwise returns None.

        Most callers will not get a MergeResult here; the round usually
        completes via `tick()` or `finalize_on_timeout()` after the
        grace window elapses.
        """
        if self._state is None:
            raise RuntimeError("submit called before start_round")
        if fragment.learner_id in self._state.failslow_excluded:
            return None
        if fragment.round_id != self._state.round_id:
            return None  # stale or future round; ignore
        # Stamp arrival time at the syncer.
        if fragment.arrival_time_s == 0.0:
            fragment.arrival_time_s = self._clock()
        self._state.fragments[fragment.learner_id] = fragment
        # Quorum satisfaction stamp.
        if (
            len(self._state.fragments) >= self.quorum_min
            and self._state.k_satisfied_at_s is None
        ):
            self._state.k_satisfied_at_s = self._clock()
        return self.tick()

    def _all_expected_present(self) -> bool:
        """True when expected-learner awareness is active, quorum is met, and
        every expected, non-fail-slow learner has already submitted a fragment
        for this round.

        Returns False when the expected set is unknown (None), preserving the
        historical quorum-then-full-grace behavior byte-for-byte.
        """
        assert self._state is not None
        expected = self._state.expected_learner_ids
        if expected is None:
            return False
        if self._state.k_satisfied_at_s is None:
            return False  # quorum not yet met
        received = set(self._state.fragments.keys())
        excluded = self._state.failslow_excluded
        # The expected learners we still need to hear from: expected, minus the
        # ones that already reported, minus the ones we've given up on
        # (fail-slow-excluded). Empty -> nothing left to wait for.
        pending = expected - received - excluded
        return not pending

    def tick(self) -> Optional[MergeResult]:
        """Check whether the round can complete now. Returns a MergeResult
        if it can finalize (all expected learners present, or grace elapsed);
        otherwise None."""
        if self._state is None:
            return None
        if self._state.k_satisfied_at_s is None:
            return None
        now = self._clock()
        elapsed_ms = (now - self._state.k_satisfied_at_s) * 1000.0
        # Early-finalize: when this round knows its expected learners and every
        # expected, non-fail-slow learner has reported, finalize immediately
        # without waiting out the remaining grace window. When the expected set
        # is unknown this is always False, so the historical quorum-then-full-
        # grace path below is taken byte-for-byte.
        if self._all_expected_present():
            return self._finalize(reason="all_present", elapsed_grace_ms=elapsed_ms)
        if elapsed_ms >= self.grace_window_ms:
            return self._finalize(reason="grace_expired", elapsed_grace_ms=elapsed_ms)
        return None

    def finalize_on_timeout(self) -> MergeResult:
        """Force the round to complete; the caller has decided that no
        more learners will report.

        If quorum is not satisfied, raises RuntimeError. The runtime is
        responsible for either waiting longer or escalating to operator
        intervention; the syncer does not silently merge below quorum."""
        if self._state is None:
            raise RuntimeError("finalize_on_timeout called before start_round")
        if len(self._state.fragments) < self.quorum_min:
            raise RuntimeError(
                f"finalize_on_timeout: only {len(self._state.fragments)} learners "
                f"reported, need {self.quorum_min}; aborting round "
                f"{self._state.round_id}"
            )
        now = self._clock()
        elapsed_ms = (
            (now - self._state.k_satisfied_at_s) * 1000.0
            if self._state.k_satisfied_at_s is not None
            else 0.0
        )
        return self._finalize(reason="quorum_satisfied", elapsed_grace_ms=elapsed_ms)

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Test hook: replace the monotonic clock with a callable. Used by
        the deterministic unit tests."""
        self._clock = clock

    def set_delay_rng(self, rng: Random) -> None:
        """Test hook: replace the delay-sampling RNG. Used by the
        deterministic unit tests so the bimodal / long-tail distribution
        sampling is reproducible."""
        self._delay_rng = rng

    def _sample_delay_ms(self) -> float:
        """Sample the actual per-round delay from the configured
        distribution. The base value `simulated_merge_delay_ms` is
        interpreted as follows:

        - constant (default): always inject exactly base ms.
        - bimodal: 95% of rounds inject 0.9 * base ms (typical mode);
          5% of rounds inject 2.9 * base ms (straggler mode). This
          mixture has mean ~= base and coefficient of variation
          ~= 0.29, matching the FALCON Table 2 inter-node RDMA CoV
          measurement (Wu et al., arXiv:2410.12588). The 5% straggler
          rate aligns with the secondary distributed-training
          literature's typical 1-10% per-iteration straggler
          frequency, and is materially more conservative than the
          prior 80/20 stress shape used through 2026-05-23 Track D.
        - long_tail: log-normal calibrated to FALCON's CoV=0.29 on
          inter-node RDMA. sigma = sqrt(log(1 + 0.29^2)) ~= 0.285;
          mu = log(base) - sigma^2 / 2. Median sits near base; the
          right tail inherits its shape parameter from production
          RDMA measurements. (FALCON reports a scalar CoV, not the
          full distribution shape, so log-normal-shape itself is a
          literature default for worker runtime rather than a
          FALCON-validated shape. The variance, however, is now
          FALCON-anchored.)

        Distributions other than 'constant' remain stress-test shapes;
        the production-grade headline measurements use 'constant' for
        reproducibility. The bimodal and long_tail parameter sets
        landed 2026-05-23 are anchored to FALCON's only published
        production variance number (inter-node RDMA CoV=0.29); they
        are no longer 4-5x miscalibrated stress heuristics. The
        +28.58% Stage D-proper Lambda V100 cross-network result
        remains the most defensible production-grounded headline.

        History: through 2026-05-23, bimodal was 80% / 20% at 50ms /
        base ms (a per-iteration restatement of FALCON's job-level
        "20% of fail-slow-affected jobs delayed >50%" statistic, at
        the wrong unit of analysis); long_tail was log-normal with
        mean base/2 and sigma 1.0 (CoV ~= 1.31). Both were replaced with
        the FALCON-CoV-anchored versions above so that the simulated
        distributions match FALCON's reported inter-node RDMA variance.
        """
        import math

        base = self.simulated_merge_delay_ms
        if self.simulated_merge_delay_distribution == "constant":
            return float(base)
        if self.simulated_merge_delay_distribution == "bimodal":
            # FALCON-CoV-anchored 95/5 mixture: 95% at 0.9*base, 5%
            # at 2.9*base. Mean ~= base; CoV ~= 0.29 (matches FALCON
            # Table 2 inter-node RDMA variance).
            if self._delay_rng.random() < 0.95:
                return float(base) * 0.9
            return float(base) * 2.9
        if self.simulated_merge_delay_distribution == "long_tail":
            # FALCON-CoV-anchored log-normal: sigma chosen so the
            # distribution's CoV matches FALCON Table 2 inter-node
            # RDMA CoV=0.29; mu chosen so the distribution's median
            # sits near base.
            cov = 0.29
            sigma = math.sqrt(math.log(1.0 + cov * cov))
            mu = math.log(max(1.0, float(base))) - sigma * sigma / 2.0
            return self._delay_rng.lognormvariate(mu, sigma)
        return float(base)

    def _finalize(self, reason: str, elapsed_grace_ms: float) -> MergeResult:
        assert self._state is not None
        # Phase 2 Week 1 Day 4-7: optional simulated grace-window wait.
        # Applied here so BOTH synchronous and orchestrator paths see the
        # same delay; the orchestrator path overlaps it with inner-step
        # compute via its asyncio task while the synchronous path blocks
        # the training thread.
        if self.simulated_merge_delay_ms > 0:
            delay_ms = self._sample_delay_ms()
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
        merger = token_weighted_merge if self.token_weighted else uniform_merge
        fragments = list(self._state.fragments.values())
        merged = merger(fragments)
        # Absentee diagnostic: expected learners neither received nor
        # fail-slow-excluded at finalize. Empty when the expected set is
        # unknown (the historical default) -- a `learners_absent` of [] then
        # carries the same "we don't track this" meaning it always has.
        expected = self._state.expected_learner_ids
        if expected is None:
            learners_absent: list[str] = []
        else:
            received = set(self._state.fragments.keys())
            learners_absent = sorted(
                expected - received - self._state.failslow_excluded
            )
        result = MergeResult(
            round_id=self._state.round_id,
            merged_delta=merged,
            learners_merged=sorted(self._state.fragments.keys()),
            learners_excluded=sorted(self._state.failslow_excluded),
            elapsed_grace_ms=elapsed_grace_ms,
            reason=reason,
            learners_absent=learners_absent,
        )
        self._state = None
        return result
