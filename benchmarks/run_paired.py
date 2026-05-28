"""Config-driven paired-run benchmark driver for tsugi-mend.

Implements the paired-run protocol in ``docs/benchmark_protocol.md``:

    Same workload, same checkpoint, same seed, same data, different
    synchronization. Only the cross-rack merge path differs between the
    ``baseline`` (vanilla synchronous reducer) and the ``sdk`` (mend
    concurrent outer-step) path.

For each path the driver:

1. Trains the SAME model on the SAME seed + deterministic data on every rank,
   syncing parameters across ranks every ``sync_period_steps`` inner steps
   through the SDK's Decoupled-DiLoCo ``token_weighted_merge`` reducer.
   - ``baseline``: drives ``GraceWindowSyncer`` synchronously (the training
     thread blocks across the grace window / merge).
   - ``sdk``: drives the ``ConcurrentOuterStep`` orchestrator via
     ``mend_init`` (the merge runs on the asyncio loop thread; the training
     thread overlaps it).
   Both paths apply the SAME token-weighted merged delta at the SAME
   logical boundary, so the per-step loss trajectory must be identical.
2. Records the per-step loss (for the bit-exact check) and per-step wall
   time (for tokens/s).

The driver then:

- **asserts bit-exact loss equivalence** between baseline and SDK in the
  default (lossless) mode -- the load-bearing invariant. It verifies this;
  it does not assume it.
- summarizes steady-state tokens/s for both paths and reports the uplift
  with a paired-bootstrap 95% CI.
- writes a public-safe result bundle (``result.json``) under
  ``benchmarks/results/<cell>/``.

Scaling: the SAME driver runs the $0 cheap cell (CPU / gloo / tiny MLP), a
torchrun/env:// launch of that cheap cell, and a GPU-deferred Hugging Face/FSDP
cell. The real multi-node run requires provisioning a real GPU cluster and is
out of scope here. The harness implements the config and workload path but does
not provision compute or run it locally.

Run the cheap cell:

    python benchmarks/run_paired.py --cell cpu_gloo_2rank_mlp

See ``benchmarks/README.md`` for the full run + expected-output contract.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

# Allow `python benchmarks/run_paired.py` from the repo root without an
# editable install (mirrors tests/conftest.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn as nn  # noqa: E402

from benchmarks.metrics import (  # noqa: E402
    bit_exact_equal,
    bootstrap_uplift_ci,
    steady_state,
)
from tsugi_mend import MendConfig  # noqa: E402
from tsugi_mend.concurrent import FragmentProvider  # noqa: E402
from tsugi_mend.reducer import (  # noqa: E402
    GraceWindowSyncer,
    LearnerFragment,
)
# Import the fully-typed runtime implementations directly (the top-level
# `tsugi_mend.mend_init` / `mend_shutdown` are deliberately untyped lazy
# facades; under mypy --strict, calling them flags no-untyped-call).
from tsugi_mend.runtime import get_runtime, mend_init, mend_shutdown  # noqa: E402

RESULTS_ROOT = _REPO_ROOT / "benchmarks" / "results"


# ----------------------------------------------------------------------
# Workload definition (public-safe synthetic regression task)
# ----------------------------------------------------------------------


class _MLP(nn.Module):
    """Deterministic regression MLP used for the cheap CPU cell.

    Public-safe synthetic workload: no tokenizer, no dataset download, no
    network. The "tokens" reported by the protocol are the synthetic
    feature count per micro-batch (batch * in_dim), which is all the
    tokens/s ratio needs. The cheap cell sizes this so a single inner step
    costs a few milliseconds, large enough that a handful of inner steps can
    overlap the simulated cross-rack merge delay (otherwise sub-millisecond
    steps could not absorb the delay and the overlap mechanism would not be
    measurable at $0).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


@dataclass
class BenchConfig:
    """Paired-run configuration. Args-only scaling from cheap cell to a
    real multi-node config."""

    cell: str = "cpu_gloo_2rank_mlp"
    launch: str = "selfspawn"           # selfspawn (default), torchrun, or auto
    backend: str = "gloo"               # gloo (CPU) for the cheap cell; nccl for GPU
    ranks: int = 2                      # process world size; real FSDP merges per-node shards
    steps: int = 120                    # total inner steps
    warmup_steps: int = 20              # protocol: exclude warmup from steady state
    sync_period_steps: int = 10         # param-sync (outer-round) cadence
    # Inner-step lag between submitting an outer round and applying its merged
    # delta. Both paths apply at this lag (Decoupled-DiLoCo "late apply"), so
    # the parameter trajectories coincide (bit-exact); the SDK overlaps the
    # merge delay across these lag steps while the baseline blocks. Must be
    # < sync_period_steps so consecutive rounds do not overlap.
    apply_lag_steps: int = 4
    seed: int = 20240527
    # Synthetic-workload shape (cheap cell). A real cell would instead set
    # model/tokenizer/dataset HF identifiers; those fields are recorded in
    # the bundle's `workload` block but the cheap path uses the MLP. Sized so
    # a single inner step costs a few ms on CPU (see _MLP docstring).
    batch: int = 256
    in_dim: int = 512
    hidden: int = 1024
    out_dim: int = 128
    sequence_length: int = 128
    lr: float = 0.05
    # Cross-rack merge knobs (apply apples-to-apples to BOTH paths via the
    # SDK's GraceWindowSyncer._finalize). The simulated merge delay is set so
    # that apply_lag_steps inner steps of compute comfortably exceed it: the
    # SDK overlaps the delay across those lag steps while the baseline blocks
    # the training thread on it. This exposes a real overlap-driven tokens/s
    # delta at $0; the baseline blocks, the SDK overlaps.
    grace_window_ms: int = 0
    simulated_merge_delay_ms: int = 12
    bootstrap_resamples: int = 10000
    # For a real (non-cheap) cell, the HF identifiers + hardware label are
    # passed through into the bundle for the reproduction contract. None on
    # the cheap synthetic cell.
    model_id: Optional[str] = None
    tokenizer_id: Optional[str] = None
    dataset_id: Optional[str] = None
    hardware_label: str = "local CPU (gloo); $0 cheap reproducible cell"
    # Internal: rendezvous file store path (set by the parent before spawn).
    _store_path: Optional[str] = field(default=None, repr=False)


@dataclass
class _DistributedRunContext:
    rank: int
    world_size: int
    object_group: Any
    object_world_size: int
    learner_id: str
    local_rank: int = 0
    local_world_size: int = 1
    fsdp_group: Any = None
    groups_to_destroy: list[Any] = field(default_factory=list)


@dataclass
class _TrainingState:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    tokens_per_step: int
    kind: str
    device: torch.device
    vocab_size: Optional[int] = None
    tokenizer: Any = None
    dataset_texts: Optional[list[str]] = None


def _is_real_cell(cfg: BenchConfig) -> bool:
    return cfg.model_id is not None


def _workload_kind(cfg: BenchConfig) -> str:
    return "huggingface" if _is_real_cell(cfg) else "synthetic-mlp-regression"


def _reproducible_label(cfg: BenchConfig) -> str:
    if _is_real_cell(cfg) or cfg.backend != "gloo":
        return "real-hardware (requires a GPU cluster)"
    return "cheap"


def _tokens_per_step(cfg: BenchConfig) -> int:
    if _is_real_cell(cfg):
        return cfg.batch * cfg.sequence_length
    return cfg.batch * cfg.in_dim


def _optimizer_label(cfg: BenchConfig) -> str:
    return "AdamW" if _is_real_cell(cfg) else "SGD"


def _quorum_min_learners_for_bundle(cfg: BenchConfig) -> int:
    if _is_real_cell(cfg) and cfg.launch == "torchrun":
        local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
        return max(1, cfg.ranks // local_world_size)
    return cfg.ranks


def _torchrun_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _resolve_launch(cfg: BenchConfig) -> str:
    if cfg.launch == "auto":
        return "torchrun" if _torchrun_env_present() else "selfspawn"
    if cfg.launch not in {"selfspawn", "torchrun"}:
        raise ValueError(
            f"launch must be 'selfspawn', 'torchrun', or 'auto'; got {cfg.launch!r}"
        )
    return cfg.launch


def _validate_common_config(cfg: BenchConfig) -> None:
    if cfg.apply_lag_steps >= cfg.sync_period_steps:
        raise ValueError(
            f"apply_lag_steps ({cfg.apply_lag_steps}) must be < "
            f"sync_period_steps ({cfg.sync_period_steps}) so consecutive "
            f"outer rounds do not overlap"
        )
    if cfg.ranks < 1:
        raise ValueError(f"ranks must be >= 1; got {cfg.ranks}")
    if cfg.steps <= cfg.warmup_steps:
        raise ValueError(
            f"steps ({cfg.steps}) must be greater than warmup_steps "
            f"({cfg.warmup_steps})"
        )


def _validate_real_cell_requirements(cfg: BenchConfig, launch: str) -> None:
    if not _is_real_cell(cfg):
        return
    missing: list[str] = []
    if launch != "torchrun":
        missing.append("--launch torchrun")
    if cfg.backend != "nccl":
        missing.append("--backend nccl")
    if not torch.cuda.is_available():
        missing.append("CUDA")
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "real Hugging Face/FSDP cell requires CUDA, --launch torchrun, "
            "--backend nccl, and the optional tsugi-mend[real-cell] extra; "
            f"missing or unavailable: {missing_text}"
        )
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "real Hugging Face/FSDP cell requires the optional "
            "tsugi-mend[real-cell] extra; install it before running a real cell"
        ) from exc


# ----------------------------------------------------------------------
# Per-rank paired training (runs in each spawned or torchrun worker)
# ----------------------------------------------------------------------


def _make_model(cfg: BenchConfig) -> _MLP:
    # Seed BEFORE constructing so every rank + both paths start identical.
    torch.manual_seed(cfg.seed)
    return _MLP(cfg.in_dim, cfg.hidden, cfg.out_dim)


def _batch_for_step(cfg: BenchConfig, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Deterministic per-step data, IDENTICAL on every rank and both paths.
    # (Data-parallel replicas see the same batch in this synthetic cell;
    # what differs between ranks is only the local SGD trajectory via the
    # token-weighted merge, exactly as the protocol's "same data" rule wants.)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed + 1000 + step)
    x = torch.randn(cfg.batch, cfg.in_dim, generator=gen)
    target = torch.randn(cfg.batch, cfg.out_dim, generator=gen)
    return x, target


def _make_training_state(cfg: BenchConfig, ctx: _DistributedRunContext) -> _TrainingState:
    if _is_real_cell(cfg):
        return _make_hf_training_state(cfg, ctx)
    model = _make_model(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    return _TrainingState(
        model=model,
        optimizer=opt,
        tokens_per_step=_tokens_per_step(cfg),
        kind="synthetic-mlp-regression",
        device=torch.device("cpu"),
    )


def _make_hf_training_state(cfg: BenchConfig, ctx: _DistributedRunContext) -> _TrainingState:
    # Imports stay inside the real-cell path so the default package and the
    # $0 CPU cell do not require the optional benchmark stack.
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert cfg.model_id is not None
    device = torch.device("cuda", ctx.local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    tokenizer_id = cfg.tokenizer_id or cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = int(len(tokenizer))
    if vocab_size <= 1:
        raise RuntimeError(f"tokenizer {tokenizer_id!r} reports invalid vocab size {vocab_size}")
    dataset_texts = _load_dataset_texts(cfg.dataset_id) if cfg.dataset_id else None

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    model.train()
    model.to(device)
    wrapped = FSDP(model, process_group=ctx.fsdp_group)
    opt = torch.optim.AdamW(wrapped.parameters(), lr=cfg.lr)
    return _TrainingState(
        model=wrapped,
        optimizer=opt,
        tokens_per_step=_tokens_per_step(cfg),
        kind="huggingface",
        device=device,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        dataset_texts=dataset_texts,
    )


def _load_dataset_texts(dataset_id: str, sample_count: int = 64) -> list[str]:
    # Optional and lazy: only users who set dataset_id need the datasets extra
    # or a dataset download. The default real cell uses generated token IDs.
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "dataset_id requires the optional datasets dependency; install "
            "tsugi-mend[real-cell] or omit dataset_id for deterministic token IDs"
        ) from exc

    dataset = load_dataset(dataset_id, split=f"train[:{sample_count}]")
    texts: list[str] = []
    for row in dataset:
        if isinstance(row, dict):
            if isinstance(row.get("text"), str):
                texts.append(row["text"])
            else:
                for value in row.values():
                    if isinstance(value, str):
                        texts.append(value)
                        break
        if len(texts) >= sample_count:
            break
    if not texts:
        raise RuntimeError(f"dataset {dataset_id!r} did not expose a string text column")
    return texts


def _loss_for_step(cfg: BenchConfig, state: _TrainingState, step: int) -> torch.Tensor:
    if state.kind == "synthetic-mlp-regression":
        x, target = _batch_for_step(cfg, step)
        # state.model is a Union including the HF model whose forward returns Any,
        # so mypy can't prove the synthetic MLP branch's result is a real Tensor.
        # It is at runtime (MLP forward returns Tensor; subtraction/pow/mean stay Tensor).
        return (state.model(x) - target).pow(2).mean()  # type: ignore[no-any-return]

    assert state.vocab_size is not None
    torch.manual_seed(cfg.seed + 2000 + step)
    torch.cuda.manual_seed_all(cfg.seed + 2000 + step)
    if state.dataset_texts:
        assert state.tokenizer is not None
        texts = [
            state.dataset_texts[(step * cfg.batch + idx) % len(state.dataset_texts)]
            for idx in range(cfg.batch)
        ]
        encoded = state.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=cfg.sequence_length,
        )
        input_ids = encoded["input_ids"].to(state.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(state.device, non_blocking=True)
    else:
        gen = torch.Generator()
        gen.manual_seed(cfg.seed + 1000 + step)
        input_ids = torch.randint(
            low=0,
            high=state.vocab_size,
            size=(cfg.batch, cfg.sequence_length),
            generator=gen,
            dtype=torch.long,
        ).to(state.device, non_blocking=True)
        attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    output = state.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss: torch.Tensor = output.loss
    return loss


def _snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _build_fragment(
    model: nn.Module,
    prev: list[torch.Tensor],
    learner_id: str,
    round_id: int,
    tokens: int,
) -> LearnerFragment:
    deltas = [
        (cur.detach() - old.to(device=cur.device, dtype=cur.dtype)).detach().cpu()
        for cur, old in zip(model.parameters(), prev)
    ]
    return LearnerFragment(
        learner_id=learner_id,
        round_id=round_id,
        params_delta=deltas,
        tokens_consumed=tokens,
    )


def _apply_merged(model: nn.Module, merged: list[torch.Tensor]) -> None:
    """Add the merged outer-step delta into the current params (in place).

    Both paths call this with the SAME ``merged`` (same token_weighted_merge
    over the same gathered fragments) at the SAME logical step, so the
    resulting parameters are identical -> the next step's loss is identical
    -> bit-exact. The application is additive (``p += merged``) so it is well
    defined regardless of how many inner steps elapsed since the snapshot was
    taken (the Decoupled-DiLoCo "late apply" the orchestrator is designed
    for); both paths apply at the same lag, isolating the overlap benefit.
    """
    with torch.no_grad():
        for p, m in zip(model.parameters(), merged):
            p.add_(m.to(device=p.device, dtype=p.dtype))


def _gather_fragments(
    local: LearnerFragment,
    ranks: int,
    object_group: Any,
) -> list[LearnerFragment]:
    gathered: list[Optional[LearnerFragment]] = [None] * ranks
    dist.all_gather_object(gathered, local, group=object_group)
    return [f for f in gathered if f is not None]


def _synchronous_merge(
    syncer: GraceWindowSyncer, fragments: list[LearnerFragment], round_id: int
) -> list[torch.Tensor]:
    """Drive the syncer to completion synchronously (blocks on the simulated
    merge delay inside _finalize). submit() can itself finalize the round when
    quorum is met and the grace window has elapsed (e.g. grace_window_ms=0),
    in which case it returns the MergeResult and clears the syncer state; only
    call finalize_on_timeout() if the round is still open."""
    syncer.start_round(round_id=round_id)
    result = None
    for f in fragments:
        maybe = syncer.submit(f)
        if maybe is not None:
            result = maybe
    if result is None:
        result = syncer.finalize_on_timeout()
    return result.merged_delta


def _run_baseline(
    cfg: BenchConfig,
    ctx: _DistributedRunContext,
) -> tuple[list[float], list[float]]:
    """Vanilla synchronous reducer path. Returns (per-step loss, per-step ms).

    At each outer-round step the training thread BLOCKS across the grace
    window + simulated merge delay (GraceWindowSyncer._finalize sleeps on the
    training thread), then defers application of the merged delta by
    ``apply_lag_steps`` inner steps -- exactly the lag the SDK path applies at,
    so the parameter trajectories coincide and loss is bit-exact. The only
    difference vs the SDK path is WHEN the delay is paid: here it blocks the
    training thread; in the SDK path it is overlapped off-thread.
    """
    state = _make_training_state(cfg, ctx)
    model = state.model
    opt = state.optimizer
    syncer = GraceWindowSyncer(
        quorum_min_learners=ctx.object_world_size,
        grace_window_ms=cfg.grace_window_ms,
        token_weighted=True,
        simulated_merge_delay_ms=cfg.simulated_merge_delay_ms,
    )
    tokens_per_step = state.tokens_per_step
    prev = _snapshot(model)
    losses: list[float] = []
    step_ms: list[float] = []
    pending: Optional[tuple[int, list[torch.Tensor]]] = None  # (apply_at_step, delta)
    for step in range(cfg.steps):
        t0 = time.perf_counter()
        loss = _loss_for_step(cfg, state, step)
        # HF forward (transformers, override-silenced) makes Tensor.backward read as
        # untyped under --strict; it is a real scalar Tensor at runtime.
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

        if step > 0 and step % cfg.sync_period_steps == 0:
            local = _build_fragment(model, prev, ctx.learner_id, step, tokens_per_step)
            fragments = _gather_fragments(local, ctx.object_world_size, ctx.object_group)
            merged = _synchronous_merge(syncer, fragments, round_id=step)
            pending = (step + cfg.apply_lag_steps, merged)
            prev = _snapshot(model)

        if pending is not None and step >= pending[0]:
            _apply_merged(model, pending[1])
            pending = None
        step_ms.append((time.perf_counter() - t0) * 1000.0)
    return losses, step_ms


def _distributed_fragment_provider_factory(
    local: LearnerFragment,
    ranks: int,
    object_group: Any,
) -> FragmentProvider:
    def provider() -> "asyncio.Queue[LearnerFragment]":
        queue: asyncio.Queue[LearnerFragment] = asyncio.Queue()
        fragments = _gather_fragments(local, ranks, object_group)
        for f in fragments:
            queue.put_nowait(f)
        return queue

    return provider


def _run_sdk(
    cfg: BenchConfig,
    ctx: _DistributedRunContext,
) -> tuple[list[float], list[float]]:
    """mend concurrent outer-step path. Returns (per-step loss, per-step ms).

    At each outer-round step the merge is SUBMITTED to the ConcurrentOuterStep
    orchestrator and runs on the asyncio loop thread; the training thread keeps
    issuing inner steps (overlapping the simulated merge delay off-thread). The
    merged delta is collected and applied ``apply_lag_steps`` inner steps later
    -- the SAME lag the baseline applies at -- so the parameter trajectories
    coincide and loss is bit-exact. This isolates exactly the overlap benefit:
    the SDK absorbs the delay into inner-step compute that the synchronous
    baseline spends blocking.
    """
    state = _make_training_state(cfg, ctx)
    model = state.model
    opt = state.optimizer
    config = MendConfig(
        quorum_min_learners=ctx.object_world_size,
        grace_window_ms=cfg.grace_window_ms,
        token_weighted_merge=True,
        sync_period_steps=cfg.sync_period_steps,
        momentum_sync_period_steps=cfg.sync_period_steps * 4,
        async_tp_enabled=False,
        concurrent_outer_step=True,
        simulated_merge_delay_ms=cfg.simulated_merge_delay_ms,
        sideband_peers=(),
        diagnostics_dir=None,
    )
    tokens_per_step = state.tokens_per_step
    mend_init(model, config, rank_id=ctx.learner_id)
    try:
        runtime = get_runtime(model)
        prev = _snapshot(model)
        losses: list[float] = []
        step_ms: list[float] = []
        apply_at: Optional[int] = None  # step at which to collect+apply
        for step in range(cfg.steps):
            t0 = time.perf_counter()
            runtime.step_begin(step)
            loss = _loss_for_step(cfg, state, step)
            # See _run_baseline: HF-forward-derived loss makes Tensor.backward
            # resolve as an untyped call under --strict.
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            runtime.step_end(step)

            if step > 0 and step % cfg.sync_period_steps == 0:
                local = _build_fragment(model, prev, ctx.learner_id, step, tokens_per_step)
                runtime.outer_step_begin(
                    round_id=step,
                    fragment_provider=_distributed_fragment_provider_factory(
                        local,
                        ctx.object_world_size,
                        ctx.object_group,
                    ),
                )
                apply_at = step + cfg.apply_lag_steps
                prev = _snapshot(model)

            if apply_at is not None and step >= apply_at:
                result = _collect(runtime)
                _apply_merged(model, result.merged_delta)
                apply_at = None
            step_ms.append((time.perf_counter() - t0) * 1000.0)
        # Drain any still-pending round so shutdown is clean.
        if apply_at is not None:
            result = _collect(runtime)
            _apply_merged(model, result.merged_delta)
        return losses, step_ms
    finally:
        mend_shutdown(model)


def _collect(runtime: Any, timeout_s: float = 10.0) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = runtime.outer_step_collect()
        if result is not None:
            return result
        time.sleep(0.001)
    raise TimeoutError("outer-step merge did not complete within timeout")


def _is_real_group(group: Any) -> bool:
    return group is not None and group != dist.GroupMember.NON_GROUP_MEMBER


def _create_rank_partition_group(
    partitions: list[list[int]],
    *,
    backend: str,
    rank: int,
) -> tuple[Any, int, list[Any]]:
    selected_group = None
    selected_size = 0
    owned_groups: list[Any] = []
    for ranks in partitions:
        group = dist.new_group(ranks=ranks, backend=backend)
        if rank in ranks and _is_real_group(group):
            selected_group = group
            selected_size = len(ranks)
            owned_groups.append(group)
    if selected_group is None:
        raise RuntimeError(f"rank {rank} did not join any partition group")
    return selected_group, selected_size, owned_groups


def _make_distributed_context(cfg: BenchConfig, rank: int) -> _DistributedRunContext:
    world_size = dist.get_world_size()
    if _is_real_cell(cfg):
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank % local_world_size)))
        if local_world_size < 1:
            raise RuntimeError(f"LOCAL_WORLD_SIZE must be positive; got {local_world_size}")
        if world_size % local_world_size != 0:
            raise RuntimeError(
                f"WORLD_SIZE ({world_size}) must be divisible by LOCAL_WORLD_SIZE "
                f"({local_world_size}) for the per-node FSDP real cell"
            )
        node_count = world_size // local_world_size
        node_rank = rank // local_world_size

        # For FSDP, shard within each node. All ranks create the node groups in
        # the same order, then each rank keeps the group for its own node.
        node_partitions = [
            list(range(node * local_world_size, (node + 1) * local_world_size))
            for node in range(node_count)
        ]
        fsdp_group, _, fsdp_groups = _create_rank_partition_group(
            node_partitions,
            backend=cfg.backend,
            rank=rank,
        )

        # For LearnerFragment exchange, gather same-local-rank shards across
        # nodes on a dedicated gloo group. This keeps Python object gather off
        # NCCL and lines up FSDP shard shapes across learners.
        shard_partitions = [
            [node * local_world_size + local for node in range(node_count)]
            for local in range(local_world_size)
        ]
        object_group, object_world_size, object_groups = _create_rank_partition_group(
            shard_partitions,
            backend="gloo",
            rank=rank,
        )
        return _DistributedRunContext(
            rank=rank,
            world_size=world_size,
            object_group=object_group,
            object_world_size=object_world_size,
            learner_id=f"node-{node_rank}/local-rank-{local_rank}",
            local_rank=local_rank,
            local_world_size=local_world_size,
            fsdp_group=fsdp_group,
            groups_to_destroy=fsdp_groups + object_groups,
        )

    object_group = dist.new_group(backend="gloo")
    return _DistributedRunContext(
        rank=rank,
        world_size=world_size,
        object_group=object_group,
        object_world_size=world_size,
        learner_id=f"rank-{rank}",
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))),
        groups_to_destroy=[object_group],
    )


def _destroy_context(ctx: _DistributedRunContext) -> None:
    for group in reversed(ctx.groups_to_destroy):
        if _is_real_group(group):
            dist.destroy_process_group(group)


def _run_rank_sequence(
    cfg: BenchConfig,
    ctx: _DistributedRunContext,
) -> tuple[list[float], list[float], list[float], list[float]]:
    base_losses, base_ms = _run_baseline(cfg, ctx)
    dist.barrier()
    sdk_losses, sdk_ms = _run_sdk(cfg, ctx)
    dist.barrier()
    return base_losses, base_ms, sdk_losses, sdk_ms


def _worker(rank: int, cfg: BenchConfig, return_dict: Any) -> None:
    assert cfg._store_path is not None
    # FileStore is a real public torch.distributed symbol but is not in the
    # bundled type stubs' explicit __all__; scope the ignore narrowly.
    store = dist.FileStore(cfg._store_path, cfg.ranks)  # type: ignore[attr-defined]
    dist.init_process_group(
        backend=cfg.backend, store=store, rank=rank, world_size=cfg.ranks
    )
    ctx = _make_distributed_context(cfg, rank)
    try:
        base_losses, base_ms, sdk_losses, sdk_ms = _run_rank_sequence(cfg, ctx)
        if rank == 0:
            return_dict["baseline_losses"] = base_losses
            return_dict["baseline_step_ms"] = base_ms
            return_dict["sdk_losses"] = sdk_losses
            return_dict["sdk_step_ms"] = sdk_ms
    finally:
        _destroy_context(ctx)
        dist.destroy_process_group()


# ----------------------------------------------------------------------
# Orchestration + bundle emission (parent process)
# ----------------------------------------------------------------------


def _build_result_bundle(
    cfg: BenchConfig,
    base_losses: list[float],
    base_ms: list[float],
    sdk_losses: list[float],
    sdk_ms: list[float],
) -> dict[str, Any]:
    bit_exact = bit_exact_equal(base_losses, sdk_losses)
    tokens_per_step = _tokens_per_step(cfg)
    base_summary = steady_state(base_ms, tokens_per_step, cfg.warmup_steps)
    sdk_summary = steady_state(sdk_ms, tokens_per_step, cfg.warmup_steps)
    ci = bootstrap_uplift_ci(
        base_ms[cfg.warmup_steps:],
        sdk_ms[cfg.warmup_steps:],
        n_resamples=cfg.bootstrap_resamples,
        seed=cfg.seed,
    )

    return {
        "schema_version": "1.0",
        "cell": cfg.cell,
        "reproducible": _reproducible_label(cfg),
        "protocol": "docs/benchmark_protocol.md",
        "hardware": {
            "label": cfg.hardware_label,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": cfg.backend,
            "ranks": cfg.ranks,
            "launch": cfg.launch,
        },
        "workload": {
            "kind": _workload_kind(cfg),
            "model_id": cfg.model_id,
            "tokenizer_id": cfg.tokenizer_id,
            "dataset_id": cfg.dataset_id,
            "batch": cfg.batch,
            "in_dim": cfg.in_dim,
            "hidden": cfg.hidden,
            "out_dim": cfg.out_dim,
            "sequence_length": cfg.sequence_length,
            "tokens_per_step": tokens_per_step,
            "optimizer": _optimizer_label(cfg),
            "lr": cfg.lr,
            "seed": cfg.seed,
        },
        "sdk_config": {
            "quorum_min_learners": _quorum_min_learners_for_bundle(cfg),
            "grace_window_ms": cfg.grace_window_ms,
            "sync_period_steps": cfg.sync_period_steps,
            "apply_lag_steps": cfg.apply_lag_steps,
            "simulated_merge_delay_ms": cfg.simulated_merge_delay_ms,
            "outer_step_compression_mode": "none",
            "concurrent_outer_step": True,
            "token_weighted_merge": True,
        },
        "run": {
            "steps": cfg.steps,
            "warmup_steps": cfg.warmup_steps,
            "n_steady_steps": base_summary.n_steps_steady,
        },
        "bit_exact_loss_equivalence": {
            "passed": bit_exact,
            "method": "elementwise IEEE-754 equality over per-step loss "
            "(rank 0), default lossless mode",
            "n_steps_compared": len(base_losses),
            "first_baseline_loss": base_losses[0] if base_losses else None,
            "last_baseline_loss": base_losses[-1] if base_losses else None,
            "max_abs_loss_diff": max(
                (abs(a - b) for a, b in zip(base_losses, sdk_losses)), default=0.0
            ),
        },
        "metrics": {
            "baseline": {
                "tokens_per_second": base_summary.tokens_per_second,
                "mean_step_time_ms": base_summary.mean_step_time_ms,
                "p50_step_time_ms": base_summary.p50_step_time_ms,
                "p95_step_time_ms": base_summary.p95_step_time_ms,
                "p99_step_time_ms": base_summary.p99_step_time_ms,
            },
            "sdk": {
                "tokens_per_second": sdk_summary.tokens_per_second,
                "mean_step_time_ms": sdk_summary.mean_step_time_ms,
                "p50_step_time_ms": sdk_summary.p50_step_time_ms,
                "p95_step_time_ms": sdk_summary.p95_step_time_ms,
                "p99_step_time_ms": sdk_summary.p99_step_time_ms,
            },
            "uplift": {
                "tokens_per_second_pct": ci.point_estimate_pct,
                "ci95_low_pct": ci.ci_low_pct,
                "ci95_high_pct": ci.ci_high_pct,
                "n_paired_steps": ci.n_paired_steps,
                "n_bootstrap_resamples": ci.n_resamples,
                "confidence": ci.confidence,
            },
        },
    }


def _run_cell_selfspawn(cfg: BenchConfig) -> dict[str, Any]:
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()
    with tempfile.TemporaryDirectory(prefix="tsugi_mend_bench_") as tmp:
        cfg._store_path = os.path.join(tmp, "rendezvous_store")
        procs = []
        for rank in range(cfg.ranks):
            p = mp.Process(target=_worker, args=(rank, cfg, return_dict))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker exited with code {p.exitcode}")

    return _build_result_bundle(
        cfg,
        list(return_dict["baseline_losses"]),
        list(return_dict["baseline_step_ms"]),
        list(return_dict["sdk_losses"]),
        list(return_dict["sdk_step_ms"]),
    )


def _run_cell_torchrun(cfg: BenchConfig) -> Optional[dict[str, Any]]:
    if not _torchrun_env_present():
        raise RuntimeError(
            "--launch torchrun requires a torchrun environment with RANK and WORLD_SIZE"
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if cfg.ranks != world_size:
        raise ValueError(
            f"configured ranks ({cfg.ranks}) must equal torchrun WORLD_SIZE ({world_size})"
        )
    if _is_real_cell(cfg):
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=cfg.backend, init_method="env://")
    ctx = _make_distributed_context(cfg, rank)
    try:
        base_losses, base_ms, sdk_losses, sdk_ms = _run_rank_sequence(cfg, ctx)
        if rank != 0:
            return None
        return _build_result_bundle(cfg, base_losses, base_ms, sdk_losses, sdk_ms)
    finally:
        _destroy_context(ctx)
        dist.destroy_process_group()


def run_cell(cfg: BenchConfig) -> Optional[dict[str, Any]]:
    """Run the paired benchmark and return the rank-0 result bundle.

    ``selfspawn`` keeps the original local ``mp.Process`` + ``FileStore`` path
    as the default. ``torchrun`` assumes this process is already one rank in an
    env:// launch and returns a bundle only on rank 0.
    """
    launch = _resolve_launch(cfg)
    cfg = replace(cfg, launch=launch)
    _validate_common_config(cfg)
    _validate_real_cell_requirements(cfg, launch)
    if launch == "torchrun":
        return _run_cell_torchrun(cfg)
    return _run_cell_selfspawn(cfg)


def write_bundle(bundle: dict[str, Any], results_root: Path = RESULTS_ROOT) -> Path:
    out_dir = results_root / str(bundle["cell"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n")
    return out_path


def _print_summary(bundle: dict[str, Any]) -> None:
    bec = bundle["bit_exact_loss_equivalence"]
    up = bundle["metrics"]["uplift"]
    base = bundle["metrics"]["baseline"]
    sdk = bundle["metrics"]["sdk"]
    status = "PASS" if bec["passed"] else "FAIL"
    print("=" * 64)
    print(f"cell: {bundle['cell']}  ({bundle['reproducible']})")
    print(f"hardware: {bundle['hardware']['label']}")
    print(
        f"bit-exact loss equivalence (default mode): {status}  "
        f"(max |loss diff| = {bec['max_abs_loss_diff']:.3e} over "
        f"{bec['n_steps_compared']} steps)"
    )
    print(
        f"baseline tokens/s: {base['tokens_per_second']:.1f}  "
        f"(mean step {base['mean_step_time_ms']:.2f} ms)"
    )
    print(
        f"sdk      tokens/s: {sdk['tokens_per_second']:.1f}  "
        f"(mean step {sdk['mean_step_time_ms']:.2f} ms)"
    )
    print(
        f"uplift: {up['tokens_per_second_pct']:+.2f}%  "
        f"(95% CI [{up['ci95_low_pct']:+.2f}%, {up['ci95_high_pct']:+.2f}%], "
        f"n={up['n_paired_steps']} paired steps, "
        f"{up['n_bootstrap_resamples']} resamples)"
    )
    print("=" * 64)


# Pre-baked cells. The cheap cell is the only one runnable at $0; the
# others document the real config shape the harness is READY for (running
# them requires provisioning a real GPU cluster, out of scope for this
# harness, which never provisions compute).
CELLS: dict[str, BenchConfig] = {
    "cpu_gloo_2rank_mlp": BenchConfig(
        cell="cpu_gloo_2rank_mlp",
        backend="gloo",
        ranks=2,
        steps=160,
        warmup_steps=20,
        sync_period_steps=10,
        apply_lag_steps=6,
        simulated_merge_delay_ms=12,
        hardware_label="local CPU (gloo); $0 cheap reproducible cell",
    ),
    "real_8xv100_2node": BenchConfig(
        cell="real_8xv100_2node",
        launch="torchrun",
        backend="nccl",
        ranks=16,
        steps=500,
        warmup_steps=50,
        sync_period_steps=128,
        apply_lag_steps=8,
        simulated_merge_delay_ms=0,
        batch=1,
        sequence_length=256,
        lr=1e-5,
        model_id="HuggingFaceTB/SmolLM-135M",
        tokenizer_id="HuggingFaceTB/SmolLM-135M",
        dataset_id=None,
        hardware_label=(
            "real GPU cluster placeholder; replace with provider, node count, "
            "GPU type, fabric, and pinned CUDA/NCCL/PyTorch versions"
        ),
    ),
}


def _parse_args(argv: Optional[list[str]] = None) -> BenchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        default="cpu_gloo_2rank_mlp",
        help="named pre-baked cell (overridable by the flags below)",
    )
    parser.add_argument(
        "--launch",
        choices=("selfspawn", "torchrun", "auto"),
        default=None,
        help="selfspawn (default cheap cell), torchrun/env://, or auto",
    )
    parser.add_argument("--backend", default=None, help="gloo (CPU) | nccl (GPU)")
    parser.add_argument("--ranks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--sync-period-steps", type=int, default=None)
    parser.add_argument("--apply-lag-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grace-window-ms", type=int, default=None)
    parser.add_argument("--simulated-merge-delay-ms", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--hardware-label", default=None)
    parser.add_argument(
        "--write",
        dest="write",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="write the result bundle to benchmarks/results/<cell>/ "
        "(default: --write; pass --no-write to only print the summary)",
    )
    args = parser.parse_args(argv)

    cfg = replace(CELLS[args.cell]) if args.cell in CELLS else BenchConfig(cell=args.cell)
    # CLI overrides (args-only scaling from cheap cell to real config).
    overrides = {
        "launch": args.launch,
        "backend": args.backend,
        "ranks": args.ranks,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "sync_period_steps": args.sync_period_steps,
        "apply_lag_steps": args.apply_lag_steps,
        "seed": args.seed,
        "batch": args.batch,
        "sequence_length": args.sequence_length,
        "lr": args.lr,
        "grace_window_ms": args.grace_window_ms,
        "simulated_merge_delay_ms": args.simulated_merge_delay_ms,
        "bootstrap_resamples": args.bootstrap_resamples,
        "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id,
        "dataset_id": args.dataset_id,
        "hardware_label": args.hardware_label,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    setattr(cfg, "_write", args.write)
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    cfg = _parse_args(argv)
    write = getattr(cfg, "_write", True)
    if cfg.backend != "gloo":
        print(
            "NOTE: non-gloo backends require a real GPU cluster; this harness "
            "is ready for them but the cheap-cell path is CPU/gloo only.",
            file=sys.stderr,
        )
    bundle = run_cell(cfg)
    if bundle is None:
        return 0
    _print_summary(bundle)
    if write:
        path = write_bundle(bundle)
        print(f"wrote bundle: {path.relative_to(_REPO_ROOT)}")
    return 0 if bundle["bit_exact_loss_equivalence"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
