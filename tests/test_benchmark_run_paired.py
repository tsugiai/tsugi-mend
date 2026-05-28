"""CPU-only tests for benchmark driver branching and real-cell metadata."""
from __future__ import annotations

import os
import sys

import pytest

# Make the repo root importable so `import benchmarks.run_paired` resolves when
# the package is not installed (mirrors tests/conftest.py's src/ insertion).
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmarks.run_paired import (  # noqa: E402
    BenchConfig,
    CELLS,
    _build_result_bundle,
    run_cell,
)


def test_real_cell_requires_cuda_before_loading_optional_stack(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("benchmarks.run_paired.torch.cuda.is_available", lambda: False)
    cfg = BenchConfig(
        cell="real_cpu_guard",
        launch="torchrun",
        backend="nccl",
        ranks=2,
        steps=4,
        warmup_steps=1,
        model_id="HuggingFaceTB/SmolLM-135M",
        tokenizer_id="HuggingFaceTB/SmolLM-135M",
    )
    with pytest.raises(RuntimeError, match="requires CUDA.*real-cell"):
        run_cell(cfg)


def test_real_cell_bundle_metadata(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    cfg = BenchConfig(
        cell="real_metadata",
        launch="torchrun",
        backend="nccl",
        ranks=16,
        steps=4,
        warmup_steps=0,
        bootstrap_resamples=200,
        batch=2,
        sequence_length=64,
        lr=1e-5,
        model_id="HuggingFaceTB/SmolLM-135M",
        tokenizer_id="HuggingFaceTB/SmolLM-135M",
        hardware_label="real GPU cluster placeholder",
    )
    bundle = _build_result_bundle(
        cfg,
        base_losses=[1.0, 0.9, 0.8, 0.7],
        base_ms=[10.0, 10.0, 10.0, 10.0],
        sdk_losses=[1.0, 0.9, 0.8, 0.7],
        sdk_ms=[8.0, 8.0, 8.0, 8.0],
    )
    assert bundle["reproducible"] == "real-hardware (requires a GPU cluster)"
    assert bundle["workload"]["kind"] == "huggingface"
    assert bundle["workload"]["model_id"] == "HuggingFaceTB/SmolLM-135M"
    assert bundle["workload"]["tokens_per_step"] == 128
    assert bundle["workload"]["optimizer"] == "AdamW"
    assert bundle["sdk_config"]["quorum_min_learners"] == 2


def test_prebaked_real_cell_is_gpu_deferred():
    cfg = CELLS["real_8xv100_2node"]
    assert cfg.launch == "torchrun"
    assert cfg.backend == "nccl"
    assert cfg.model_id == "HuggingFaceTB/SmolLM-135M"
    assert cfg.simulated_merge_delay_ms == 0
    assert cfg.steps == 500
    assert cfg.warmup_steps == 50
