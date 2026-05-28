# Candidate Validation Notes

This branch adds capability for the paired benchmark harness; it does not add a
new real-hardware result bundle.

## CPU-validated path

- Default self-spawn `cpu_gloo_2rank_mlp` remains the cheap $0 cell.
- Torchrun/env:// `cpu_gloo_2rank_mlp` is the new CPU-validatable cross-node
  launch path and should produce a bit-exact PASS with `max |loss diff| = 0.0`.
- Fragment object gather runs on a dedicated gloo process group, including when
  the data-plane backend is configured as `nccl`.

## GPU-deferred path

- `real_8xv100_2node` is structurally implemented for a real CUDA cluster:
  torchrun/env:// launch, `nccl` data plane, Hugging Face causal LM workload,
  and per-node FSDP sharding.
- The real cell uses `HuggingFaceTB/SmolLM-135M` and a deterministic synthetic
  token stream by default, so no dataset download is required.
- The GPU/FSDP path is validated by the maintainer on real hardware. This
  branch does not provision a cluster, does not run a GPU job, and does not
  claim a measured real-cell uplift.
