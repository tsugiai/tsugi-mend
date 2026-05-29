# Multi-node getting started

This runbook takes a local `pip install` through a real two-node `torchrun`
launch. It also covers the CPU-only two-rank smoke example in
`examples/torchrun_two_rank.py`, sideband port planning, diagnostics, and a
minimal Docker path.

Use this document for first contact and operational smoke tests. For
benchmark-quality measurements, follow
[`docs/benchmark_protocol.md`](benchmark_protocol.md) instead. That protocol
defines the paired baseline versus SDK method, warmup exclusion, confidence
intervals, and hardware reporting requirements.

## 1. Install and run the local smoke test

Install the package in the environment that will run `torchrun`:

```bash
python -m pip install -e .
```

Then run the two-rank CPU example:

```bash
torchrun --standalone --nproc-per-node=2 examples/torchrun_two_rank.py
```

The example uses the `gloo` backend, so it runs on a CPU-only laptop. It starts
two local sideband endpoints on `127.0.0.1`, wraps a small model in
`DistributedDataParallel`, calls `mend_init`, trains for a few steps, and submits
tiny learner fragments to the reducer at DES-LOC sync boundaries.

Expected output includes one line per rank confirming that the sideband saw the
peer rank, plus progress lines like:

```text
[rank 0] sideband peer local-node/rank-1 observed
[rank 1] sideband peer local-node/rank-0 observed
[rank 0] step 4: outer round merged 2 learners
[rank 1] step 4: outer round merged 2 learners
```

If the default sideband ports are busy, choose a different local base port:

```bash
MEND_SIDEBAND_PORT_BASE=52900 \
  torchrun --standalone --nproc-per-node=2 examples/torchrun_two_rank.py
```

On some macOS hosts, Python may resolve the local FQDN to an IPv6 reverse-DNS
name and `torchrun --standalone` can stall before launching worker code. In that
case, pin the local address explicitly:

```bash
torchrun --standalone --local-addr=127.0.0.1 \
  --nproc-per-node=2 examples/torchrun_two_rank.py
```

The paired benchmark harness has a matching CPU/gloo torchrun path. This runs
the same cheap cell as the default self-spawn benchmark, but validates the
env:// launch mode used by real multi-node jobs:

```bash
torchrun --standalone --local-addr=127.0.0.1 --nproc-per-node=2 \
  benchmarks/run_paired.py --launch torchrun --cell cpu_gloo_2rank_mlp --no-write
```

Expected output includes a bit-exact PASS with `max |loss diff| = 0.000e+00`.
Only rank 0 prints the benchmark summary or writes a result bundle.

Diagnostics for the smoke test are written under
`./results/torchrun_two_rank/rank*/`.

## 2. Required ordering in your training script

Initialize regular PyTorch distributed first, then build and wrap the model, and
only then call `mend_init`.

```python
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tsugi_mend import MendConfig, mend_init, mend_shutdown

process_group_timeout_s = float(os.environ.get("MEND_PROCESS_GROUP_TIMEOUT_S", "180"))
dist.init_process_group(
    backend="nccl",
    timeout=timedelta(seconds=process_group_timeout_s),
)

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

model = build_model().to(local_rank)
model = FSDP(model)  # Or your tensor-parallel or pipeline-parallel wrapper.

sideband_base = int(os.environ.get("MEND_SIDEBAND_PORT_BASE", "51900"))
sideband_port = sideband_base + local_rank
peer_hosts = [
    host
    for host in os.environ["MEND_SIDEBAND_PEERS"].split(",")
    if host
]

config = MendConfig(
    sideband_addr=f"tcp://0.0.0.0:{sideband_port}",
    sideband_peers=tuple(
        f"tcp://{host}:{sideband_port}" for host in peer_hosts
    ),
    diagnostics_dir=f"./results/mend_diag/rank{global_rank}",
)

mend_init(model, config, rank_id=f"node-{os.environ['NODE_RANK']}/rank-{global_rank}")

try:
    train(model)
finally:
    mend_shutdown(model)
    dist.destroy_process_group()
```

Important: call `mend_init` after FSDP, tensor parallelism, pipeline
parallelism, or DDP wrapping. The runtime attaches state and hooks to the module
object it receives. If you call it before wrapping, the wrapper may hide the
instrumented module from the training path.

The sideband port formula above opens one sideband listener per local process.
For an eight-GPU node with `MEND_SIDEBAND_PORT_BASE=51900`, local ranks use
TCP ports `51900` through `51907`. Each local rank peers with the same local-rank
port on the other node.

## 3. Failure contract and timeout

The SDK does not redesign PyTorch's failure semantics. If a rank raises or exits
mid-outer-step, peer ranks may be inside a PyTorch collective such as
`all_gather_object` or `barrier`. The integrator owns the job-level policy around
that failure: fail the job, restart from checkpoint, or implement a higher-level
retry loop around the training step and `outer_step_collect`.

The process-group timeout bounds the blast radius. It converts a peer failure
from an effectively unbounded wait into a PyTorch exception after a configured
interval. The examples and benchmark default this to `180` seconds:

```bash
export MEND_PROCESS_GROUP_TIMEOUT_S=180
```

For the benchmark harness, the same setting is exposed as:

```bash
python benchmarks/run_paired.py \
  --cell cpu_gloo_2rank_mlp \
  --process-group-timeout-s 180
```

This timeout does not change happy-path numerics. It only controls how long a
rank waits for a failed peer before the backend reports an error.

## 4. Two-node torchrun recipe

Choose one node to host rendezvous. In these commands, node 0 is the rendezvous
host and both nodes run eight local processes. The concrete values below are
examples: node 0 has training IP `10.0.0.10`, node 1 has training IP
`10.0.0.11`, and both nodes use training interface `eth0`. Replace them with
addresses and an interface name reachable on your private training fabric.

Common environment on both nodes:

```bash
export NNODES=2
export NPROC_PER_NODE=8
export MASTER_ADDR=10.0.0.10
export MASTER_PORT=29500
export RDZV_BACKEND=c10d
export RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
export RDZV_ID=tsugi-mend-smoke-001
export MEND_SIDEBAND_PORT_BASE=51900
export MEND_PROCESS_GROUP_TIMEOUT_S=180
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

Node 0 only:

```bash
export NODE_RANK=0
export MEND_SIDEBAND_PEERS=10.0.0.11
```

Node 1 only:

```bash
export NODE_RANK=1
export MEND_SIDEBAND_PEERS=10.0.0.10
```

Run the same `torchrun` command on both nodes:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend="${RDZV_BACKEND}" \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --rdzv-id="${RDZV_ID}" \
  train.py
```

`MASTER_ADDR` and `MASTER_PORT` identify the rendezvous endpoint. The explicit
`RDZV_*` variables make the command stable across shells and launch managers.
Use a unique `RDZV_ID` per active job so independent jobs do not join the same
rendezvous group.

### Multi-NIC interface selection

On hosts with multiple network interfaces, make `torchrun`, NCCL, gloo, and the
Mend sideband use the same routable training fabric. First identify the private
interface on each node:

```bash
ip -br addr
```

Then set `NCCL_SOCKET_IFNAME` to that interface before launching. For example,
use `eth0` on Ethernet clusters or `ib0` on InfiniBand clusters:

```bash
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
```

If your host also has Docker, loopback, management, or public-cloud metadata
interfaces, do not rely on NCCL's auto-selection. A wrong interface commonly
shows up as rendezvous success followed by NCCL connection timeouts or peers
trying to dial unroutable addresses. Keep `MASTER_ADDR` and
`MEND_SIDEBAND_PEERS` on the same private fabric as `NCCL_SOCKET_IFNAME`.

## 5. Ports and firewall rules

Allow these inbound TCP paths between the training nodes:

| Purpose | Listener | Required access |
|---|---|---|
| Torch rendezvous | `MASTER_ADDR:MASTER_PORT`, usually node 0 port `29500` | All training nodes |
| PyTorch distributed data plane | NCCL or gloo runtime selected ports | All training nodes |
| Mend sideband | `MEND_SIDEBAND_PORT_BASE` through `base + NPROC_PER_NODE - 1` on every node | Peer training nodes only |

For the sideband, with `NPROC_PER_NODE=8` and base port `51900`, open
`51900-51907/TCP` between the two training nodes. The sideband is a low-bandwidth
control plane for progress metadata. It is currently intended for a trusted
training network only. Until opt-in authentication lands, do not expose the
sideband ports to the public Internet or to unrelated tenants. Restrict access
with a VPC, security group, firewall allow-list, or equivalent private fabric
control.

If your launch environment uses NAT or containers, make sure
`MEND_SIDEBAND_PEERS` contains addresses that peer nodes can actually dial, not
container-local loopback addresses.

## 6. Diagnostics JSONL

Set `diagnostics_dir` in `MendConfig` for every rank. The runtime writes one
append-only JSONL file per process:

```text
results/mend_diag/rank0/max_sdk_pid12345.jsonl
results/mend_diag/rank1/max_sdk_pid12346.jsonl
...
```

Each line is one JSON object with an `event` field. Useful first checks:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("results/mend_diag").glob("rank*/max_sdk_pid*.jsonl")):
    print(f"\n{path}")
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event["event"] in {
            "mend_init",
            "outer_step_begin",
            "outer_step_collect",
            "failslow_decision",
            "mend_shutdown",
        }:
            print(event)
PY
```

For a healthy smoke test, every rank should emit `mend_init` and
`mend_shutdown`. Runs that exercise the reducer through the runtime outer-step
hooks should also show `outer_step_begin` and `outer_step_collect`. Check
`learners_merged` on `outer_step_collect`; it should meet or exceed
`quorum_min_learners` for the round.

Diagnostics are operational evidence, not a benchmark by themselves. For
published or externally compared throughput numbers, use the paired-run protocol
in [`docs/benchmark_protocol.md`](benchmark_protocol.md).

## 7. Docker

Build the image from the repository root:

```bash
docker build -t tsugi-mend:local .
```

Run the local two-rank smoke test in the container:

```bash
docker run --rm --network host tsugi-mend:local \
  torchrun --standalone --nproc-per-node=2 examples/torchrun_two_rank.py
```

For multi-node Docker runs, `--network host` is the simplest option because
`torchrun`, NCCL or gloo, and the Mend sideband can all use the same ports as the
host. A minimal two-node launch looks like:

```bash
docker run --rm --gpus all --network host \
  -e NNODES=2 \
  -e NPROC_PER_NODE=8 \
  -e NODE_RANK="${NODE_RANK}" \
  -e MASTER_ADDR="10.0.0.10" \
  -e MASTER_PORT=29500 \
  -e RDZV_BACKEND=c10d \
  -e RDZV_ENDPOINT="10.0.0.10:29500" \
  -e RDZV_ID=tsugi-mend-smoke-001 \
  -e MEND_SIDEBAND_PORT_BASE=51900 \
  -e MEND_PROCESS_GROUP_TIMEOUT_S=180 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e MEND_SIDEBAND_PEERS="${MEND_SIDEBAND_PEERS}" \
  tsugi-mend:local \
  torchrun \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="${NODE_RANK}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="10.0.0.10:29500" \
    --rdzv-id=tsugi-mend-smoke-001 \
    train.py
```

For Docker Compose or another scheduler, keep the same shape: host networking or
explicit TCP mappings for the rendezvous port, PyTorch distributed ports, and
the sideband port range. Pass node-specific `NODE_RANK` and
`MEND_SIDEBAND_PEERS` through the scheduler rather than hard-coding hostnames in
the image.

## 8. Real benchmark cell

The benchmark harness also defines `real_8xv100_2node`, a GPU-deferred
Hugging Face/FSDP cell for maintainer-run hardware validation. It is not a
laptop or CI target. It requires CUDA, `nccl`, torchrun/env://, and the optional
real-cell dependencies:

```bash
python -m pip install -e ".[real-cell]"
```

The expected launch shape is the same two-node torchrun recipe above, with the
benchmark driver as the entry point:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend="${RDZV_BACKEND}" \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --rdzv-id="${RDZV_ID}" \
  benchmarks/run_paired.py \
    --launch torchrun \
    --cell real_8xv100_2node \
    --hardware-label "Provider, nodes x GPUs, fabric, pinned CUDA/NCCL/PyTorch"
```

That cell uses `HuggingFaceTB/SmolLM-135M`, per-node FSDP groups, and a
deterministic token stream by default. If `--dataset-id` is supplied, the real
path lazily loads and tokenizes a small deterministic training slice through
the optional `datasets` dependency. It sets `simulated_merge_delay_ms=0` so the
measured delay is the real cross-network synchronization cost. The harness will
fail early on a non-CUDA host instead of falling back to the cheap MLP.
