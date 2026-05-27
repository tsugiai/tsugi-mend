"""Rack-aware topology classification.

Identifies which ranks are co-located in the same NVLink domain
(intra-rack) and which cross an inter-rack boundary (InfiniBand /
Ethernet). The runtime uses this classification to decide which
collectives go through stock NCCL and which go through the
GraceWindowSyncer.

Detection strategy, in order of preference:

1. NCCL_TOPO_FILE: if the env var or `MendConfig.nccl_topo_file` points
   to a valid XML topology file, parse it. We look for `<system>` and
   `<gpu>` nodes and pair them by NVLink. This is the canonical source
   in production NCCL deployments.

2. Hostname grouping: ranks on the same hostname are intra-rack; ranks
   on different hostnames are cross-rack. Works for typical cloud
   layouts where one VM = one node = one rack of GPUs.

3. Fallback: all ranks treated as one rack. Logs a warning. The SDK
   still functions but the cross-rack reducer never engages, so the
   uplift collapses to async-TP only.

Patent-independence note: rack-aware topology classification is a
generic distributed-systems engineering technique and is unrelated to
TsugiCinema's patent estates.
"""
from __future__ import annotations

import logging
import os
import socket
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankInfo:
    """Per-rank topology information."""
    rank: int
    hostname: str
    rack_id: str


@dataclass
class Topology:
    """Topology classification result."""
    ranks: list[RankInfo]
    rack_to_ranks: dict[str, list[int]]
    detection_method: str  # "nccl_topo_file" | "hostname" | "fallback"

    def rack_for_rank(self, rank: int) -> Optional[str]:
        for r in self.ranks:
            if r.rank == rank:
                return r.rack_id
        return None

    def is_cross_rack(self, rank_a: int, rank_b: int) -> bool:
        a = self.rack_for_rank(rank_a)
        b = self.rack_for_rank(rank_b)
        if a is None or b is None:
            return False
        return a != b

    def n_racks(self) -> int:
        return len(self.rack_to_ranks)


def classify_from_nccl_topo(topo_xml_path: str, rank_count: int) -> Optional[Topology]:
    """Parse an NCCL topology XML and group GPUs by NVLink connectivity.

    NCCL_TOPO_FILE schema (simplified):
        <system>
            <cpu numaid="...">
                <gpu rank="N" ... />
            </cpu>
            ...
        </system>

    GPUs under the same <cpu> node are taken to share NVLink (or at
    least PCI fabric); GPUs under different <cpu> nodes are inter-rack.
    Real NCCL XML is richer; the SDK only needs the rack partition, and
    the <cpu>-grouping heuristic is enough for the Stage A target.

    Returns None if the file cannot be parsed; caller falls back.
    """
    try:
        tree = ET.parse(topo_xml_path)
    except (FileNotFoundError, ET.ParseError) as e:
        _LOG.warning("classify_from_nccl_topo: cannot parse %s: %s", topo_xml_path, e)
        return None
    root = tree.getroot()
    ranks: list[RankInfo] = []
    for cpu_idx, cpu_node in enumerate(root.findall(".//cpu")):
        rack_id = f"rack-{cpu_idx}"
        for gpu_node in cpu_node.findall("./gpu"):
            rank_attr = gpu_node.get("rank")
            if rank_attr is None:
                continue
            try:
                rank = int(rank_attr)
            except ValueError:
                continue
            ranks.append(RankInfo(rank=rank, hostname="unknown", rack_id=rack_id))
    if not ranks:
        return None
    if len(ranks) != rank_count:
        _LOG.warning(
            "classify_from_nccl_topo: file lists %d ranks; runtime reports %d",
            len(ranks), rank_count,
        )
    rack_to_ranks: dict[str, list[int]] = {}
    for r in ranks:
        rack_to_ranks.setdefault(r.rack_id, []).append(r.rank)
    return Topology(
        ranks=sorted(ranks, key=lambda r: r.rank),
        rack_to_ranks=rack_to_ranks,
        detection_method="nccl_topo_file",
    )


def classify_from_hostnames(rank_to_hostname: dict[int, str]) -> Topology:
    """Group ranks by hostname; each hostname = one rack."""
    hostname_to_rack: dict[str, str] = {}
    ranks: list[RankInfo] = []
    for rank in sorted(rank_to_hostname.keys()):
        hostname = rank_to_hostname[rank]
        if hostname not in hostname_to_rack:
            hostname_to_rack[hostname] = f"rack-{len(hostname_to_rack)}"
        ranks.append(
            RankInfo(rank=rank, hostname=hostname, rack_id=hostname_to_rack[hostname])
        )
    rack_to_ranks: dict[str, list[int]] = {}
    for r in ranks:
        rack_to_ranks.setdefault(r.rack_id, []).append(r.rank)
    return Topology(
        ranks=ranks,
        rack_to_ranks=rack_to_ranks,
        detection_method="hostname",
    )


def classify_fallback(rank_count: int) -> Topology:
    """All ranks treated as one rack. Logs a warning."""
    _LOG.warning(
        "classify_fallback: no topology source; treating all %d ranks as one rack. "
        "The cross-rack reducer will not engage; uplift collapses to async-TP only.",
        rank_count,
    )
    ranks = [
        RankInfo(rank=r, hostname="unknown", rack_id="rack-0") for r in range(rank_count)
    ]
    return Topology(
        ranks=ranks,
        rack_to_ranks={"rack-0": list(range(rank_count))},
        detection_method="fallback",
    )


def detect(
    rank_count: int,
    nccl_topo_file: Optional[str] = None,
    rank_to_hostname: Optional[dict[int, str]] = None,
) -> Topology:
    """Top-level dispatch. Tries NCCL_TOPO_FILE first, then hostname, then
    fallback. `nccl_topo_file` overrides the env var if non-None.
    `rank_to_hostname` is provided by the runtime (gathered from each
    rank via the sideband) before this is called."""
    topo_path = nccl_topo_file or os.environ.get("NCCL_TOPO_FILE", None)
    if topo_path:
        result = classify_from_nccl_topo(topo_path, rank_count)
        if result is not None:
            return result
    if rank_to_hostname:
        return classify_from_hostnames(rank_to_hostname)
    return classify_fallback(rank_count)


def local_hostname() -> str:
    """Best-effort local hostname. Used by sideband heartbeats so the
    runtime can build the rank_to_hostname map at startup."""
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"
