"""Sideband control-plane metadata channel.

Carries step-id, vector-clock progress, queue-depth, and rank health-bit
metadata between racks over a low-bandwidth TCP channel separate from the
NCCL data plane.

This is the engineering control-plane / data-plane split common to most
distributed systems and shown in Decoupled DiLoCo Section 3.2. The
implementation here is a fresh write; it does NOT import from
`tsugiai-kpool-sdk` so the two SDKs share zero code per the launch
doctrine.

Patent-independence note: this is a generic TCP heartbeat / progress
channel and is unrelated to TsugiCinema's Infinity provisional claims.
Infinity's sideband claim is at LoRA-adapter granularity with a specific
buffer-fill payload; this SDK's payload is rack-level progress metadata
only.

Differences from tsugi_kpool.sideband.Sideband:
    - Payload schema includes `rank_id`, `step_id`, `vector_clock_us`,
      `queue_depth`, `health_bit`, `hostname`. The kpool sideband payload
      is `sender_id`, `ts_monotonic_ns`, `buffer_fill`.
    - Drift computed on `vector_clock_us` (microsecond-resolution),
      not wall-clock monotonic ns.
    - No buffer-fill semantics; this sideband is not coupled to a
      gradient aggregator. The aggregator concept does not exist here.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ProgressHeartbeat:
    """One sideband progress message.

    Attributes:
        rank_id: stable rank identifier (e.g., "rack-0/rank-0").
        hostname: sender hostname (for the rank_to_hostname map).
        step_id: monotonic inner-step counter on the sender at send time.
        vector_clock_us: sender's monotonic clock in microseconds.
        queue_depth: number of fragments the sender has waiting to send
            to the syncer this round; informational only.
        health_bit: True if the sender thinks it is healthy. The
            fail-slow detector still gets the final say at the syncer.
    """
    rank_id: str
    hostname: str
    step_id: int
    vector_clock_us: int
    queue_depth: int
    health_bit: bool


class Sideband:
    """TCP-based sideband channel for rack-level progress metadata.

    Lifecycle:
        sb = Sideband(local_rank_id, local_hostname, addr, peers,
                      heartbeat_ms, connect_timeout_s)
        await sb.start()
        ... runtime calls sb.set_local_state(step_id, queue_depth, health) ...
        ... runtime reads sb.snapshot() for peer state ...
        await sb.stop()

    The sideband is bandwidth-budgeted under 100KB/s/peer; payloads are
    sub-200 bytes JSON lines and the heartbeat cadence is configurable
    (default 100 ms).
    """

    def __init__(
        self,
        rank_id: str,
        hostname: str,
        addr: str,
        peers: tuple[str, ...],
        heartbeat_ms: int,
        connect_timeout_s: float,
    ) -> None:
        self.rank_id = rank_id
        self.hostname = hostname
        self.addr = addr
        self.peers = peers
        self.heartbeat_ms = heartbeat_ms
        self.connect_timeout_s = connect_timeout_s

        self._local_step_id = 0
        self._local_queue_depth = 0
        self._local_health = True

        self._peer_state: dict[str, ProgressHeartbeat] = {}
        self._peer_last_recv_us: dict[str, int] = {}

        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._server: Optional[asyncio.AbstractServer] = None

    # -------- public mutators --------
    def set_local_state(
        self,
        step_id: int,
        queue_depth: int,
        health_bit: bool,
    ) -> None:
        """Update the local progress state the next heartbeat will send."""
        self._local_step_id = step_id
        self._local_queue_depth = queue_depth
        self._local_health = health_bit

    # -------- public observers --------
    def peer_snapshot(self) -> dict[str, ProgressHeartbeat]:
        """Defensive-copied snapshot of the latest progress per peer."""
        return dict(self._peer_state)

    def peer_hostname_map(self) -> dict[str, str]:
        """Map of rank_id -> hostname collected from peers. Used by
        topology classification."""
        return {rid: hb.hostname for rid, hb in self._peer_state.items()}

    def peer_drift_us(self, peer_rank_id: str, now_us: Optional[int] = None) -> float:
        """Microseconds between the latest progress message received from
        a peer and the current monotonic clock. Returns +inf if no
        message has arrived from that peer yet."""
        last = self._peer_last_recv_us.get(peer_rank_id, None)
        if last is None:
            return float("inf")
        if now_us is None:
            now_us = time.monotonic_ns() // 1000
        return float(abs(now_us - last))

    # -------- lifecycle --------
    async def start(self) -> None:
        host, port = self._parse_addr(self.addr)
        # Keep a handle on the server so stop() can close the listening
        # socket. asyncio.start_server() returns a Server object; cancelling
        # serve_forever() alone does NOT close the underlying socket —
        # server.close() + server.wait_closed() are required for actual
        # port release. Without this, the listening socket lingers past
        # mend_shutdown and the next cell in the same container hits
        # EADDRINUSE on the same port (issue observed 2026-05-23 during
        # Track A retry).
        self._server = await asyncio.start_server(self._handle_inbound, host, port)
        self._running = True
        self._tasks.append(asyncio.create_task(self._server.serve_forever()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

    async def stop(self) -> None:
        self._running = False
        # Close the listening socket FIRST so the port is released
        # before any pending tasks attempt to use it.
        server = getattr(self, "_server", None)
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:
                pass
            self._server = None
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    # -------- internals --------
    async def _heartbeat_loop(self) -> None:
        while self._running:
            for peer in self.peers:
                await self._send_to_peer(peer)
            await asyncio.sleep(self.heartbeat_ms / 1000.0)

    async def _send_to_peer(self, peer: str) -> None:
        host, port = self._parse_addr(peer)
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.connect_timeout_s,
            )
        except (OSError, asyncio.TimeoutError):
            return
        msg = ProgressHeartbeat(
            rank_id=self.rank_id,
            hostname=self.hostname,
            step_id=self._local_step_id,
            vector_clock_us=time.monotonic_ns() // 1000,
            queue_depth=self._local_queue_depth,
            health_bit=self._local_health,
        )
        try:
            writer.write(json.dumps(asdict(msg)).encode() + b"\n")
            await writer.drain()
        except (OSError, ConnectionError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, ConnectionError):
                pass

    async def _handle_inbound(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        now_us = time.monotonic_ns() // 1000
        try:
            line = await reader.readline()
        except (OSError, ConnectionError):
            writer.close()
            return
        if not line:
            writer.close()
            return
        try:
            payload = json.loads(line.decode())
            hb = ProgressHeartbeat(**payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            writer.close()
            return
        self._peer_state[hb.rank_id] = hb
        self._peer_last_recv_us[hb.rank_id] = now_us
        writer.close()

    @staticmethod
    def _parse_addr(addr: str) -> tuple[str, int]:
        if not addr.startswith("tcp://"):
            raise ValueError(f"addr must start with tcp://; got {addr!r}")
        host, _, port = addr.removeprefix("tcp://").partition(":")
        return host, int(port)
