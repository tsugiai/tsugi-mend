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
import hashlib
import hmac
import ipaddress
import json
import ssl
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Optional, TypeGuard, cast


DEFAULT_SIDEBAND_MAX_LINE_BYTES = 4096
DEFAULT_SIDEBAND_INBOUND_READ_TIMEOUT_S = 1.0
DEFAULT_SIDEBAND_MAX_INBOUND_CONNECTIONS = 64
DEFAULT_SIDEBAND_REPLAY_CACHE_PEERS = 1024
_MAX_HEARTBEAT_INT = 2**63 - 1
_MAX_HEARTBEAT_TEXT_BYTES = 1024
_HEARTBEAT_FIELDS = frozenset(
    {
        "rank_id",
        "hostname",
        "step_id",
        "vector_clock_us",
        "queue_depth",
        "health_bit",
    }
)
_HMAC_PAYLOAD_FIELD = "payload"
_HMAC_NONCE_FIELD = "nonce_ns"
_HMAC_FIELD = "hmac_sha256"
_HMAC_ENVELOPE_FIELDS = frozenset({_HMAC_PAYLOAD_FIELD, _HMAC_NONCE_FIELD, _HMAC_FIELD})


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
        *,
        psk: Optional[str] = None,
        tls: bool = False,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        tls_ca_file: Optional[str] = None,
        peer_allowlist: Optional[tuple[str, ...]] = None,
        max_line_bytes: int = DEFAULT_SIDEBAND_MAX_LINE_BYTES,
        inbound_read_timeout_s: float = DEFAULT_SIDEBAND_INBOUND_READ_TIMEOUT_S,
        max_inbound_connections: int = DEFAULT_SIDEBAND_MAX_INBOUND_CONNECTIONS,
    ) -> None:
        if psk == "":
            raise ValueError("psk must be non-empty when configured")
        if max_line_bytes < 1:
            raise ValueError(f"max_line_bytes must be >= 1; got {max_line_bytes}")
        if inbound_read_timeout_s <= 0:
            raise ValueError(
                f"inbound_read_timeout_s must be > 0; got {inbound_read_timeout_s}"
            )
        if max_inbound_connections < 1:
            raise ValueError(
                f"max_inbound_connections must be >= 1; got {max_inbound_connections}"
            )
        if tls and (
            tls_certfile is None or tls_keyfile is None or tls_ca_file is None
        ):
            raise ValueError(
                "tls=True requires tls_certfile, tls_keyfile, and tls_ca_file"
            )

        self.rank_id = rank_id
        self.hostname = hostname
        self.addr = addr
        self.peers = peers
        self.heartbeat_ms = heartbeat_ms
        self.connect_timeout_s = connect_timeout_s
        self.max_line_bytes = max_line_bytes
        self.inbound_read_timeout_s = inbound_read_timeout_s
        self.max_inbound_connections = max_inbound_connections

        self._psk: Optional[bytes] = psk.encode("utf-8") if psk is not None else None
        self._tls = tls
        self._tls_certfile = tls_certfile
        self._tls_keyfile = tls_keyfile
        self._tls_ca_file = tls_ca_file
        self._peer_allowlist: Optional[frozenset[str]] = (
            frozenset(peer_allowlist) if peer_allowlist is not None else None
        )

        self._local_step_id = 0
        self._local_queue_depth = 0
        self._local_health = True

        self._peer_state: dict[str, ProgressHeartbeat] = {}
        self._peer_last_recv_us: dict[str, int] = {}
        self._last_nonce_by_rank: OrderedDict[str, int] = OrderedDict()
        self._active_inbound_connections = 0

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
        self._warn_if_insecure_non_loopback_bind(host)
        # Keep a handle on the server so stop() can close the listening
        # socket. asyncio.start_server() returns a Server object; cancelling
        # serve_forever() alone does NOT close the underlying socket —
        # server.close() + server.wait_closed() are required for actual
        # port release. Without this, the listening socket lingers past
        # mend_shutdown and the next cell in the same container hits
        # EADDRINUSE on the same port (issue observed 2026-05-23 during
        # Track A retry).
        self._server = await asyncio.start_server(
            self._handle_inbound,
            host,
            port,
            ssl=self._server_ssl_context(),
            limit=self.max_line_bytes,
        )
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
        ssl_context = self._client_ssl_context()
        try:
            if ssl_context is None:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.connect_timeout_s,
                )
            else:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(
                        host,
                        port,
                        ssl=ssl_context,
                        server_hostname=host,
                    ),
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
            writer.write(self._encode_heartbeat(msg))
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
        if self._active_inbound_connections >= self.max_inbound_connections:
            await self._close_writer(writer)
            return
        self._active_inbound_connections += 1
        try:
            await self._read_inbound_heartbeat(reader)
        finally:
            self._active_inbound_connections -= 1
            await self._close_writer(writer)

    async def _read_inbound_heartbeat(self, reader: asyncio.StreamReader) -> None:
        now_us = time.monotonic_ns() // 1000
        try:
            line = await asyncio.wait_for(
                reader.readline(),
                timeout=self.inbound_read_timeout_s,
            )
        except (OSError, ConnectionError, ValueError, asyncio.TimeoutError):
            return
        if not line:
            return
        if len(line) > self.max_line_bytes:
            return
        hb = self._decode_heartbeat(line)
        if hb is None:
            return
        self._peer_state[hb.rank_id] = hb
        self._peer_last_recv_us[hb.rank_id] = now_us

    def _encode_heartbeat(self, msg: ProgressHeartbeat) -> bytes:
        payload: dict[str, object] = asdict(msg)
        if self._psk is None:
            return json.dumps(payload).encode() + b"\n"

        body: dict[str, object] = {
            _HMAC_PAYLOAD_FIELD: payload,
            _HMAC_NONCE_FIELD: time.monotonic_ns(),
        }
        envelope: dict[str, object] = {
            **body,
            _HMAC_FIELD: self._hmac_hex(body),
        }
        return json.dumps(envelope).encode() + b"\n"

    def _decode_heartbeat(self, line: bytes) -> Optional[ProgressHeartbeat]:
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if self._psk is None:
            return self._heartbeat_from_payload(raw)

        signed_payload = self._verify_signed_payload(raw)
        if signed_payload is None:
            return None
        payload, nonce = signed_payload
        hb = self._heartbeat_from_payload(payload)
        if hb is None:
            return None
        if not self._accept_signed_nonce(hb.rank_id, nonce):
            return None
        return hb

    def _verify_signed_payload(
        self,
        raw: object,
    ) -> Optional[tuple[dict[str, object], int]]:
        if self._psk is None:
            return None
        if not isinstance(raw, dict):
            return None
        envelope = cast(dict[object, object], raw)
        if set(envelope.keys()) != _HMAC_ENVELOPE_FIELDS:
            return None

        payload = envelope[_HMAC_PAYLOAD_FIELD]
        nonce = envelope[_HMAC_NONCE_FIELD]
        received_hmac = envelope[_HMAC_FIELD]
        if not isinstance(payload, dict):
            return None
        if not self._valid_nonnegative_int(nonce):
            return None
        if not isinstance(received_hmac, str):
            return None

        body: dict[str, object] = {
            _HMAC_PAYLOAD_FIELD: payload,
            _HMAC_NONCE_FIELD: nonce,
        }
        expected_hmac = self._hmac_hex(body)
        if not hmac.compare_digest(received_hmac, expected_hmac):
            return None
        return cast(dict[str, object], payload), nonce

    def _accept_signed_nonce(self, rank_id: str, nonce: int) -> bool:
        previous = self._last_nonce_by_rank.get(rank_id)
        if previous is not None and nonce <= previous:
            return False

        self._last_nonce_by_rank[rank_id] = nonce
        self._last_nonce_by_rank.move_to_end(rank_id)
        while len(self._last_nonce_by_rank) > DEFAULT_SIDEBAND_REPLAY_CACHE_PEERS:
            self._last_nonce_by_rank.popitem(last=False)
        return True

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter) -> None:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, ConnectionError):
            pass

    def _heartbeat_from_payload(self, payload: object) -> Optional[ProgressHeartbeat]:
        if not isinstance(payload, dict):
            return None
        payload_dict = cast(dict[object, object], payload)
        if set(payload_dict.keys()) != _HEARTBEAT_FIELDS:
            return None

        rank_id = payload_dict["rank_id"]
        hostname = payload_dict["hostname"]
        step_id = payload_dict["step_id"]
        vector_clock_us = payload_dict["vector_clock_us"]
        queue_depth = payload_dict["queue_depth"]
        health_bit = payload_dict["health_bit"]
        if not self._valid_text(rank_id):
            return None
        if not self._valid_text(hostname):
            return None
        if not self._valid_nonnegative_int(step_id):
            return None
        if not self._valid_nonnegative_int(vector_clock_us):
            return None
        if not self._valid_nonnegative_int(queue_depth):
            return None
        if not isinstance(health_bit, bool):
            return None
        if self._peer_allowlist is not None and rank_id not in self._peer_allowlist:
            return None
        return ProgressHeartbeat(
            rank_id=rank_id,
            hostname=hostname,
            step_id=step_id,
            vector_clock_us=vector_clock_us,
            queue_depth=queue_depth,
            health_bit=health_bit,
        )

    def _hmac_hex(self, body: dict[str, object]) -> str:
        if self._psk is None:
            raise RuntimeError("sideband HMAC requested without psk")
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hmac.new(self._psk, canonical, hashlib.sha256).hexdigest()

    def _server_ssl_context(self) -> Optional[ssl.SSLContext]:
        if not self._tls:
            return None
        if (
            self._tls_certfile is None
            or self._tls_keyfile is None
            or self._tls_ca_file is None
        ):
            raise ValueError(
                "tls=True requires tls_certfile, tls_keyfile, and tls_ca_file"
            )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=self._tls_certfile, keyfile=self._tls_keyfile)
        return context

    def _client_ssl_context(self) -> Optional[ssl.SSLContext]:
        if not self._tls:
            return None
        if self._tls_ca_file is None:
            raise ValueError("tls=True requires tls_ca_file")
        context = ssl.create_default_context(cafile=self._tls_ca_file)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    def _warn_if_insecure_non_loopback_bind(self, host: str) -> None:
        if self._psk is not None or self._tls:
            return
        if self._is_loopback_host(host):
            return
        warnings.warn(
            "tsugi-mend sideband is binding a non-loopback address without "
            "sideband_psk authentication or TLS peer verification. This "
            "preserves 0.1.x zero-config behavior for trusted networks; "
            "secure-by-default auth is planned for 0.2.0.",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _valid_text(value: object) -> TypeGuard[str]:
        if not isinstance(value, str) or value == "":
            return False
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return len(encoded) <= _MAX_HEARTBEAT_TEXT_BYTES

    @staticmethod
    def _valid_nonnegative_int(value: object) -> TypeGuard[int]:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value <= _MAX_HEARTBEAT_INT
        )

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        normalized = host.strip("[]").lower()
        if normalized == "localhost":
            return True
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _parse_addr(addr: str) -> tuple[str, int]:
        if not addr.startswith("tcp://"):
            raise ValueError(f"addr must start with tcp://; got {addr!r}")
        host, _, port = addr.removeprefix("tcp://").partition(":")
        return host, int(port)
