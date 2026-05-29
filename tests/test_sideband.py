"""Sideband heartbeat tests (two-localhost-process)."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tsugi_mend.sideband import ProgressHeartbeat, Sideband


def _free_addrs(count: int) -> tuple[str, ...]:
    """Return distinct localhost addresses with random free ports."""
    import socket

    socks = []
    try:
        addrs: list[str] = []
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            addrs.append(f"tcp://127.0.0.1:{port}")
            socks.append(s)
        return tuple(addrs)
    finally:
        for s in socks:
            s.close()


def _free_addr_pair() -> tuple[str, str]:
    """Return two distinct localhost addresses with random free ports."""
    addr_a, addr_b = _free_addrs(2)
    return addr_a, addr_b


def _free_addr() -> str:
    """Return one localhost address with a random free port."""
    return _free_addrs(1)[0]


def _heartbeat_payload(rank_id: str = "rack-peer") -> dict[str, object]:
    return {
        "rank_id": rank_id,
        "hostname": f"host-{rank_id}",
        "step_id": 7,
        "vector_clock_us": 123_456,
        "queue_depth": 2,
        "health_bit": True,
    }


def _heartbeat_line(rank_id: str = "rack-peer") -> bytes:
    return json.dumps(_heartbeat_payload(rank_id), separators=(",", ":")).encode() + b"\n"


def _tls_cert_pair(tmp_path: Path, name: str) -> tuple[str, str]:
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is required for local TLS certificate tests")

    cert = tmp_path / f"{name}.crt"
    key = tmp_path / f"{name}.key"
    config = tmp_path / f"{name}.cnf"
    config.write_text(
        "\n".join(
            [
                "[req]",
                "distinguished_name = req_distinguished_name",
                "x509_extensions = v3_req",
                "prompt = no",
                "[req_distinguished_name]",
                "CN = localhost",
                "[v3_req]",
                "subjectAltName = @alt_names",
                "[alt_names]",
                "DNS.1 = localhost",
                "IP.1 = 127.0.0.1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-config",
            str(config),
            "-extensions",
            "v3_req",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return str(cert), str(key)


async def _send_raw(addr: str, line: bytes) -> None:
    host, port = Sideband._parse_addr(addr)
    _, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(line)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass


@pytest.mark.asyncio
async def test_two_localhost_heartbeat_exchange():
    """Two Sideband instances exchange heartbeats; each should see the other."""
    addr_a, addr_b = _free_addr_pair()
    sb_a = Sideband(
        rank_id="rack-0",
        hostname="host-a",
        addr=addr_a,
        peers=(addr_b,),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    sb_b = Sideband(
        rank_id="rack-1",
        hostname="host-b",
        addr=addr_b,
        peers=(addr_a,),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    sb_a.set_local_state(step_id=1, queue_depth=0, health_bit=True)
    sb_b.set_local_state(step_id=2, queue_depth=0, health_bit=True)
    await sb_a.start()
    await sb_b.start()
    try:
        # Give the heartbeat loop a few cycles to exchange.
        await asyncio.sleep(0.5)
        snap_a = sb_a.peer_snapshot()
        snap_b = sb_b.peer_snapshot()
        assert "rack-1" in snap_a, f"sb_a should see rack-1; saw {list(snap_a.keys())}"
        assert "rack-0" in snap_b, f"sb_b should see rack-0; saw {list(snap_b.keys())}"
        assert snap_a["rack-1"].hostname == "host-b"
        assert snap_b["rack-0"].hostname == "host-a"
    finally:
        await sb_a.stop()
        await sb_b.stop()


@pytest.mark.asyncio
async def test_sideband_drift_us_is_finite_after_heartbeat():
    addr_a, addr_b = _free_addr_pair()
    sb_a = Sideband(
        rank_id="rack-0",
        hostname="ha",
        addr=addr_a,
        peers=(addr_b,),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    sb_b = Sideband(
        rank_id="rack-1",
        hostname="hb",
        addr=addr_b,
        peers=(addr_a,),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    await sb_a.start()
    await sb_b.start()
    try:
        await asyncio.sleep(0.3)
        drift = sb_a.peer_drift_us("rack-1")
        # Drift must be finite once we've received at least one heartbeat.
        assert drift != float("inf"), "drift should be finite after heartbeat exchange"
        # Localhost: under 250 ms drift comfortably.
        assert drift < 250_000, f"localhost drift too high: {drift} us"
    finally:
        await sb_a.stop()
        await sb_b.stop()


@pytest.mark.asyncio
async def test_sideband_peer_drift_inf_before_any_heartbeat():
    addr_a, addr_b = _free_addr_pair()
    sb_a = Sideband(
        rank_id="rack-0",
        hostname="ha",
        addr=addr_a,
        peers=(addr_b,),  # peer never starts; we never receive
        heartbeat_ms=50,
        connect_timeout_s=0.1,
    )
    await sb_a.start()
    try:
        await asyncio.sleep(0.2)
        assert sb_a.peer_drift_us("rack-1") == float("inf")
        assert sb_a.peer_snapshot() == {}
    finally:
        await sb_a.stop()


def test_sideband_addr_parse_validation():
    """Sideband.__init__ does not raise on bad URI; we get the validation
    via _parse_addr only when start() is called. Direct check of the
    static parser."""
    with pytest.raises(ValueError, match="tcp://"):
        Sideband._parse_addr("udp://1.2.3.4:5")


def test_default_no_knob_wire_format_unchanged():
    sb = Sideband(
        rank_id="rack-0",
        hostname="host-a",
        addr="tcp://127.0.0.1:0",
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    msg = ProgressHeartbeat(
        rank_id="rack-0",
        hostname="host-a",
        step_id=1,
        vector_clock_us=2,
        queue_depth=3,
        health_bit=True,
    )

    assert sb._encode_heartbeat(msg) == (
        b'{"rank_id": "rack-0", "hostname": "host-a", "step_id": 1, '
        b'"vector_clock_us": 2, "queue_depth": 3, "health_bit": true}\n'
    )


@pytest.mark.asyncio
async def test_sideband_rejects_malformed_payloads():
    addr = _free_addr()
    sb = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr,
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    await sb.start()
    try:
        await _send_raw(addr, b"not-json\n")
        await _send_raw(addr, json.dumps({"rank_id": "missing-fields"}).encode() + b"\n")
        wrong_type = _heartbeat_payload("wrong-type")
        wrong_type["step_id"] = "7"
        await _send_raw(addr, json.dumps(wrong_type).encode() + b"\n")
        bool_counter = _heartbeat_payload("bool-counter")
        bool_counter["queue_depth"] = False
        await _send_raw(addr, json.dumps(bool_counter).encode() + b"\n")

        await asyncio.sleep(0.05)
        assert sb.peer_snapshot() == {}
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_sideband_rejects_oversized_line_and_keeps_server_alive():
    addr = _free_addr()
    sb = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr,
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
        max_line_bytes=256,
    )
    await sb.start()
    try:
        await _send_raw(addr, b"x" * 1024 + b"\n")
        await _send_raw(addr, _heartbeat_line("valid-peer"))

        await asyncio.sleep(0.05)
        snap = sb.peer_snapshot()
        assert list(snap) == ["valid-peer"]
        assert snap["valid-peer"].step_id == 7
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_sideband_inbound_read_timeout_closes_slow_sender():
    addr = _free_addr()
    sb = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr,
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
        inbound_read_timeout_s=0.05,
    )
    await sb.start()
    host, port = Sideband._parse_addr(addr)
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(b'{"rank_id"')
        await writer.drain()

        assert await asyncio.wait_for(reader.read(), timeout=0.5) == b""

        await _send_raw(addr, _heartbeat_line("valid-peer"))
        await asyncio.sleep(0.05)
        assert list(sb.peer_snapshot()) == ["valid-peer"]
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        await sb.stop()


@pytest.mark.asyncio
async def test_sideband_caps_concurrent_inbound_handlers():
    addr = _free_addr()
    sb = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr,
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
        inbound_read_timeout_s=0.5,
        max_inbound_connections=1,
    )
    await sb.start()
    host, port = Sideband._parse_addr(addr)
    _reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(b'{"rank_id"')
        await writer.drain()
        for _ in range(20):
            if sb._active_inbound_connections == 1:
                break
            await asyncio.sleep(0.01)
        assert sb._active_inbound_connections == 1

        await _send_raw(addr, _heartbeat_line("capped-peer"))
        await asyncio.sleep(0.05)
        assert sb.peer_snapshot() == {}

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)

        await _send_raw(addr, _heartbeat_line("accepted-peer"))
        await asyncio.sleep(0.05)
        assert list(sb.peer_snapshot()) == ["accepted-peer"]
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        await sb.stop()


@pytest.mark.asyncio
async def test_sideband_hmac_accepts_matching_psk():
    addr_sender, addr_receiver = _free_addr_pair()
    sender = Sideband(
        rank_id="sender",
        hostname="host-sender",
        addr=addr_sender,
        peers=(addr_receiver,),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="shared-secret",
    )
    receiver = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr_receiver,
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="shared-secret",
    )
    await receiver.start()
    await sender.start()
    try:
        await asyncio.sleep(0.2)
        snap = receiver.peer_snapshot()
        assert "sender" in snap
        assert snap["sender"].hostname == "host-sender"
    finally:
        await sender.stop()
        await receiver.stop()


@pytest.mark.asyncio
async def test_sideband_hmac_rejects_mismatched_psk():
    addr_sender, addr_receiver = _free_addr_pair()
    sender = Sideband(
        rank_id="sender",
        hostname="host-sender",
        addr=addr_sender,
        peers=(addr_receiver,),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="wrong-secret",
    )
    receiver = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr_receiver,
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="shared-secret",
    )
    await receiver.start()
    await sender.start()
    try:
        await asyncio.sleep(0.2)
        assert receiver.peer_snapshot() == {}
    finally:
        await sender.stop()
        await receiver.stop()


def test_sideband_hmac_rejects_replayed_signed_frame():
    sender = Sideband(
        rank_id="sender",
        hostname="host-sender",
        addr="tcp://127.0.0.1:0",
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="shared-secret",
    )
    receiver = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr="tcp://127.0.0.1:0",
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        psk="shared-secret",
    )
    frame = sender._encode_heartbeat(
        ProgressHeartbeat(
            rank_id="sender",
            hostname="host-sender",
            step_id=1,
            vector_clock_us=2,
            queue_depth=0,
            health_bit=True,
        )
    )

    first = receiver._decode_heartbeat(frame)
    replay = receiver._decode_heartbeat(frame)

    assert first is not None
    assert first.rank_id == "sender"
    assert replay is None


def test_sideband_tls_requires_ca_file():
    with pytest.raises(ValueError, match="tls_ca_file"):
        Sideband(
            rank_id="rank-0",
            hostname="host-a",
            addr="tcp://127.0.0.1:0",
            peers=(),
            heartbeat_ms=50,
            connect_timeout_s=0.5,
            tls=True,
            tls_certfile="server.crt",
            tls_keyfile="server.key",
        )


@pytest.mark.asyncio
async def test_sideband_tls_rejects_untrusted_server_cert(tmp_path: Path):
    trusted_cert, trusted_key = _tls_cert_pair(tmp_path, "trusted")
    untrusted_cert, untrusted_key = _tls_cert_pair(tmp_path, "untrusted")
    addr_receiver = _free_addr()
    receiver = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr_receiver,
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        tls=True,
        tls_certfile=untrusted_cert,
        tls_keyfile=untrusted_key,
        tls_ca_file=untrusted_cert,
    )
    sender = Sideband(
        rank_id="sender",
        hostname="host-sender",
        addr=_free_addr(),
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        tls=True,
        tls_certfile=trusted_cert,
        tls_keyfile=trusted_key,
        tls_ca_file=trusted_cert,
    )
    await receiver.start()
    try:
        await sender._send_to_peer(addr_receiver)
        await asyncio.sleep(0.05)
        assert receiver.peer_snapshot() == {}
    finally:
        await receiver.stop()


@pytest.mark.asyncio
async def test_sideband_tls_accepts_trusted_server_cert(tmp_path: Path):
    cert, key = _tls_cert_pair(tmp_path, "trusted")
    addr_receiver = _free_addr()
    receiver = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr_receiver,
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        tls=True,
        tls_certfile=cert,
        tls_keyfile=key,
        tls_ca_file=cert,
    )
    sender = Sideband(
        rank_id="sender",
        hostname="host-sender",
        addr=_free_addr(),
        peers=(),
        heartbeat_ms=20,
        connect_timeout_s=0.5,
        tls=True,
        tls_certfile=cert,
        tls_keyfile=key,
        tls_ca_file=cert,
    )
    await receiver.start()
    try:
        await sender._send_to_peer(addr_receiver)
        await asyncio.sleep(0.05)
        snap = receiver.peer_snapshot()
        assert list(snap) == ["sender"]
        assert snap["sender"].hostname == "host-sender"
    finally:
        await receiver.stop()


@pytest.mark.asyncio
async def test_sideband_peer_allowlist_drops_unknown_rank():
    addr = _free_addr()
    sb = Sideband(
        rank_id="receiver",
        hostname="host-receiver",
        addr=addr,
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
        peer_allowlist=("expected-peer",),
    )
    await sb.start()
    try:
        await _send_raw(addr, _heartbeat_line("unknown-peer"))
        await asyncio.sleep(0.05)
        assert sb.peer_snapshot() == {}

        await _send_raw(addr, _heartbeat_line("expected-peer"))
        await asyncio.sleep(0.05)
        snap = sb.peer_snapshot()
        assert list(snap) == ["expected-peer"]
    finally:
        await sb.stop()


@pytest.mark.asyncio
async def test_sideband_warns_for_each_non_loopback_bind_without_auth():
    first = Sideband(
        rank_id="rank-0",
        hostname="host-a",
        addr="tcp://0.0.0.0:0",
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    second = Sideband(
        rank_id="rank-1",
        hostname="host-b",
        addr="tcp://0.0.0.0:0",
        peers=(),
        heartbeat_ms=50,
        connect_timeout_s=0.5,
    )
    try:
        with pytest.warns(RuntimeWarning, match="non-loopback"):
            await first.start()
        await first.stop()

        with pytest.warns(RuntimeWarning, match="non-loopback"):
            await second.start()
        await second.stop()
    finally:
        await first.stop()
        await second.stop()
