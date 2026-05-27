"""Sideband heartbeat tests (two-localhost-process)."""
from __future__ import annotations

import asyncio

import pytest

from tsugi_mend.sideband import Sideband


def _free_addr_pair() -> tuple[str, str]:
    """Return two distinct localhost addresses with random free ports."""
    import socket
    socks = []
    try:
        addrs: list[str] = []
        for _ in range(2):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            addrs.append(f"tcp://127.0.0.1:{port}")
            socks.append(s)
        return addrs[0], addrs[1]
    finally:
        for s in socks:
            s.close()


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
