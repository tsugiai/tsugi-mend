"""Tests for the tsugi-mend command-line interface.

These tests exercise argument parsing and the torch-free `version`/`info`
paths. They do not require torch to be installed: `version` and `info` import
only the standard library plus the torch-free `MendConfig`.
"""
from __future__ import annotations

import builtins

import pytest

from tsugi_mend.__main__ import (
    _parse_peers,
    _parse_tcp_addr,
    build_parser,
    main,
)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def test_parser_no_command_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_parser_version_subcommand():
    parser = build_parser()
    args = parser.parse_args(["version"])
    assert args.command == "version"
    assert callable(args.func)


def test_parser_info_subcommand():
    parser = build_parser()
    args = parser.parse_args(["info"])
    assert args.command == "info"
    assert callable(args.func)


def test_parser_doctor_defaults():
    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.peers is None
    assert args.timeout == 2.0


def test_parser_doctor_with_peers_and_timeout():
    parser = build_parser()
    args = parser.parse_args(
        ["doctor", "--peers", "tcp://a:1,tcp://b:2", "--timeout", "0.5"]
    )
    assert args.peers == "tcp://a:1,tcp://b:2"
    assert args.timeout == 0.5


def test_parser_unknown_command_errors():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["nonsense"])


# --------------------------------------------------------------------------
# --peers parsing helper
# --------------------------------------------------------------------------
def test_parse_peers_empty():
    assert _parse_peers(None) == []
    assert _parse_peers("") == []


def test_parse_peers_strips_and_drops_blanks():
    assert _parse_peers("tcp://a:1, tcp://b:2 ,") == ["tcp://a:1", "tcp://b:2"]


# --------------------------------------------------------------------------
# tcp address parsing helper
# --------------------------------------------------------------------------
def test_parse_tcp_addr_ok():
    assert _parse_tcp_addr("tcp://host.example:51900") == ("host.example", 51900)


@pytest.mark.parametrize(
    "addr",
    [
        "udp://host:1",       # wrong scheme
        "tcp://host",         # no port
        "tcp://:51900",       # no host
        "tcp://host:notint",  # non-integer port
        "tcp://host:0",       # port out of range (low)
        "tcp://host:70000",   # port out of range (high)
    ],
)
def test_parse_tcp_addr_malformed(addr):
    with pytest.raises(ValueError):
        _parse_tcp_addr(addr)


# --------------------------------------------------------------------------
# version / info end-to-end (torch-free path)
# --------------------------------------------------------------------------
def test_version_command_prints_version(capsys):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # A non-empty, dotted version string.
    assert out
    assert "." in out


def test_info_command_torch_free(capsys, monkeypatch):
    """`info` must run even if torch import is forced to fail.

    We shadow the real import machinery so that any attempt to `import torch`
    raises ImportError, then assert `info` still succeeds and prints the
    MendConfig defaults. This guards the torch-free guarantee.
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not available (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MendConfig defaults:" in out
    # A couple of representative defaults appear in the dump.
    assert "quorum_min_learners" in out
    assert "sideband_addr" in out


def test_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


# --------------------------------------------------------------------------
# doctor: peer reachability hard-failure semantics (no torch needed)
# --------------------------------------------------------------------------
def test_doctor_unreachable_peer_is_hard_failure(capsys):
    # Port 1 on localhost is almost certainly closed -> connect fails fast.
    rc = main(["doctor", "--peers", "tcp://127.0.0.1:1", "--timeout", "0.2"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_doctor_malformed_peer_is_hard_failure(capsys):
    rc = main(["doctor", "--peers", "not-a-tcp-addr", "--timeout", "0.2"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_doctor_reachable_peer_passes(capsys):
    """Spin up a throwaway listener and confirm doctor reports it reachable."""
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        rc = main(["doctor", "--peers", f"tcp://127.0.0.1:{port}", "--timeout", "1.0"])
        out = capsys.readouterr().out
        assert "PASS" in out
        # doctor's overall RESULT depends on torch presence (WARNs are not
        # hard failures); the peer line itself must be reachable, so the
        # exit code is 0 when the only requested peer is reachable.
        assert rc == 0
    finally:
        srv.close()
