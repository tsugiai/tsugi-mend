"""Command-line interface for tsugi-mend."""
from __future__ import annotations

import argparse
import socket
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from typing import Literal, TextIO, cast
from urllib.parse import urlparse

from tsugi_mend import __version__
from tsugi_mend.config import MendConfig


Status = Literal["PASS", "WARN", "FAIL"]
CommandHandler = Callable[[argparse.Namespace, TextIO], int]


@dataclass(frozen=True)
class CheckResult:
    status: Status
    name: str
    detail: str


@dataclass(frozen=True)
class PeerAddress:
    original: str
    host: str
    port: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsugi-mend",
        description="Inspect tsugi-mend version, defaults, and environment readiness.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser(
        "version",
        help="Print the package version.",
    )
    version_parser.set_defaults(handler=_cmd_version)

    info_parser = subparsers.add_parser(
        "info",
        help="Print a short description and MendConfig defaults.",
    )
    info_parser.set_defaults(handler=_cmd_info)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run a local environment preflight.",
    )
    doctor_parser.add_argument(
        "--peers",
        default="",
        metavar="tcp://host:port,...",
        help="Optional comma-separated sideband peers to test for TCP reachability.",
    )
    doctor_parser.set_defaults(handler=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler, getattr(args, "handler"))
    return handler(args, sys.stdout)


def _cmd_version(_args: argparse.Namespace, stdout: TextIO) -> int:
    stdout.write(f"{__version__}\n")
    return 0


def _cmd_info(_args: argparse.Namespace, stdout: TextIO) -> int:
    config = MendConfig()
    stdout.write(f"tsugi-mend {__version__}\n")
    stdout.write("Cross-rack distributed-training reducer with conservative defaults.\n")
    stdout.write("MendConfig defaults:\n")
    for config_field in fields(config):
        value = getattr(config, config_field.name)
        stdout.write(f"  {config_field.name}: {value!r}\n")
    return 0


def _cmd_doctor(args: argparse.Namespace, stdout: TextIO) -> int:
    peer_values = _split_peer_arg(str(getattr(args, "peers", "")))
    results = _collect_torch_checks()
    results.extend(_collect_peer_checks(peer_values))

    for result in results:
        stdout.write(f"{result.status} {result.name}: {result.detail}\n")

    return 1 if any(result.status == "FAIL" for result in results) else 0


def _collect_torch_checks() -> list[CheckResult]:
    try:
        import torch
    except Exception as exc:
        return [
            CheckResult(
                "FAIL",
                "torch import",
                f"not importable ({type(exc).__name__}: {exc})",
            )
        ]

    results = [
        CheckResult(
            "PASS",
            "torch import",
            f"version {getattr(torch, '__version__', 'unknown')}",
        )
    ]

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        results.append(
            CheckResult(
                "WARN",
                "CUDA availability",
                f"could not query ({type(exc).__name__}: {exc})",
            )
        )
    else:
        results.append(
            CheckResult(
                "PASS" if cuda_available else "WARN",
                "CUDA availability",
                "available" if cuda_available else "not available",
            )
        )

    try:
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        results.append(
            CheckResult(
                "WARN",
                "visible CUDA devices",
                f"could not query ({type(exc).__name__}: {exc})",
            )
        )
    else:
        results.append(
            CheckResult(
                "PASS" if device_count > 0 else "WARN",
                "visible CUDA devices",
                str(device_count),
            )
        )

    try:
        import torch.distributed as dist

        nccl_available = bool(dist.is_nccl_available())
    except Exception as exc:
        results.append(
            CheckResult(
                "WARN",
                "NCCL availability",
                f"could not query ({type(exc).__name__}: {exc})",
            )
        )
    else:
        results.append(
            CheckResult(
                "PASS" if nccl_available else "WARN",
                "NCCL availability",
                "available" if nccl_available else "not available",
            )
        )

    return results


def _collect_peer_checks(peer_values: Sequence[str]) -> list[CheckResult]:
    if not peer_values:
        return [
            CheckResult(
                "WARN",
                "sideband peers",
                "no peers supplied; skipping TCP reachability",
            )
        ]

    config = MendConfig()
    results: list[CheckResult] = []
    for peer_value in peer_values:
        try:
            peer = _parse_peer(peer_value)
        except ValueError as exc:
            results.append(CheckResult("FAIL", f"peer {peer_value}", str(exc)))
            continue

        results.append(_check_peer_reachable(peer, config.sideband_connect_timeout_s))

    return results


def _check_peer_reachable(peer: PeerAddress, timeout_s: float) -> CheckResult:
    try:
        with socket.create_connection((peer.host, peer.port), timeout=timeout_s):
            pass
    except OSError as exc:
        return CheckResult(
            "FAIL",
            f"peer {peer.original}",
            f"TCP connect failed ({type(exc).__name__}: {exc})",
        )

    return CheckResult("PASS", f"peer {peer.original}", "TCP reachable")


def _split_peer_arg(peers: str) -> list[str]:
    if not peers.strip():
        return []
    return [peer.strip() for peer in peers.split(",") if peer.strip()]


def _parse_peer(peer: str) -> PeerAddress:
    parsed = urlparse(peer)
    if parsed.scheme != "tcp":
        raise ValueError("expected tcp://host:port")
    if parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("expected tcp://host:port without auth, path, query, or fragment")
    if parsed.path not in ("", "/"):
        raise ValueError("expected tcp://host:port without a path")
    if not parsed.hostname:
        raise ValueError("missing host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid port ({exc})") from exc
    if port is None:
        raise ValueError("missing port")
    if port < 1:
        raise ValueError("port must be >= 1")
    return PeerAddress(original=peer, host=parsed.hostname, port=port)


if __name__ == "__main__":
    raise SystemExit(main())
