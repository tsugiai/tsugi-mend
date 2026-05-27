"""Command-line interface for tsugi-mend.

Console-script entry point (registered in pyproject.toml as
``tsugi-mend = "tsugi_mend.__main__:main"``) and also runnable as
``python -m tsugi_mend``.

Subcommands:

    version   Print the installed package version.
    info      Print the MendConfig defaults plus a one-line description.
    doctor    Run an environment preflight: torch importability + version,
              CUDA availability, NCCL availability, visible device count,
              and optional TCP reachability of sideband peers given
              ``--peers tcp://host:port,...``.

Design notes:
    - ``version`` and ``info`` are torch-free. They import only the standard
      library and ``tsugi_mend.config.MendConfig`` (which is itself torch-free),
      so they work in a lighter environment without torch installed.
    - ``doctor`` lazy-imports torch inside its handler and degrades gracefully
      when torch is absent (the torch/CUDA/NCCL checks WARN instead of crashing).
    - The CLI is read-only with respect to runtime behavior. It does not import
      or invoke the training path, so the bit-exact default-mode invariant is
      untouched.
"""
from __future__ import annotations

import argparse
import socket
from dataclasses import fields
from typing import Any, Optional, Sequence


def _get_version() -> str:
    """Resolve the package version.

    Prefer installed-distribution metadata so the reported version matches
    the wheel the user actually installed; fall back to the in-tree
    ``__version__`` when running from a source checkout that is not installed.
    """
    try:
        from importlib.metadata import version

        return version("tsugi-mend")
    except Exception:
        from tsugi_mend import __version__

        return __version__


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_get_version())
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    """Print the one-line description plus MendConfig defaults (torch-free)."""
    from tsugi_mend.config import MendConfig

    description = (
        "tsugi-mend: maximum-uplift cross-rack distributed-training reducer "
        "built on Decoupled DiLoCo + DES-LOC + async tensor parallelism + "
        "FALCON fail-slow mitigation."
    )
    print(description)
    print()
    print(f"version: {_get_version()}")
    print()
    print("MendConfig defaults:")
    defaults = MendConfig()
    width = max(len(f.name) for f in fields(defaults))
    for f in fields(defaults):
        value = getattr(defaults, f.name)
        print(f"  {f.name:<{width}}  {value!r}")
    return 0


def _parse_peers(raw: Optional[str]) -> list[str]:
    """Split a ``--peers`` value into a list of non-empty address strings."""
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_tcp_addr(addr: str) -> tuple[str, int]:
    """Parse a ``tcp://host:port`` address into (host, port).

    Raises ValueError on a malformed address so the caller can WARN cleanly.
    """
    if not addr.startswith("tcp://"):
        raise ValueError("address must start with tcp://")
    host, sep, port = addr[len("tcp://"):].partition(":")
    if not host or not sep or not port:
        raise ValueError("address must be of the form tcp://host:port")
    try:
        port_num = int(port)
    except ValueError as exc:
        raise ValueError(f"port is not an integer: {port!r}") from exc
    if not (0 < port_num < 65536):
        raise ValueError(f"port out of range: {port_num}")
    return host, port_num


def _check_tcp_reachable(host: str, port: int, timeout_s: float) -> Optional[str]:
    """Attempt a TCP connect. Return None on success, else an error string."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return None
    except OSError as exc:
        return str(exc)


def _try_import_torch() -> Optional[Any]:
    """Lazy-import torch for the doctor preflight.

    Returns the torch module, or None if torch is not importable. Typed as
    Optional[Any] so the rest of the doctor handler can probe torch attributes
    without a hard build-time dependency on torch's type stubs.
    """
    try:
        import torch

        return torch
    except Exception:
        return None


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Environment preflight. Exit non-zero only on a hard failure.

    Hard failures are reachability checks for explicitly-requested peers:
    if the user passed ``--peers`` and any peer is unreachable (or malformed),
    that is a FAIL and the command exits non-zero. Missing torch / CUDA / NCCL
    are WARNs (the package imports torch-free and many environments are
    CPU-only or single-node), not hard failures.
    """
    hard_failure = False

    print("tsugi-mend doctor")
    print(f"  package version : {_get_version()}")
    print()

    # --- torch / CUDA / NCCL (lazy import; degrade gracefully) ---
    torch_mod = _try_import_torch()
    if torch_mod is None:
        print("  [WARN] torch        : not importable")
    else:
        print(f"  [PASS] torch        : {torch_mod.__version__}")

    if torch_mod is None:
        print("  [WARN] CUDA         : skipped (torch not importable)")
        print("  [WARN] NCCL         : skipped (torch not importable)")
        print("  [WARN] devices      : skipped (torch not importable)")
    else:
        try:
            cuda_available = bool(torch_mod.cuda.is_available())
        except Exception as exc:
            cuda_available = False
            print(f"  [WARN] CUDA         : query failed ({exc})")
        else:
            if cuda_available:
                print("  [PASS] CUDA         : available")
            else:
                print("  [WARN] CUDA         : not available (CPU-only environment)")

        try:
            nccl_available = bool(torch_mod.distributed.is_nccl_available())
        except Exception as exc:
            print(f"  [WARN] NCCL         : query failed ({exc})")
        else:
            if nccl_available:
                print("  [PASS] NCCL         : available")
            else:
                print("  [WARN] NCCL         : not available")

        try:
            device_count = int(torch_mod.cuda.device_count()) if cuda_available else 0
        except Exception as exc:
            print(f"  [WARN] devices      : query failed ({exc})")
        else:
            if device_count > 0:
                print(f"  [PASS] devices      : {device_count} visible CUDA device(s)")
            else:
                print("  [WARN] devices      : 0 visible CUDA devices")

    # --- sideband peer reachability (user-supplied only) ---
    peers = _parse_peers(getattr(args, "peers", None))
    if peers:
        print()
        print(f"  sideband peers ({len(peers)}):")
        for addr in peers:
            try:
                host, port = _parse_tcp_addr(addr)
            except ValueError as exc:
                hard_failure = True
                print(f"    [FAIL] {addr} : malformed ({exc})")
                continue
            err = _check_tcp_reachable(host, port, args.timeout)
            if err is None:
                print(f"    [PASS] {addr} : reachable")
            else:
                hard_failure = True
                print(f"    [FAIL] {addr} : unreachable ({err})")

    print()
    if hard_failure:
        print("  RESULT: FAIL (one or more requested peers unreachable/malformed)")
        return 1
    print("  RESULT: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the tsugi-mend CLI."""
    parser = argparse.ArgumentParser(
        prog="tsugi-mend",
        description="tsugi-mend command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p_version = subparsers.add_parser("version", help="print the package version")
    p_version.set_defaults(func=_cmd_version)

    p_info = subparsers.add_parser(
        "info", help="print config defaults and a one-line description"
    )
    p_info.set_defaults(func=_cmd_info)

    p_doctor = subparsers.add_parser("doctor", help="run an environment preflight")
    p_doctor.add_argument(
        "--peers",
        default=None,
        help="comma-separated sideband peer addresses (tcp://host:port,...) "
        "to TCP-reachability-check",
    )
    p_doctor.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="per-peer TCP connect timeout in seconds (default: 2.0)",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
