# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-27

### Added

- `tsugi-mend` console-script CLI (also runnable as `python -m tsugi_mend`)
  with three subcommands:
  - `version` — print the installed package version.
  - `info` — print the `MendConfig` defaults plus a one-line description
    (torch-free, like `MendConfig` itself).
  - `doctor` — environment preflight: torch importability + version, CUDA
    availability, NCCL availability, visible CUDA device count, and optional
    TCP reachability of sideband peers via `--peers tcp://host:port,...`.
    Prints a clear PASS/WARN per check and exits non-zero only on hard
    failures (an explicitly-requested peer being unreachable or malformed).
    torch is lazy-imported so `version`/`info` work without torch installed.
- PEP 561 `py.typed` marker so downstream type checkers treat the package as
  typed; included in the built wheel.

### Changed

- Bumped version to `0.1.1` (the `0.1.0` wheel on PyPI is frozen).

## [0.1.0] - 2026-05-27

### Added

- Initial public release of `tsugi-mend`.
