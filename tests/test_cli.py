"""CLI behavior tests."""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_parser_accepts_doctor_peers() -> None:
    cli = importlib.import_module("tsugi_mend.__main__")
    parser = cli.build_parser()

    args = parser.parse_args(
        ["doctor", "--peers", "tcp://127.0.0.1:51900,tcp://localhost:51901"]
    )

    assert args.command == "doctor"
    assert args.peers == "tcp://127.0.0.1:51900,tcp://localhost:51901"


def test_version_command_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    cli = importlib.import_module("tsugi_mend.__main__")

    assert cli.main(["version"]) == 0

    assert capsys.readouterr().out == "0.1.1\n"


def test_info_command_does_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sys.modules.pop("tsugi_mend.__main__", None)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError(f"info command imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cli = importlib.import_module("tsugi_mend.__main__")

    assert cli.main(["info"]) == 0

    output = capsys.readouterr().out
    assert "tsugi-mend 0.1.1" in output
    assert "MendConfig defaults:" in output
    assert "  quorum_min_learners: 4" in output
