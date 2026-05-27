# tsugi-mend 0.1.1 release-readiness audit

Verdict: NO-GO until the BLOCKER items below are resolved or explicitly accepted
by the maintainer.

## BLOCKER

- Disclosure and packaging: the repo and 0.1.1 sdist include `AGENTS.md` and
  `CLAUDE.md`, which contain local workspace paths and credential-path handling
  instructions. Evidence:
  - `tar -tzf dist/tsugi_mend-0.1.1.tar.gz` lists
    `tsugi_mend-0.1.1/AGENTS.md` and `tsugi_mend-0.1.1/CLAUDE.md`.
  - `AGENTS.md:38` contains an absolute local `MasterVision` path.
  - `AGENTS.md:49` through `AGENTS.md:51` reference local credential-vault
    handling and `GH_TOKEN`.
  - `CLAUDE.md:15` references the same local credential-vault path.
  Suggested maintainer action: remove or public-sanitize those files in the repo
  before release, and make the sdist exclude any contributor-agent private
  operating notes.

- PyPI Trusted Publisher registration is not publicly verifiable. The workflow
  identifiers and `RELEASING.md` values match `owner=tsugiai`,
  `repo=tsugi-mend`, `workflow=release.yml`, `environment=pypi`, but the actual
  PyPI project settings require maintainer access to verify. A missing or
  mismatched publisher would make the OIDC publish fail.

## SHOULD-FIX

- `CHANGELOG.md` 0.1.1 was incomplete before this PR. It listed the CLI and
  `py.typed`, but not the release workflow, sideband hardening, README honesty
  update, governance files, or multi-node runbook. This branch updates it.

- `README.md` status still said `Pre-Alpha (0.1.0)` before this PR despite
  `pyproject.toml` and `src/tsugi_mend/__init__.py` both being `0.1.1`. This
  branch updates it to `0.1.1`.

- `RELEASING.md` said the workflow also ran on any pushed `v*` tag, but
  `.github/workflows/release.yml` only has `on: release: types: [published]`.
  This branch corrects the docs and leaves the workflow untouched, as required
  by the task constraints.

## NICE-TO-HAVE

- Add a `.dockerignore`. The Dockerfile copies a constrained set of files, so
  image contents are controlled, but Docker still sends the full repo context
  unless ignored by the client.

- Consider narrowing the sdist contents to source, README, license files,
  changelog, docs, and examples. The current sdist also includes `.github/`,
  tests, and contributor metadata. Tests and docs may be acceptable in an sdist;
  the agent files are the release-blocking part.

## Positive Evidence

- Version consistency: `pyproject.toml:7`, `src/tsugi_mend/__init__.py:18`, and
  `CHANGELOG.md:8` are `0.1.1`.

- Build: `python -m build` produced
  `tsugi_mend-0.1.1.tar.gz` and `tsugi_mend-0.1.1-py3-none-any.whl`.

- Distribution metadata: `twine check dist/*` passed for both artifacts.

- Wheel contents: the wheel includes `tsugi_mend/py.typed`,
  `tsugi_mend/__main__.py`, and `entry_points.txt` containing
  `tsugi-mend = tsugi_mend.__main__:main`.

- Torch-free CLI smoke: installing the wheel with `--no-deps` in a clean Python
  3.12 venv showed `importlib.util.find_spec("torch") is None`; `import
  tsugi_mend`, `tsugi-mend version`, `tsugi-mend info`, and `tsugi-mend doctor`
  all succeeded.

- Quality gates:
  - `ruff check src tests`: passed.
  - `mypy src`: `Success: no issues found in 13 source files`.
  - `pytest -q`: `117 passed in 11.17s`.

- Examples:
  - `python examples/minimal_single_process.py`: passed.
  - `python examples/concurrent_orchestrator.py`: passed.
  - `MEND_SIDEBAND_PORT_BASE=53900 torchrun --standalone --local-addr=127.0.0.1
    --nproc-per-node=2 examples/torchrun_two_rank.py`: passed with both ranks
    observing peers and two outer rounds merging two learners.

- Release workflow review:
  - Trigger is `release: types: [published]`.
  - Top-level `permissions: {}`.
  - Build job has `contents: read`.
  - Publish job has `id-token: write` and `contents: read`, uses
    `environment: name: pypi`, downloads build artifacts, and publishes with
    `pypa/gh-action-pypi-publish@v1.14.0`.
  - No `password:`, no secrets, and no `pull_request_target` found in
    `release.yml`.

- Invariants:
  - Default compression remains off:
    `MendConfig.outer_step_compression_mode == "none"`.
  - Lossy compression paths are opt-in only.
  - Sideband security controls are opt-in for 0.1.x:
    `sideband_psk=None`, `sideband_peer_allowlist=None`, and
    `sideband_tls=False` by default.
  - `import tsugi_mend` and the CLI `version` and `info` paths remain torch-free.

- PyPI state checked publicly: PyPI currently reports latest `tsugi-mend` as
  `0.1.0`, so `0.1.1` is a new version number for upload.
