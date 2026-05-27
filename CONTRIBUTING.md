# Contributing to tsugi-mend

Thanks for contributing to `tsugi-mend`. This repository is a public,
Apache-2.0 licensed Python SDK for the cross-rack distributed-training reducer.

## Development Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Checks

Run the same local checks before opening a pull request:

```bash
ruff check src tests
mypy src
pytest -q
```

When a change affects the examples or README quickstart, also run the relevant
example command and include the result in the pull request.

## Branches and Pull Requests

Create a short topic branch from the current upstream `main` branch. Keep each
pull request focused on one logical change, and send it against
`tsugiai/tsugi-mend` `main`.

Please include:

- What changed.
- Why the change is needed.
- Test evidence for `ruff`, `mypy`, `pytest`, and any relevant examples.
- Risk or review notes, including dependency or workflow changes.

Maintainers squash-merge accepted pull requests. Write commit messages and pull
request titles that describe the user-visible change clearly.

## Project Expectations

Keep public API changes deliberate and called out in the pull request. Do not
add telemetry, network calls, or new third-party dependencies unless they are
necessary for the change and explained in the review notes.
