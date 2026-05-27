# Releasing tsugi-mend

This repository publishes distributions to PyPI from a GitHub Release. The
release workflow uses PyPI Trusted Publishing through GitHub OIDC, so it must
not rely on stored upload credentials.

## One-time maintainer prerequisites

Before the first release, a project owner must register a PyPI Trusted Publisher
for the `tsugi-mend` project with these exact values:

- PyPI project: `tsugi-mend`
- GitHub owner: `tsugiai`
- GitHub repository: `tsugi-mend`
- Workflow filename: `release.yml`
- GitHub environment: `pypi`

The maintainer also needs permission to create tags and GitHub Releases in the
`tsugiai/tsugi-mend` repository. No stored PyPI credential is required for this
workflow.

## Release steps

1. Prepare a normal release PR that updates the package version in
   `pyproject.toml` and any release notes. This workflow PR intentionally leaves
   the version at `0.1.0`.
2. Merge the release PR after CI is green on `main`.
3. Create and publish a GitHub Release from `main` with a tag that matches the
   package version, such as `v0.1.1`.
4. After the GitHub Release is published, the `Release` workflow builds the sdist
   and wheel with `python -m build`.
5. The workflow publishes the contents of `dist/` to PyPI with
   `pypa/gh-action-pypi-publish` using Trusted Publishing.
6. Verify that the workflow completed successfully and that the new version is
   visible on PyPI.

If publishing fails because PyPI rejects the OIDC claim, confirm that the PyPI
Trusted Publisher registration matches the repository, workflow filename, and
environment listed above.
