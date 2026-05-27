# Releasing tsugi-mend

`tsugi-mend` is published to [PyPI](https://pypi.org/project/tsugi-mend/) by the
`.github/workflows/release.yml` workflow using **PyPI Trusted Publishing (OIDC)**.
This is tokenless: there is no API token, no `password:`, and no secret stored in
the repository or the workflow. GitHub Actions mints a short-lived OIDC identity
for this exact repository + workflow, and PyPI verifies it against a Trusted
Publisher you register once (below).

## One-time maintainer setup (account-bound; cannot be automated in the repo)

These steps require a PyPI account with owner rights on the project and must be
done by hand. They are intentionally NOT in any workflow file.

1. **Create the project on PyPI** (only needed if `tsugi-mend` does not already
   exist there). For a brand-new project you may need a one-time "pending"
   publisher; see step 2.

2. **Register the GitHub Actions Trusted Publisher** on PyPI:

   - Go to <https://pypi.org/manage/account/publishing/> (or, for an existing
     project, the project's **Settings -> Publishing** page).
   - Add a new "GitHub" trusted publisher with these EXACT values:

     | Field                | Value                  |
     | -------------------- | ---------------------- |
     | PyPI project name    | `tsugi-mend`           |
     | Owner                | `tsugiai`              |
     | Repository name      | `tsugi-mend`           |
     | Workflow filename    | `release.yml`          |
     | Environment name     | `pypi`                 |

   The workflow filename and environment name MUST match what the workflow
   declares (`release.yml` and `environment: name: pypi`). A mismatch causes the
   publish step to fail the OIDC check.

3. **(Recommended) Protect the `pypi` GitHub Environment**: in the GitHub repo,
   under **Settings -> Environments -> pypi**, add a required-reviewer or
   branch/tag restriction so only intended releases can publish.

## Cutting a release

The release pipeline runs on a published GitHub Release (recommended) and also on
any pushed `v*` tag. Version is set in `pyproject.toml`.

1. Bump `version` in `pyproject.toml` (e.g. `0.1.0 -> 0.1.1`) and commit on `main`
   via the normal PR flow. (This PR does NOT bump the version; the workflow is
   first exercised on the next real release.)

2. Tag and create a GitHub Release on the merge commit:

   ```bash
   git checkout main && git pull
   git tag v0.1.1
   git push origin v0.1.1
   gh release create v0.1.1 --title "v0.1.1" --notes "..."
   ```

   The tag and the `pyproject.toml` version should match.

3. The `Release` workflow then:
   - builds the sdist + wheel with `python -m build`,
   - runs `twine check` on the artifacts,
   - publishes to PyPI via `pypa/gh-action-pypi-publish` using OIDC (no token).

4. Confirm the new version on <https://pypi.org/project/tsugi-mend/> and install:

   ```bash
   pip install --upgrade tsugi-mend
   ```

## Notes

- The workflow grants `id-token: write` only on the publish job (plus
  `contents: read`); everything else is read-only. It uses no secrets and does
  not use `pull_request_target`.
- If you ever need to publish to TestPyPI first, add a separate step with
  `repository-url: https://test.pypi.org/legacy/` and register a matching Trusted
  Publisher on TestPyPI. Do not add tokens.
