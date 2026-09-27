# Releasing (maintainers)

Wheels are built and published to PyPI by the `release` workflow through **Trusted Publishing**
(OpenID Connect); no API token is stored in the repository. This page is for maintainers and is
not part of the documentation site.

## One-time setup (PyPI)

1. Sign in to PyPI and open **Your projects → Publishing** (or, for the first release, **Add a new
   pending publisher** at https://pypi.org/manage/account/publishing/).
2. Register a trusted publisher with:
   - PyPI project name: `nefi`
   - Owner: `ContinuumCoder`
   - Repository name: `NEFI`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. On GitHub, create the `pypi` environment (Settings → Environments) — optionally restrict it to
   tags `v*` and require a reviewer.

## Cutting a release

```bash
# bump the version in pyproject.toml and nefi/__init__.py, update CHANGELOG.md, then
git tag v0.1.0
git push origin main --tags
```

The tag triggers `.github/workflows/release.yml`: it builds the sdist and wheel, smoke-tests the
wheel (`nefi list instances`), and publishes to PyPI. Afterwards:

```bash
pip install nefi
```

To test the packaging locally without publishing:

```bash
python -m pip install build && python -m build
python -m venv /tmp/v && /tmp/v/bin/pip install dist/nefi-*.whl && /tmp/v/bin/nefi list instances
```

## Versioning

Semantic versioning; `0.x` while the public API settles. Every release must pass
`make lint`, `make test` and `make smoke`, and the documentation site must build with
`mkdocs build --strict`.
