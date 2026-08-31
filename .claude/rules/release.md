---
description: Release, packaging and CI rules — version lockstep, publish workflows, extras.
paths:
  - "pyproject.toml"
  - "sdk-ts/package.json"
  - "npm-wrapper/**"
  - "scripts/release.sh"
  - "Makefile"
  - ".github/workflows/**"
---

# Release / packaging / CI rules

### Critical Rule 15 — Version bump on release

Run `./scripts/release.sh X.Y.Z` (or `make release VERSION=X.Y.Z`) before creating a GitHub release;
it bumps `pyproject.toml` `version` and `sdk-ts/package.json` `"version"` (plus
`npm-wrapper/package.json` for hygiene) from one input and greps for stragglers. The publish
workflows (`publish-pypi.yml`, `publish-npm.yml`) trigger on `release published` events and will fail
with 403 if the version already exists on PyPI/npm; a left-behind `sdk-ts/package.json` bump is
additionally caught pre-publish by the `version-lockstep` CI job and a tag-match guard in
`publish-npm.yml`.

## Releases

PyPI and npm publishes are triggered by GitHub Release events (`.github/workflows/publish-pypi.yml`,
`publish-npm.yml`, both `on: release: types: [published]`). Release flow:

1. Bump versions with `./scripts/release.sh X.Y.Z` (or `make release VERSION=X.Y.Z`) — this bumps `pyproject.toml`, `sdk-ts/package.json`, and `npm-wrapper/package.json` from one input and greps for any other file still referencing the old version.
2. Commit and merge to `main`.
3. Create a GitHub Release with tag `vX.Y.Z` (e.g. via `gh release create vX.Y.Z --generate-notes`). Publishing the release fires both workflows.

`publish-npm.yml` has **two independent jobs**: `publish-npm` (the `@tokenjam/sdk` package from
`sdk-ts/`, version read from `sdk-ts/package.json`) and `publish-npm-wrapper` (the unscoped
`tokenjam` wrapper from `npm-wrapper/`). The wrapper job **derives its published version from the
release tag** (`vX.Y.Z` → `X.Y.Z`, via `npm version --no-git-tag-version --allow-same-version`), so
the wrapper can never drift from the release and needs no manual version bump —
`npm-wrapper/package.json`'s literal is only a floor for local `npm pack`. Both jobs use
`secrets.NPM_TOKEN`, which must be able to publish the unscoped `tokenjam` name in addition to the
`@tokenjam` scope.

Two guards catch a left-behind `sdk-ts/package.json` bump (the one file that is hand-published as-is,
with no CI auto-sync): the `version-lockstep` job in `ci.yml` fails every PR/push to `main` where it
doesn't match `pyproject.toml`, and `publish-npm.yml`'s `publish-npm` job re-checks it against the
release tag itself before installing/building/publishing — so a mismatch fails fast rather than
mid-publish with a confusing "version already exists" error.

If a version already exists on PyPI or npm, the publish workflow fails with 403 — bump again rather
than retrying.

## Packaging

Build system is hatchling. `[tool.hatch.build.targets.wheel] packages = ["tokenjam"]` — the package
directory is `tokenjam/` (matching the PyPI name); only the *CLI command* is `tj`
(`[project.scripts] tj = "tokenjam.cli.main:cli"`). Non-`.py` assets under the package ship in the
wheel automatically — this is how the vendored UI (`tokenjam/ui/index.html`,
`tokenjam/ui/vendor/*`) and `tokenjam/pricing/models.toml` reach users.

Key runtime dependency: `pytz` is required by DuckDB for `TIMESTAMPTZ` column handling — it's listed
explicitly in `dependencies` because DuckDB doesn't declare it on all platforms.

**The `tj` npm wrapper** (`npm-wrapper/`) is a separate, dependency-free npm package named
`tokenjam` (unscoped, distinct from the `@tokenjam/sdk` SDK package; the bare `tj` name is already
taken on npm by an unrelated pub/sub library, so the PACKAGE is `tokenjam` while the installed BIN is
still `tj`) whose only job is to make `npx tokenjam` work. `bin/tj.js` shells out to the Python CLI
via the first available runner (`uvx --from tokenjam tj` → `pipx run --spec tokenjam tj` → an
installed `tj` on PATH) and passes every arg through. It is published to npm by the
`publish-npm-wrapper` job in `publish-npm.yml`, which derives the published version from the
release tag — you do **not** need to bump `npm-wrapper/package.json` on release (CI overwrites it
from the tag; keep the literal roughly current so local `npm pack` is sane). `npm-wrapper/` has no CI
test (no Python to drive in the JS lane); the publish job runs a `node -c bin/tj.js` syntax guard,
and you can validate the same locally with `node -c npm-wrapper/bin/tj.js`.

**Optional extras** (declared under `[project.optional-dependencies]`):
- `tokenjam[bloat]` — `llmlingua>=0.2`, used by the Trim analyzer. Pulls PyTorch + transformers, which is a multi-gigabyte download — that weight is the whole reason it is kept out of the base install. The analyzer self-registers without the extra installed; the deferred `import llmlingua` inside the analysis function body raises a typed message pointing the user at the install command.
- Framework extras `[langchain]`, `[crewai]`, `[autogen]`, `[litellm]` for SDK patches.
- `[dev]` for local development (`pytest`, `ruff`, `mypy`, `httpx`).
- `[mcp]` — empty no-op alias. `fastmcp` moved into the base install in v0.3.5 (#101), so the FastMCP stdio server (`tj mcp`) works on a plain `pipx install tokenjam`. The extra is kept so old `tokenjam[mcp]` install commands still succeed; it pulls nothing extra.

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`:
- **`lint`** job: Python 3.12 — `ruff check tokenjam/` and `mypy tokenjam/`
- **`test`** job: Python 3.10/3.11/3.12 matrix — `pytest -n auto --dist loadscope tests/unit/ tests/synthetic/ tests/agents/ tests/integration/` (parallelized via `pytest-xdist`; local runs, e.g. from `Makefile` or `CONTRIBUTING.md`, stay serial — no need for `-n auto` outside CI)
- **`test-ts`** job: Node 22 — `npm install && npm test` in `sdk-ts/`

All steps are blocking. There is no pre-commit configuration in this repo; `ruff` and `mypy` only run
in CI. Run them locally before pushing.
