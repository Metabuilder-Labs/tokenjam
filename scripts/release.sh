#!/usr/bin/env bash
# Bump every lockstep-required version file from one source of truth.
#
# Usage: scripts/release.sh X.Y.Z
#
# Updates:
#   - pyproject.toml            (PyPI package version)
#   - sdk-ts/package.json       (npm @tokenjam/sdk version — hand-published, not CI-synced)
#   - npm-wrapper/package.json  (local `npm pack` floor; CI overwrites this from the release
#                                tag on publish, but keeping it current avoids local drift)
#
# After bumping, greps the repo for any other literal occurrence of the OLD version
# string outside the files above, so a straggler (Dockerfile ARG, docs snippet, etc.)
# surfaces here instead of silently shipping stale.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 X.Y.Z" >&2
  exit 1
fi

NEW_VERSION="$1"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: '$NEW_VERSION' is not a plain X.Y.Z semver" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OLD_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"

if [[ -z "$OLD_VERSION" ]]; then
  echo "error: could not find current version in pyproject.toml" >&2
  exit 1
fi

if [[ "$OLD_VERSION" == "$NEW_VERSION" ]]; then
  echo "error: new version ($NEW_VERSION) matches current version — nothing to bump" >&2
  exit 1
fi

echo "Bumping to $NEW_VERSION (pyproject.toml was $OLD_VERSION)"

# pyproject.toml: only the top-level [project] version line, not any dependency pin.
# Done via python (not sed -i) so this works identically on BSD sed (macOS) and GNU
# sed (CI) — the "replace first match only" range address is a GNU-only extension.
python3 -c "
import re, sys
path = 'pyproject.toml'
with open(path) as f:
    content = f.read()
new_content, count = re.subn(
    r'^version = \"$OLD_VERSION\"$',
    'version = \"$NEW_VERSION\"',
    content,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    print('error: expected exactly one version line to replace, found ' + str(count), file=sys.stderr)
    sys.exit(1)
with open(path, 'w') as f:
    f.write(new_content)
"
echo "  updated pyproject.toml"

# sdk-ts is hand-published as-is (no CI auto-sync) — this is the one that must be
# in lockstep with pyproject.toml or `npm publish` fails at release time.
# npm-wrapper is CI-synced from the release tag on publish, so its checked-in value
# is only a floor for local `npm pack` and may already be behind — bump it too for
# hygiene, but don't require it to have matched pyproject.toml beforehand.
for pkg in sdk-ts/package.json npm-wrapper/package.json; do
  # Targeted regex replace (not JSON.parse + stringify) so the diff stays a
  # one-line version bump instead of reformatting the whole file.
  node -e "
    const fs = require('fs');
    const path = '$pkg';
    const content = fs.readFileSync(path, 'utf8');
    const updated = content.replace(/^(\s*\"version\":\s*\")[^\"]*(\")/m, \`\$1$NEW_VERSION\$2\`);
    if (updated === content) {
      console.error(\`error: no version field found in \${path}\`);
      process.exit(1);
    }
    fs.writeFileSync(path, updated);
  "
  echo "  updated $pkg"
done

echo
echo "Checking for stragglers still referencing $OLD_VERSION..."
STRAGGLERS="$(grep -rl --fixed-strings "$OLD_VERSION" \
  --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=dist \
  --exclude-dir=.venv --exclude-dir=build --exclude-dir='*.egg-info' \
  . 2>/dev/null || true)"

if [[ -n "$STRAGGLERS" ]]; then
  echo "warning: the following files still reference the old version ($OLD_VERSION) — check whether they need a manual bump too:" >&2
  echo "$STRAGGLERS" >&2
else
  echo "no stragglers found."
fi

echo
echo "Done. Review the diff, commit, then cut the release:"
echo "  git add pyproject.toml sdk-ts/package.json npm-wrapper/package.json"
echo "  git commit -m 'chore: bump version to $NEW_VERSION'"
echo "  gh release create v$NEW_VERSION --generate-notes"
