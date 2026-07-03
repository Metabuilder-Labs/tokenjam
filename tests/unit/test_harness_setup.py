"""Unit tests for harness run-linkage instrumentation (core.harness_setup)."""
from __future__ import annotations

from tokenjam.core.harness_setup import (
    HELPER_RELPATH,
    MAX_SPAWN_HITS,
    build_run_env_helper,
    python_launcher_snippet,
    scan_spawn_points,
)


def test_helper_stamps_run_id_and_mints_per_launch():
    helper = build_run_env_helper()
    assert "tokenjam.run_id=" in helper
    assert "TJ_RUN_ID" in helper
    # Minted at runtime (per launch), not a baked-in constant.
    assert "date -u" in helper
    assert "OTEL_RESOURCE_ATTRIBUTES" in helper
    # Idempotent: doesn't double-add if already tagged.
    assert "do not double-add" in helper


def test_python_snippet_sets_attribute():
    snip = python_launcher_snippet()
    assert "tokenjam.run_id=" in snip
    assert "OTEL_RESOURCE_ATTRIBUTES" in snip
    assert "setdefault" in snip


def test_scan_detects_spawn_points(tmp_path):
    (tmp_path / "run-loop.sh").write_text(
        "#!/bin/bash\nfor t in tickets; do\n  claude -p \"$t\" &\ndone\n",
        encoding="utf-8",
    )
    (tmp_path / "spawn.py").write_text(
        "import subprocess\nsubprocess.run(['claude', '-p', task])\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("just docs, claude mentioned\n", encoding="utf-8")
    hits = scan_spawn_points(tmp_path)
    files = {h["file"] for h in hits}
    assert "run-loop.sh" in files
    assert "spawn.py" in files
    # Markdown isn't a scanned source extension.
    assert "README.md" not in files
    for h in hits:
        assert "line" in h and "text" in h


def test_scan_skips_vendored_dirs(tmp_path):
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("child_process.spawn('claude')\n", encoding="utf-8")
    assert scan_spawn_points(tmp_path) == []


def test_scan_is_bounded(tmp_path):
    # Far more spawn lines than the hit cap -> bounded, never unbounded.
    big = "\n".join("subprocess.run(['claude'])" for _ in range(MAX_SPAWN_HITS * 3))
    (tmp_path / "many.py").write_text(big, encoding="utf-8")
    assert len(scan_spawn_points(tmp_path)) <= MAX_SPAWN_HITS


def test_scan_prioritizes_harness_files_over_budget(tmp_path):
    # A big monorepo: many unrelated source files (alphabetically before the
    # launcher) must NOT exhaust the budget before the real spawn script is read.
    bulk = tmp_path / "admin-panel"
    bulk.mkdir()
    for i in range(MAX_SPAWN_HITS * 0 + 700):  # > MAX_SCAN_FILES unrelated files
        (bulk / f"a_{i:04d}.ts").write_text("export const x = 1;\n", encoding="utf-8")
    launcher = tmp_path / "scripts" / "govern"
    launcher.mkdir(parents=True)
    (launcher / "run-loop.sh").write_text(
        '#!/bin/bash\n"${GOVERN_CLAUDE_BIN:-claude}" -p "$prompt"\n', encoding="utf-8"
    )
    hits = scan_spawn_points(tmp_path)
    assert any(h["file"].endswith("run-loop.sh") for h in hits)


def test_scan_detects_otel_instrumentation_point(tmp_path):
    (tmp_path / "spawn.sh").write_text(
        'OTEL_RESOURCE_ATTRIBUTES="$attrs" claude -p "$prompt"\n', encoding="utf-8"
    )
    hits = scan_spawn_points(tmp_path)
    assert any("spawn.sh" == h["file"] for h in hits)


def test_scan_missing_dir_returns_empty(tmp_path):
    assert scan_spawn_points(tmp_path / "nope") == []


def test_helper_relpath_is_under_tj():
    assert HELPER_RELPATH.startswith(".tj/")
