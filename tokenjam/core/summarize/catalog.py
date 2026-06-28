"""Loader for the agent-prompt-file catalog — where known prompt files live.

Mirrors `core/pricing.py`: a packaged default (`agent_files.toml` beside this
module) merged with an optional user override (`~/.config/tj/agent_files.toml`),
cached. Curated data — PRs welcome, like `pricing/models.toml`; Anil decides
what ships. Enumerates, per tool (Claude / Gemini / Codex), the global/system
fixed paths plus the project-relative filenames and globs the summarize scan
looks at (DEC-020).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

log = logging.getLogger(__name__)

CATALOG_FILE = Path(__file__).parent / "agent_files.toml"
USER_CATALOG = Path.home() / ".config" / "tj" / "agent_files.toml"

_SAFETY_SECTION = "safety"


@dataclass(frozen=True)
class Catalog:
    """Flattened catalog the scan consumes (unioned across tool sections)."""

    project_files: frozenset[str]      # bare filenames checked at a scan root
    project_globs: tuple[str, ...]     # root-relative globs (e.g. .claude/skills/*/SKILL.md)
    global_paths: tuple[str, ...]      # user/system paths ("~" expanded by the caller)
    forbidden_roots: tuple[str, ...]   # extra dirs never treated as a repo root


def _read(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:  # malformed override → skip, never fatal
        log.warning("Could not read agent-file catalog %s (%s); skipping it.", path, exc)
        return {}


def _merge(base: dict, over: dict) -> dict:
    """Per-section merge: a user section updates the packaged one key-by-key
    (lists replace, not concatenate — predictable)."""
    out: dict = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for section, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(section), dict):
            out[section].update(val)
        else:
            out[section] = val
    return out


@lru_cache(maxsize=1)
def load_catalog() -> Catalog:
    """Packaged catalog merged with the optional user override, cached."""
    raw = _merge(_read(CATALOG_FILE), _read(USER_CATALOG))
    files: set[str] = set()
    globs: list[str] = []
    globals_: list[str] = []
    forbidden: list[str] = []
    for section, body in raw.items():
        if not isinstance(body, dict):
            continue
        if section == _SAFETY_SECTION:
            forbidden.extend(str(x) for x in body.get("forbidden_roots", []))
            continue
        files.update(str(x) for x in body.get("project_files", []))
        globs.extend(str(x) for x in body.get("project_globs", []))
        globals_.extend(str(x) for x in body.get("global_paths", []))
    return Catalog(
        project_files=frozenset(files),
        project_globs=tuple(dict.fromkeys(globs)),       # dedupe, keep order
        global_paths=tuple(dict.fromkeys(globals_)),
        forbidden_roots=tuple(dict.fromkeys(forbidden)),
    )


def clear_catalog_cache() -> None:
    """Drop the cache so the next load re-reads from disk (test hook / runtime reload)."""
    load_catalog.cache_clear()
