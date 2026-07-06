"""Best-effort SDK / agent-framework detection for `tj onboard`'s bare-onboard
outro (#85).

Bare `tj onboard` (no `--claude-code`/`--codex`) prints one hardcoded
`patch_anthropic()` + `@watch()` snippet regardless of the project's actual
stack. This scans the project's dependency manifests for known LLM provider
SDKs and agent frameworks so the outro can print the *tailored* `patch_*()`
call, import path, and `tokenjam[extra]` install hint instead.

Manifest-only by design: it never imports or executes project code, so it
can't be fooled by conditional imports and can't crash on a project with
import side effects. Detection is necessarily incomplete (a project can
declare deps in a lockfile-only or Pipfile flow this doesn't read) — that's
fine, the caller always has the generic snippet as a fallback.
"""
from __future__ import annotations

import configparser
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class SdkMatch:
    """One detected provider SDK or agent framework, with its onboard advice."""

    key: str
    label: str
    kind: str  # "provider" | "framework"
    import_line: str
    patch_call: str
    extra: str | None  # tokenjam[<extra>] name, or None when no extra applies


# Public patch functions, from tokenjam/sdk/integrations/*.py — keep these in
# sync with that package; a renamed patch_* function or extra silently makes
# the onboard advice wrong (fail-open, no test catches a typo in the string
# itself, only that *some* string got printed).
_PROVIDERS: tuple[SdkMatch, ...] = (
    SdkMatch("anthropic", "Anthropic", "provider",
             "from tokenjam.sdk.integrations.anthropic import patch_anthropic",
             "patch_anthropic()", None),
    SdkMatch("openai", "OpenAI", "provider",
             "from tokenjam.sdk.integrations.openai import patch_openai",
             "patch_openai()", None),
    SdkMatch("gemini", "Gemini", "provider",
             "from tokenjam.sdk.integrations.gemini import patch_gemini",
             "patch_gemini()", None),
    SdkMatch("bedrock", "Bedrock", "provider",
             "from tokenjam.sdk.integrations.bedrock import patch_bedrock",
             "patch_bedrock()", None),
    SdkMatch("litellm", "LiteLLM", "provider",
             "from tokenjam.sdk.integrations.litellm import patch_litellm",
             "patch_litellm()", "litellm"),
)

_FRAMEWORKS: tuple[SdkMatch, ...] = (
    SdkMatch("langchain", "LangChain", "framework",
             "from tokenjam.sdk.integrations.langchain import patch_langchain",
             "patch_langchain()", "langchain"),
    SdkMatch("langgraph", "LangGraph", "framework",
             "from tokenjam.sdk.integrations.langgraph import patch_langgraph",
             "patch_langgraph()", None),
    SdkMatch("crewai", "CrewAI", "framework",
             "from tokenjam.sdk.integrations.crewai import patch_crewai",
             "patch_crewai()", "crewai"),
    SdkMatch("autogen", "AutoGen", "framework",
             "from tokenjam.sdk.integrations.autogen import patch_autogen",
             "patch_autogen()", "autogen"),
    SdkMatch("llamaindex", "LlamaIndex", "framework",
             "from tokenjam.sdk.integrations.llamaindex import patch_llamaindex",
             "patch_llamaindex()", None),
)

_ALL_MATCHES: dict[str, SdkMatch] = {m.key: m for m in (*_PROVIDERS, *_FRAMEWORKS)}

# PEP 503 normalized declared-dependency name -> detection key. Several
# distribution names can map to one key (both Google SDKs, both llama-index
# dists, autogen's pyautogen/ag2 fork).
_PACKAGE_TO_KEY: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google-generativeai": "gemini",
    "google-genai": "gemini",
    "boto3": "bedrock",
    "litellm": "litellm",
    "langchain": "langchain",
    "langchain-core": "langchain",
    "langgraph": "langgraph",
    "crewai": "crewai",
    "pyautogen": "autogen",
    "ag2": "autogen",
    "llama-index": "llamaindex",
    "llama-index-core": "llamaindex",
}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")


def _normalize(name: str) -> str:
    """PEP 503 normalize: lowercase, collapse runs of -/_/. to a single '-'."""
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _requirement_name(line: str) -> str | None:
    """Bare package name from one requirements.txt-style line, or None."""
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http://", "https://")):
        return None
    m = _NAME_RE.match(line)
    if not m:
        return None
    return m.group(1).split("[", 1)[0]


def _names_from_pyproject(path: Path) -> set[str]:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        # UnicodeDecodeError (a ValueError, not an OSError) fires when the
        # manifest has non-UTF-8 bytes; fail open to "no detection".
        return set()
    names: set[str] = set()

    project = data.get("project", {}) or {}
    for dep in project.get("dependencies", []) or []:
        if isinstance(dep, str):
            n = _requirement_name(dep)
            if n:
                names.add(n)
    for deps in (project.get("optional-dependencies", {}) or {}).values():
        for dep in deps or []:
            if isinstance(dep, str):
                n = _requirement_name(dep)
                if n:
                    names.add(n)

    poetry = ((data.get("tool", {}) or {}).get("poetry", {}) or {})
    for section in ("dependencies", "dev-dependencies"):
        for name in (poetry.get(section, {}) or {}):
            if name.lower() != "python":
                names.add(name)
    for group in (poetry.get("group", {}) or {}).values():
        for name in (group.get("dependencies", {}) or {}):
            if name.lower() != "python":
                names.add(name)

    return names


def _names_from_requirements(path: Path) -> set[str]:
    names: set[str] = set()
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        # UnicodeDecodeError (a ValueError, not an OSError) fires when the
        # manifest has non-UTF-8 bytes; fail open to "no detection".
        return names
    for line in text.splitlines():
        n = _requirement_name(line)
        if n:
            names.add(n)
    return names


def _names_from_setup_cfg(path: Path) -> set[str]:
    names: set[str] = set()
    parser = configparser.ConfigParser()
    try:
        parser.read(path)
    except (OSError, UnicodeDecodeError, configparser.Error):
        # UnicodeDecodeError (a ValueError, not an OSError) fires when the
        # manifest has non-UTF-8 bytes; fail open to "no detection".
        return names
    if parser.has_option("options", "install_requires"):
        for line in parser.get("options", "install_requires").splitlines():
            n = _requirement_name(line)
            if n:
                names.add(n)
    if parser.has_section("options.extras_require"):
        for _extra_name, raw in parser.items("options.extras_require"):
            for line in raw.splitlines():
                n = _requirement_name(line)
                if n:
                    names.add(n)
    return names


def declared_package_names(project_dir: str | Path = ".") -> set[str]:
    """Every dependency name declared across this project's manifests.

    PEP-503 normalized (lowercase, `-`/`_`/`.` collapsed). Reads
    `pyproject.toml` (PEP 621 `[project]` + Poetry `[tool.poetry]`),
    `requirements*.txt`, and `setup.cfg` at the project root — no recursion,
    no lockfiles.
    """
    root = Path(project_dir)
    names: set[str] = set()

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        names |= _names_from_pyproject(pyproject)

    for req_file in sorted(root.glob("requirements*.txt")):
        names |= _names_from_requirements(req_file)

    setup_cfg = root / "setup.cfg"
    if setup_cfg.is_file():
        names |= _names_from_setup_cfg(setup_cfg)

    return {_normalize(n) for n in names}


def detect_stack(project_dir: str | Path = ".") -> list[SdkMatch]:
    """Detect known LLM provider SDKs / agent frameworks declared in this project.

    Returns matches in a stable order (providers before frameworks, each
    alphabetical by key) so the onboard outro's advice order never depends on
    manifest iteration order. Empty when nothing recognized is declared —
    callers fall back to the generic snippet in that case.
    """
    declared = declared_package_names(project_dir)
    found_keys = {key for name, key in _PACKAGE_TO_KEY.items() if name in declared}
    return sorted(
        (_ALL_MATCHES[k] for k in found_keys),
        key=lambda m: (0 if m.kind == "provider" else 1, m.key),
    )


def install_hint(match: SdkMatch) -> str | None:
    """The `pip install tokenjam[...]` hint for a match, or None if no extra applies."""
    if match.extra is None:
        return None
    return f"pip install 'tokenjam[{match.extra}]'"
