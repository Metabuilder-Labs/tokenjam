"""The two model-routing write kinds, rendered.

Both are deterministic edits of a value that is already written down somewhere,
which is the only reason either is allowed a real write at all:

  * ``agent_model`` (Claude Code) rewrites the ``model:`` key in the YAML
    frontmatter of a ``.claude/agents/<name>.md`` file. The key either exists or
    is added inside the existing fence; nothing else in the file moves.
  * ``model_swap`` (SDK) replaces an exact model-id string with another exact
    model-id string, in exactly one file of a git repo the user registered.

Neither kind ever generates code, inserts multiple lines of logic, or edits a
file it had to interpret to find. A fix that needs code understanding is a
one-paste artifact by definition, and stays on the card.

Rendering is separated from the apply machinery in ``relearn_apply`` (backup,
git commit, ledger, revert) so both kinds inherit that machinery unchanged: the
functions here only compute the new bytes and the reasons to refuse.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from tokenjam.core.optimize.relearn_apply import (
    RelearnApplyRefused,
    _run_git,
    git_repo_root,
)

#: Apply-kind discriminators, carried on the proposal and on the ledger record.
APPLY_KIND_AGENT_MODEL = "agent_model"
APPLY_KIND_MODEL_SWAP = "model_swap"
APPLY_KINDS = (APPLY_KIND_AGENT_MODEL, APPLY_KIND_MODEL_SWAP)

#: Where a model id may be swapped. Deliberately narrow: source and config
#: files where a model id is a literal, never documentation or lockfiles.
MODEL_SWAP_EXTENSIONS = frozenset({
    ".py", ".ts", ".js", ".go", ".rb", ".java", ".yaml", ".yml", ".json", ".env",
})

#: Directories never searched for the model id. Vendored or generated trees
#: would produce spurious extra matches and turn a clean single match into a
#: refusal, or worse, into a write somewhere the user does not edit.
MODEL_SWAP_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".tox", "vendor", ".next",
})

#: Cap on files walked, so a precheck against a huge checkout stays bounded.
MODEL_SWAP_MAX_FILES = 20_000


def is_swappable_file(path: Path) -> bool:
    """Whether a model id may be swapped in this file.

    ``.env`` has no suffix of its own, so it is matched by name; every other
    allowed type is matched by extension.
    """
    return path.suffix.lower() in MODEL_SWAP_EXTENSIONS or path.name == ".env"

_FRONTMATTER_FENCE = "---"


# --------------------------------------------------------------------------- #
# agent_model: the `model:` key of a Claude Code agent file
# --------------------------------------------------------------------------- #

def default_agent_file_path(scope: str, repo_cwd: str, agent_name: str) -> str:
    """Where ``agent_name``'s definition lives for ``scope``.

    Project scope is tried first by the caller; ``user-global`` is only correct
    when the agent's sessions span repos, which is the same scope routing the
    relearn's note/skill deliveries use. Returns ``""`` when a project-scoped lookup
    has no repo to anchor on, so the caller must ask rather than guess.
    """
    if not agent_name:
        return ""
    filename = f"{agent_name}.md"
    if scope == "user-global":
        return str(Path.home() / ".claude" / "agents" / filename)
    if not repo_cwd:
        return ""
    return str(Path(repo_cwd) / ".claude" / "agents" / filename)


def is_agent_file(target: Path) -> bool:
    """A Markdown file inside an ``agents`` directory. The narrow allowlist that
    keeps this apply kind from being pointed at an arbitrary path."""
    return target.suffix.lower() == ".md" and target.parent.name == "agents"


def _frontmatter_bounds(text: str) -> tuple[int, int] | None:
    """``(open_end, close_start)`` character offsets of the frontmatter body, or
    ``None`` when the file does not open with a fence."""
    if not text.startswith(_FRONTMATTER_FENCE + "\n"):
        return None
    open_end = len(_FRONTMATTER_FENCE) + 1
    close = re.search(rf"^{_FRONTMATTER_FENCE}\s*$", text[open_end:], re.MULTILINE)
    if close is None:
        return None
    return open_end, open_end + close.start()


def render_agent_model(pre_image: str | None, proposed_model: str) -> tuple[str | None, str]:
    """The agent file's new content with ``model:`` set to ``proposed_model``.

    Returns ``(content, "")`` on success and ``(None, reason)`` when the edit
    cannot be made deterministically: no file yet (the caller falls back to the
    guidance-block write), no frontmatter fence to edit, or the key already
    holds the proposed value (an apply that changes nothing must not be
    recorded as a fix).
    """
    if not proposed_model:
        return None, "no proposed model given for the agent file edit."
    if pre_image is None:
        return None, "no agent file at that path yet."
    bounds = _frontmatter_bounds(pre_image)
    if bounds is None:
        return None, (
            "that file has no YAML frontmatter block, so there is no model key "
            "to set. Leaving it untouched."
        )
    open_end, close_start = bounds
    head, block, tail = (
        pre_image[:open_end], pre_image[open_end:close_start], pre_image[close_start:],
    )
    existing = re.search(r"^model:[ \t]*(.*)$", block, re.MULTILINE)
    if existing is not None:
        if existing.group(1).strip() == proposed_model:
            return None, f"that agent already runs on {proposed_model}."
        new_block = (
            block[:existing.start()] + f"model: {proposed_model}" + block[existing.end():]
        )
    else:
        separator = "" if block.endswith("\n") else "\n"
        new_block = f"{block}{separator}model: {proposed_model}\n"
    return head + new_block + tail, ""


# --------------------------------------------------------------------------- #
# model_swap: one exact model-id string, in one registered repo
# --------------------------------------------------------------------------- #

def _candidate_files(root: Path) -> list[Path]:
    """Allowlisted files under ``root``, skipping vendored/generated trees."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if len(files) >= MODEL_SWAP_MAX_FILES:
            break
        if path.is_symlink() or not path.is_file():
            continue
        if not is_swappable_file(path):
            continue
        if any(part in MODEL_SWAP_SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def find_model_id_files(root: Path, model_id: str) -> list[Path]:
    """Every allowlisted file under ``root`` containing ``model_id`` verbatim.

    Exact substring matching only. The swap is safe precisely because it never
    interprets the surrounding code, so it must never guess at a model id it did
    not find written out in full.
    """
    if not model_id:
        return []
    hits: list[Path] = []
    for path in _candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if model_id in text:
            hits.append(path)
    return sorted(hits)


def _is_clean_in_git(repo_root: Path, target: Path) -> bool:
    """True when ``target`` has no uncommitted change. A dirty file is refused:
    the revert path restores a pre-image, and mixing that with edits the user
    has not committed could throw their work away."""
    rel = str(target.relative_to(repo_root))
    status = _run_git(["status", "--porcelain", "--", rel], repo_root)
    if status is None or status.returncode != 0:
        return False
    return not (status.stdout or "").strip()


#: The one precheck failure the USER can clear from the Review inbox, flagged
#: separately from every other. The remaining gates report something about the
#: repo the reader cannot fix by answering a question (not a git repo, the model
#: id in several files, the file dirty), so a row that hits one of those stays
#: advise-only and says so. "We were never told where the source is" is a
#: different kind of gap: it is a question, and a question is answerable.
#:
#: Callers must branch on this flag, never on the reason string — an
#: exit-code-style tolerance re-derived from prose is how two gates come to look
#: like one.
MODEL_SWAP_NEEDS_SOURCE_PATH = "needs_source_path"

MODEL_SWAP_NO_SOURCE_PATH_REASON = (
    "tokenjam has not been told where this agent's source lives, so it does "
    "not know which file sets the model id. Point it at the agent's local "
    "checkout once and it can make the substitution."
)


def model_swap_precheck(source_path: str, current_model: str) -> dict:
    """Whether the gated model-id swap may run, and where it would write.

    Every precondition must hold; any failure returns ``{"ok": False, "reason":
    ...}``. The preconditions, in order: a registered source path that exists,
    a git repo, exactly one file carrying the model id, and that file clean in
    the working tree.

    The FIRST gate additionally sets ``needs_source_path``, because it is the
    only one the reader can clear themselves — see
    ``MODEL_SWAP_NEEDS_SOURCE_PATH``. Nothing else about the contract changes:
    the path still has to be registered in the user's own config before a write
    is attempted, and every later gate is re-run against the registered path.
    """
    if not source_path:
        return {
            "ok": False,
            "reason": MODEL_SWAP_NO_SOURCE_PATH_REASON,
            MODEL_SWAP_NEEDS_SOURCE_PATH: True,
        }
    root = Path(source_path).expanduser()
    if not root.is_dir():
        return {"ok": False, "reason": f"the registered source path {root} is not a directory."}
    repo_root = git_repo_root(root)
    if repo_root is None:
        return {"ok": False, "reason": (
            f"{root} is not a git repository. The swap is only offered where the "
            f"edit is revertable through git."
        )}
    matches = find_model_id_files(root, current_model)
    if not matches:
        return {"ok": False, "reason": (
            f"the model id {current_model} does not appear in any source file "
            f"under {root}, so there is nothing to swap. It is probably set "
            f"from an environment variable or a value built at runtime."
        )}
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in matches[:5])
        return {"ok": False, "reason": (
            f"the model id {current_model} appears in {len(matches)} files "
            f"({listed}), so the edit is not a single deterministic "
            f"substitution. Paste the change instead."
        )}
    target = matches[0]
    if not _is_clean_in_git(repo_root, target):
        return {"ok": False, "reason": (
            f"{target} has uncommitted changes. Commit or stash them first so "
            f"the swap can be reverted cleanly."
        )}
    return {
        "ok": True,
        "reason": "",
        "target_path": str(target),
        "repo_root": str(repo_root),
    }


def render_model_swap(
    pre_image: str | None, current_model: str, proposed_model: str,
) -> tuple[str | None, str]:
    """The file's new content with every ``current_model`` occurrence replaced.

    Returns ``(None, reason)`` when the substitution is not well defined: no
    file, a missing model id, or identical ids.
    """
    if not current_model or not proposed_model:
        return None, "both the current and the proposed model id are required."
    if current_model == proposed_model:
        return None, "the current and proposed model ids are identical."
    if pre_image is None:
        return None, "no file at that path to edit."
    if current_model not in pre_image:
        return None, f"the model id {current_model} is not in that file any more."
    return pre_image.replace(current_model, proposed_model), ""


def build_model_plan(cluster: dict, target: Path, pre_image: str | None) -> str:
    """New content for whichever model apply kind ``cluster`` names.

    Raises ``RelearnApplyRefused`` with the refusal reason, which the API layer
    surfaces as a 409 and the card renders as the fallback explanation.
    """
    apply_kind = str(cluster.get("apply_kind") or "")
    if apply_kind == APPLY_KIND_AGENT_MODEL:
        if not is_agent_file(target):
            raise RelearnApplyRefused(
                f"{target} is not an agent definition file. This edit only "
                f"applies to a .md file in an agents directory."
            )
        content, reason = render_agent_model(
            pre_image, str(cluster.get("proposed_model") or ""),
        )
    elif apply_kind == APPLY_KIND_MODEL_SWAP:
        if not is_swappable_file(target):
            raise RelearnApplyRefused(
                f"{target} is not one of the file types a model id may be "
                f"swapped in."
            )
        # Re-run every precondition here, not only when the card was built: the
        # repo can have moved, gained a second occurrence of the model id, or
        # picked up uncommitted edits between the two moments.
        check = model_swap_precheck(
            str(cluster.get("source_path") or ""), str(cluster.get("current_model") or ""),
        )
        if not check["ok"]:
            raise RelearnApplyRefused(check["reason"])
        if Path(check["target_path"]) != target:
            raise RelearnApplyRefused(
                f"the model id now lives in {check['target_path']}, not "
                f"{target}. Refusing to write the stale target."
            )
        content, reason = render_model_swap(
            pre_image,
            str(cluster.get("current_model") or ""),
            str(cluster.get("proposed_model") or ""),
        )
    else:
        raise RelearnApplyRefused(f"unknown apply kind {apply_kind!r}.")
    if content is None:
        raise RelearnApplyRefused(reason)
    return content


def register_agent_source_path(config, agent_id: str, source_path: str) -> str:
    """Persist ``source_path`` as ``[agents.<agent_id>] source_path`` in the
    user's own tj config, and return the resolved absolute path.

    This exists so the Review inbox can offer a model swap on an agent whose
    source path was never registered. It does NOT weaken the safety property
    that made ``source_path`` config-only: the write still reads the path out of
    the config, so a caller cannot aim an edit at an arbitrary repo by putting a
    path in a request body — it can only ask the user's config to be changed,
    which is an act the user performed deliberately from their own machine, and
    which is durable and inspectable afterwards.

    Registration is per AGENT, which is what makes one answer unlock every other
    proposal for the same agent: the next recompute reads this path for all of
    them.

    Refuses rather than guesses. A path that does not exist, is not a directory,
    or is not a git repo cannot be registered at all, because every one of those
    would produce a card that offers an apply and then fails on the click.
    """
    from tokenjam.core.config import write_config

    if not agent_id:
        raise RelearnApplyRefused("no agent to register a source path for.")
    resolved = resolve_source_path(source_path)

    path = getattr(config, "config_path", None)
    if path is None:
        raise RelearnApplyRefused(
            "no tj config file to register the path in. Run `tj onboard` first."
        )
    agents = getattr(config, "agents", None)
    if agents is None:
        raise RelearnApplyRefused("this config carries no agents table.")
    existing = agents.get(agent_id)
    if existing is None:
        from tokenjam.core.config import AgentConfig

        existing = AgentConfig()
        agents[agent_id] = existing
    existing.source_path = resolved
    write_config(config, Path(path))
    return resolved


def resolve_source_path(source_path: str) -> str:
    """The absolute source root ``source_path`` names, or a refusal.

    Split out of ``register_agent_source_path`` so a PREVIEW can resolve and gate
    a path without persisting anything: a dry run that left a registration behind
    would make "preview" a write, and a refused apply would leave the config
    pointing at a repo the swap turned out to be impossible in.

    Refuses rather than guesses, because each of these would otherwise produce a
    card that offers an apply and then fails on the click.
    """
    if not source_path.strip():
        raise RelearnApplyRefused("a source path is required.")
    root = Path(source_path).expanduser()
    # A file is accepted and read as "the repo this file is in": the reader is
    # being asked where the agent's code lives, and answering with the file they
    # know sets the model is a reasonable reading of that question.
    if root.is_file():
        root = root.parent
    if not root.is_dir():
        raise RelearnApplyRefused(f"{root} is not a directory that exists.")
    if git_repo_root(root) is None:
        raise RelearnApplyRefused(
            f"{root} is not inside a git repository. The swap is only offered "
            f"where the edit is revertable through git."
        )
    return str(root.resolve())


def preview_model_swap(
    target_path: str, current_model: str, proposed_model: str,
) -> dict:
    """The diff a swap WOULD write, in the dry-run shape the card renders.

    Reads the target and computes the substitution, and writes nothing — neither
    the file nor the config. Same keys the apply path's dry run returns, so the
    row renders one preview regardless of which endpoint produced it.
    """
    target = Path(target_path)
    try:
        pre_image = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RelearnApplyRefused(f"cannot read {target}: {exc}") from exc
    new_content, reason = render_model_swap(pre_image, current_model, proposed_model)
    if new_content is None:
        raise RelearnApplyRefused(reason)
    return {
        "dry_run": True,
        "kind": APPLY_KIND_MODEL_SWAP,
        "target_path": str(target),
        "diff": "".join(difflib.unified_diff(
            pre_image.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile="before", tofile="after", n=2,
        )),
    }
