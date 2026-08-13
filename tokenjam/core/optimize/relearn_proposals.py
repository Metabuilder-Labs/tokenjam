"""Server-side proposal record: the stored proposals an apply must name.

Before this module, ``POST /relearn/apply`` took the whole cluster in the
request body. That made "human-gated" a property of the UI flow rather than of
the server: any local caller could hand-build a cluster, aim ``target_path``
wherever it liked, and the write machinery would render and apply it. The
detector's own output was never consulted.

Here the detector's clusters ARE the proposals. Every cluster the recompute
persists gets a stable ``proposal_id`` stamped into the cache at detection
time (``relearn_store.write_cache``), and every write path (the API route and
``tj relearn apply``) can only name one of those IDs. A caller cannot invent a
proposal that the detector never produced; a stale client cannot smuggle
different cluster content past the review it was shown.

The ID is a pure function of the cluster's signature, so it is stable across
recomputes: the same recurring failure keeps the same ID from one scan to the
next, and a CLI user who copied an ID out of ``tj relearn list`` yesterday can
still act on it today. Scope and target path stay caller-supplied (the human
edits both before approving; see the apply route) -- only the CONTENT of the
proposal is pinned to what the detector actually found.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

#: Prefix on every stored-proposal ID. Short, greppable, and impossible to
#: confuse with a fix_id (a bare uuid4 hex) in a CLI transcript or a log line.
PROPOSAL_ID_PREFIX = "rp_"

_ID_HEX_LEN = 12

#: The cluster fields the apply machinery reads. Anything else on a stored
#: proposal (display-only counts, novelty flags, suggested targets) is not part
#: of what gets written, so it never reaches ``relearn_apply``.
#:
#: The trailing five are the model-routing fields a cost proposal carries
#: (``core.optimize.model_apply``). They are projected from the STORE for the
#: same reason the rest of the cluster is: the card the human approved was
#: rendered from the stored proposal, so the stored proposal — not the request
#: that echoes it back — is the authoritative record of what they approved.
#: ``source_path`` especially: the model_swap safety case rests on that path
#: having been REGISTERED in the user's own config, and a caller-supplied path
#: would aim the write at any repo on disk.
APPLY_CLUSTER_FIELDS = (
    "signature", "family_key", "title", "proposed_fix", "delivery",
    "sessions", "occurrences", "repos", "examples",
    "apply_kind", "agent_name", "current_model", "proposed_model", "source_path",
    # Past-overspend figure (the one canonical dollar field): carried through
    # to apply time so `apply_relearn_fix` can snapshot the human-reviewed
    # figure onto the applied-fix ledger record's `verify` dict for the
    # Applied tab's figure — a bounded, single read, not a live
    # re-measurement.
    "past_overspend_usd", "past_overspend_tokens",
)

#: Per apply kind, the stored fields the write genuinely cannot be built
#: without. A proposal missing one is refused by name rather than silently
#: falling back to anything the caller sent.
MODEL_ROUTING_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "agent_model": ("proposed_model",),
    "model_swap": ("source_path", "current_model", "proposed_model"),
}


def missing_apply_fields(cluster: dict[str, Any]) -> tuple[str, ...]:
    """The required fields this cluster's ``apply_kind`` needs and does not
    have. Empty for a delivery-mechanism fix (no ``apply_kind``) and for a complete
    model-routing proposal."""
    apply_kind = str(cluster.get("apply_kind") or "")
    if not apply_kind:
        return ()
    required = MODEL_ROUTING_REQUIRED_FIELDS.get(apply_kind, ())
    return tuple(f for f in required if not str(cluster.get(f) or "").strip())


#: Why an advise-only proposal has no apply path, in one line the Review inbox
#: and `tj relearn list` both show. The flag itself is computed by the detector
#: (``analyzers.relearn.build_proposals``): every agent that contributed to the
#: cluster is one tokenjam has no workspace for, so there is no file to write a
#: fix into. Stated rather than merely enforced, because a card that silently
#: lacks an apply button reads as a bug.
ADVISE_ONLY_REASON = (
    "advise-only: every agent in this cluster is a service tokenjam has no "
    "workspace for, so there is no file to write a fix into. Apply the "
    "recommendation yourself, then measure it with "
    "`tj relearn verify --otel`."
)


def advise_only_reason(proposal: dict[str, Any]) -> str | None:
    """The human reason a proposal cannot be applied, or ``None`` when it can.

    One source of truth for the wording, so the CLI, the API payload and the
    inbox row cannot drift into three different explanations of one flag.
    ``advise_only`` is the only reason left: the workspace-less (OTel) lane has
    no file to write into, an advisory family says no action is needed, or the
    window's persona has no workspace surface. There is no budget/ceiling gate
    on top of that any more.
    """
    return ADVISE_ONLY_REASON if proposal.get("advise_only") else None


def advise_snippet_offered(proposal: dict[str, Any]) -> bool:
    """Is this cluster's ``proposed_fix`` a real recommendation the user can act
    on themselves, rather than something tokenjam is about to write for them?

    The Review inbox routes every row onto a three-valued mechanism axis: tj
    can apply it, or tj hands over the exact change. False when the write WAS
    offered (``not advise_only``), and that is the load-bearing half. Unlike a
    cost proposal, where ``suggestion`` and the apply path are separate fields
    describing separate things (``deadweight``'s mcp_remove legitimately
    carries both), a relearn cluster's ``proposed_fix`` is the SAME content the
    write would write. Handing it over is only a distinct offer when tokenjam
    will not write it, so a cluster tokenjam is about to write must not also
    advertise "copy this" and duplicate its own fix on the row. Hence
    "advise" in the name.
    """
    if not str(proposal.get("proposed_fix") or "").strip():
        return False
    return bool(proposal.get("advise_only"))


def proposal_id_for(signature: str) -> str:
    """The stable ID for a cluster signature. Deterministic across processes
    and recomputes: the same signature always yields the same ID."""
    digest = hashlib.sha256((signature or "").encode("utf-8")).hexdigest()
    return f"{PROPOSAL_ID_PREFIX}{digest[:_ID_HEX_LEN]}"


def stamp_proposal_ids(finding: dict[str, Any]) -> dict[str, Any]:
    """Return a COPY of a serialised finding whose clusters each carry a
    ``proposal_id`` and, when they cannot be applied, an
    ``advise_only_reason``. Pure: the input dict (and its cluster dicts) is
    never mutated. A non-dict cluster is passed through untouched rather than
    raising; this runs inside the detector's cache write, which must never
    fail on odd input.

    Both fields are derived, not detected: stamping them here (on write, and
    again on read) means a cache written before either existed still resolves
    without a recompute.
    """
    clusters = finding.get("clusters")
    if not isinstance(clusters, list):
        return dict(finding)
    stamped = [
        {
            **c,
            "proposal_id": proposal_id_for(str(c.get("signature") or "")),
            "advise_only_reason": advise_only_reason(c),
            # Derived on read for the same reason as above, and deliberately
            # NOT stored on the dataclass: it is a pure function of fields
            # already on the cluster, so a stored copy could only ever
            # disagree with them. A cache written before this existed resolves
            # correctly on the first read, with no recompute.
            "advise_snippet_offered": advise_snippet_offered(c),
        }
        if isinstance(c, dict) else c
        for c in clusters
    ]
    return {**finding, "clusters": stamped}


def list_cost_proposals(
    config: TjConfig | None = None, *, path: Path | None = None,
    persona: str | None = None,
) -> list[dict[str, Any]]:
    """Every stored COST proposal from the last completed optimize pass, each
    with its ``proposal_id``.

    Cost proposals live in the same cache file under their own key and are
    addressable by ID on the same terms as a relearn cluster: the model-routing
    apply kinds are cost cards, so an apply that names one has to be able to
    resolve it from the store.

    ``persona`` returns that persona's own scoped list instead of the
    whole-corpus one. Use :func:`cost_proposals_scoped_to_persona` when the
    caller has to be able to tell "this persona's list, which is empty" from
    "this ledger cannot answer for that persona" — this function collapses the
    second case to ``[]``, which is only safe for callers that resolve a
    proposal BY ID (an apply path), never for one that publishes a total.
    """
    rows, _resolved = cost_proposals_scoped_to_persona(
        config, path=path, persona=persona,
    )
    return rows


def cost_proposals_scoped_to_persona(
    config: TjConfig | None = None, *, path: Path | None = None,
    persona: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """``(proposals, resolved)`` for ``persona``.

    ``resolved`` is the whole point. A ledger written before per-persona
    proposals existed — or by a lone refresh that had no per-persona reports to
    adapt — holds ONE whole-corpus list and cannot answer for a persona. Its
    figures are not that persona's money, and serving them under that persona's
    label is precisely the defect this parameter exists to fix. So the caller is
    told, and a surface that publishes a figure must render NOT-YET-KNOWN rather
    than a number (root anti-pattern 22: "not yet known" and "known and empty"
    are different states and zero is the worst placeholder for the first).

    ``resolved`` is ``True`` whenever no narrowing was asked for: the
    whole-corpus list genuinely IS the answer to "everything".
    """
    from tokenjam.core.optimize import relearn_store
    from tokenjam.core.optimize.cost_proposals import (
        backfill_legacy_past_overspend_fields,
    )
    from tokenjam.core.persona_scope import persona_scopes_population

    block = relearn_store.read_cost_proposals(path, config=config)
    if not isinstance(block, dict):
        return [], True

    resolved = True
    raw = block.get("cost_proposals") or []
    if persona_scopes_population(persona):
        by_persona = block.get("cost_proposals_by_persona") or {}
        if block.get("cost_persona_scoped") and persona in by_persona:
            raw = by_persona.get(persona) or []
        else:
            # NOT the whole-corpus list as a fallback. Falling back is how a
            # mixed-corpus total came to be published under a persona label in
            # the first place; an empty list plus `resolved=False` is the only
            # honest answer, and the caller renders it as unknown.
            raw, resolved = [], False

    return [
        backfill_legacy_past_overspend_fields(
            {**pr, "proposal_id": proposal_id_for(str(pr.get("signature") or ""))}
        )
        for pr in raw if isinstance(pr, dict)
    ], resolved


def list_proposals(
    config: TjConfig | None = None, *, path: Path | None = None,
) -> list[dict[str, Any]]:
    """Every stored proposal from the last completed detector pass, each with
    its ``proposal_id``. Empty list when no pass has ever completed.

    IDs are stamped defensively on read as well as on write, so a cache file
    written by an older build still resolves by ID without a recompute.
    """
    from tokenjam.core.optimize import relearn_store

    cached = relearn_store.read_cache(path, config=config)
    if not isinstance(cached, dict):
        return []
    finding = cached.get("finding")
    if not isinstance(finding, dict):
        return []
    clusters = stamp_proposal_ids(finding).get("clusters") or []
    return [c for c in clusters if isinstance(c, dict)]


def get_proposal(
    proposal_id: str, *, config: TjConfig | None = None, path: Path | None = None,
) -> dict[str, Any] | None:
    """The stored proposal with this ID, or ``None`` when no proposal by that
    name was ever produced by the detector."""
    if not proposal_id:
        return None
    for proposal in list_proposals(config, path=path):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    for proposal in list_cost_proposals(config, path=path):
        if proposal.get("proposal_id") == proposal_id:
            return proposal
    return None


def relearn_cluster_from(proposal: dict[str, Any]) -> Any:
    """Rebuild the ``RelearnCluster`` dataclass a stored proposal was serialised
    from, for the readers that take the dataclass rather than the dict (the
    eval-case artifact builder).

    A stored proposal carries display-only keys the dataclass has no field for
    (``proposal_id``, the model-routing keys), so unknown keys are dropped
    rather than passed through, and the nested examples are rebuilt too.
    """
    from dataclasses import fields

    from tokenjam.core.optimize.analyzers.relearn import RelearnCluster, RelearnExample

    known = {f.name for f in fields(RelearnCluster)}
    kwargs = {k: v for k, v in proposal.items() if k in known}
    example_known = {f.name for f in fields(RelearnExample)}
    kwargs["examples"] = [
        RelearnExample(**{k: v for k, v in ex.items() if k in example_known})
        for ex in (proposal.get("examples") or []) if isinstance(ex, dict)
    ]
    return RelearnCluster(**kwargs)


def cluster_for_apply(proposal: dict[str, Any]) -> dict[str, Any]:
    """The apply-relevant subset of a stored proposal, in the cluster shape
    ``relearn_apply.apply_relearn_fix`` expects. Shared by the API route and
    the CLI so the two can never diverge on what a proposal ID means."""
    return {k: proposal[k] for k in APPLY_CLUSTER_FIELDS if k in proposal}
