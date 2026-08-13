"""Repeat-task clustering and the before/after codification delta.

This module answers one question and refuses to answer it when it cannot:
**did sessions doing the same repeated work get cheaper after the relevant
lesson was written down?**

It exists because ``analyzers/relearn.py`` prices FAILURE EPISODES — individual
erroring tool calls — and the interesting waste is plausibly the inflated
SESSION around them: an agent that does not know a project's constraints
explores, backtracks, and re-sends a growing context on every turn of that
flailing. Pricing the inflated session requires deciding that two sessions were
"the same work", and a sloppy similarity measure produces confident nonsense,
which is worse than a small honest number.

So this module ships the similarity methods **with their measured precision**
and a gate that will not price a cluster whose method scores below the bar.
Nothing here is registered as an analyzer and nothing here emits a dollar
figure into a report: on the corpus it was validated against, the gate does not
open. See ``docs/internal/repeat-task-codification-measurement.md`` for the
measurement and the numbers quoted below.

Three primitives, in decreasing order of trustworthiness:

``task_cluster_key``
    Exact match on a normalized TASK STATEMENT (the session's first user
    prompt, with ids/paths/numbers masked), scoped to a project. Machine-issued
    prompts are templates, so this is close to an identity test rather than a
    similarity test. Requires a transcript, and transcripts are rotated by the
    agent harness — measured availability on the validation corpus was 99.9%
    at <30 days, 3.6% at 30-60 days, 0% beyond.

``tool_shape_signature`` / ``tool_profile_cosine``
    Span-only shape similarity: the opening tool sequence and the tool-mix
    profile. This is the ONLY signal that survives transcript rotation, and it
    is the same primitive ``analyzers/workflow_restructure.py`` (the ``script``
    analyzer) clusters on. Measured against the task-statement ground truth it
    reaches precision 0.28 unscoped and 0.54 project-scoped — i.e. roughly half
    the pairs it calls "the same work" are not. It is a legitimate SHAPE
    clusterer (which is all ``script`` claims) and an unusable SAME-WORK
    clusterer, which is why ``may_price`` rejects it.

``measure_codification_delta``
    The before/after test itself, with a bootstrap CI on the ratio of medians.
    Returns ``NULL`` whenever the interval spans 1.0 — a wide interval around a
    big point estimate is not a finding.

Governing principle: waste is only what could have been avoided. A measured
before/after delta IS avoidable spend by construction, because the "after" runs
prove the cheaper path exists. That is why the bar for CALLING something a
delta has to be this high — the claim it licenses is unusually strong.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# --- Task-statement normalization --------------------------------------------

#: How much of the normalized task statement forms the cluster key. Long enough
#: that two different templates cannot collide on a shared preamble, short
#: enough that the per-invocation tail (the ticket body, the file list) does not
#: split one template into N singletons.
TASK_KEY_CHARS = 160

_WS_RE = re.compile(r"\s+")
_HEX_RE = re.compile(r"\b[0-9a-f]{7,}\b")
_NUM_RE = re.compile(r"\d+")
_PATH_RE = re.compile(r"/[\w./~-]{6,}")


def normalize_task_statement(prompt: str) -> str:
    """Mask the per-invocation variables out of a session's first user prompt.

    Ids, paths and numbers are what differ between two runs of the same
    templated task, so they are exactly what must not reach the cluster key.
    Case is folded and whitespace collapsed because the same template reaches
    the transcript with different wrapping depending on how it was invoked.
    """
    text = _WS_RE.sub(" ", prompt or "").strip().lower()
    text = _HEX_RE.sub("<hex>", text)
    text = _NUM_RE.sub("<n>", text)
    return _PATH_RE.sub("<path>", text)


def hash_task_statement(prompt: str | None) -> str | None:
    """One-way fingerprint of a session's first user prompt, safe to persist
    on the `sessions` row (`SessionRecord.task_statement_hash`) unlike the raw
    prompt or even `normalize_task_statement`'s masked-but-still-readable
    output — the masking above leaves enough of the sentence intact to be
    legible (only ids/paths/numbers are replaced), so it is not itself safe
    for a column every read path can see.

    Hashes the NORMALIZED statement, not the raw one: two invocations of the
    same templated task should collide (the entire point of clustering),
    which only holds if the ids/paths/numbers are masked out FIRST. Not
    project-scoped — callers needing `task_cluster_key`'s project-scoping
    combine this with the session's own already-stored project field rather
    than duplicating that logic into the hash.

    Returns None for an empty/missing prompt so "no prompt captured" and "hash
    of an empty string" are never confused.
    """
    normalized = normalize_task_statement(prompt or "")
    if not normalized:
        return None
    import hashlib
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


#: A per-ticket / per-task worktree is the same project as its parent repo, and
#: splitting on it would shatter every cluster the harness generates.
_WORKTREE_SUFFIX_RE = re.compile(r"(-wt|\.wt)?-ticket-\d+$")


def project_key(project: str) -> str:
    """Collapse a project's per-task worktrees onto the parent project."""
    return _WORKTREE_SUFFIX_RE.sub("", (project or "").lower())


def task_cluster_key(prompt: str, project: str) -> str:
    """The high-confidence cluster key: normalized task statement x project.

    Scoping to the project is not cosmetic. The same template run against two
    different repos is not the same work — the codification that would help one
    has no bearing on the other — and leaving it unscoped was measured to
    roughly halve precision.
    """
    return f"{project_key(project)}::{normalize_task_statement(prompt)[:TASK_KEY_CHARS]}"


# --- Span-only shape similarity (retained, but not trusted to price) ---------

#: Length of the opening tool sequence used as a shape signature. Measured
#: precision is flat-to-falling past this, while recall keeps dropping.
OPENING_TOOLS = 8


def tool_shape_signature(tool_names: Sequence[str], k: int = OPENING_TOOLS) -> tuple[str, ...]:
    """The session's opening ``k`` tool calls, in order."""
    return tuple(tool_names[:k])


def tool_profile(tool_names: Iterable[str]) -> dict[str, float]:
    """L1-normalized tool-mix profile for a session."""
    counts: dict[str, float] = {}
    for name in tool_names:
        counts[name] = counts.get(name, 0.0) + 1.0
    total = sum(counts.values())
    if not total:
        return {}
    return {name: n / total for name, n in counts.items()}


def tool_profile_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosine similarity between two tool-mix profiles."""
    if not a or not b:
        return 0.0
    num = sum(weight * b.get(name, 0.0) for name, weight in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if not na or not nb:
        return 0.0
    return num / (na * nb)


# --- Similarity methods, carrying their own measured precision ---------------


@dataclass(frozen=True)
class SimilarityMethod:
    """A same-work similarity method and how well it was measured to work.

    ``precision`` is the measured fraction of session PAIRS the method calls
    "the same work" that really are, scored against a task-statement ground
    truth. It is not a guess and not a tunable — changing it means re-running
    the measurement, not editing the constant.
    """

    name: str
    precision: float
    recall: float
    basis: str

    @property
    def may_price(self) -> bool:
        return self.precision >= MIN_PRECISION_TO_PRICE


#: Precision a similarity method must reach before a cluster derived from it may
#: carry a dollar figure. Set at 0.90 because the delta claim is causal ("this
#: work got cheaper because the lesson was written down"): every mis-clustered
#: pair does not merely add noise, it moves the estimate in an unknown
#: direction, and there is no way for a reader to discount it.
MIN_PRECISION_TO_PRICE = 0.90

#: Exact match on a normalized, project-scoped task statement. Machine-issued
#: prompts are templates, so a match is near-identity rather than similarity;
#: the residual error is a human writing a near-identical prompt for genuinely
#: different work.
TASK_STATEMENT_MATCH = SimilarityMethod(
    name="task-statement-exact",
    precision=1.00,
    recall=1.00,
    basis=(
        "exact match on the first user prompt with ids/paths/numbers masked, "
        "scoped to the project; ground truth by construction"
    ),
)

#: Span-only shape similarity. Retained so the boundary with the ``script``
#: analyzer is explicit and so the number is not re-derived by hand later.
TOOL_SHAPE_MATCH = SimilarityMethod(
    name="tool-shape",
    precision=0.54,
    recall=0.04,
    basis=(
        "identical opening-8 tool sequence + tool-mix cosine >= 0.95, scoped to "
        "the project; scored against task-statement ground truth over 972 "
        "sessions / 200k sampled pairs"
    ),
)


# --- The before/after test ---------------------------------------------------

#: Minimum sessions on EACH side of a codification event. Below this the
#: bootstrap interval is wider than any effect worth reporting.
MIN_SESSIONS_PER_SIDE = 12

#: Bootstrap resamples for the ratio-of-medians interval.
BOOTSTRAP_RESAMPLES = 4000

#: Fixed seed: the same corpus must yield the same interval on every run, or the
#: verdict is not reproducible and cannot be cited.
BOOTSTRAP_SEED = 3


@dataclass(frozen=True)
class CodificationDelta:
    """The measured cost change across a codification event, or a refusal."""

    verdict: str  # "cheaper" | "dearer" | "null" | "refused"
    n_before: int
    n_after: int
    median_before: float
    median_after: float
    ratio: float | None
    ci_low: float | None
    ci_high: float | None
    basis: str

    @property
    def is_priceable(self) -> bool:
        """Only a ``cheaper`` verdict licenses an avoidable-spend figure."""
        return self.verdict == "cheaper"

    @property
    def avoidable_usd(self) -> float:
        """Avoidable spend implied by the delta, or 0.0 when not priceable.

        The conservative end of the interval is used, never the point estimate:
        the "after" runs prove a path at ``ci_high x`` the cost exists, so only
        that much of the "before" spend is demonstrably avoidable.
        """
        if not self.is_priceable or self.ci_high is None:
            return 0.0
        return max(0.0, self.median_before * (1.0 - self.ci_high) * self.n_before)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _bootstrap_ratio_ci(
    before: Sequence[float],
    after: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        mb = _median([before[rng.randrange(len(before))] for _ in before])
        ma = _median([after[rng.randrange(len(after))] for _ in after])
        if mb:
            ratios.append(ma / mb)
    if not ratios:
        return (float("nan"), float("nan"))
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[min(len(ratios) - 1, int(0.975 * len(ratios)))]
    return (lo, hi)


def measure_codification_delta(
    before_costs: Sequence[float],
    after_costs: Sequence[float],
    *,
    method: SimilarityMethod,
    min_per_side: int = MIN_SESSIONS_PER_SIDE,
) -> CodificationDelta:
    """Compare per-session cost across a codification event.

    Refuses — rather than guesses — in the two cases that produce confident
    nonsense: a similarity method whose measured precision is below the bar, and
    too few sessions on either side to bound the estimate. Returns ``null``
    when the interval spans 1.0, because a ratio of 0.4 with an interval of
    [0.2, 1.6] is not evidence that anything got cheaper.
    """
    if not method.may_price:
        return CodificationDelta(
            verdict="refused",
            n_before=len(before_costs),
            n_after=len(after_costs),
            median_before=_median(before_costs),
            median_after=_median(after_costs),
            ratio=None,
            ci_low=None,
            ci_high=None,
            basis=(
                f"refused: similarity method {method.name!r} has measured precision "
                f"{method.precision:.2f}, below the {MIN_PRECISION_TO_PRICE:.2f} bar "
                f"required to price a cluster ({method.basis})"
            ),
        )
    if len(before_costs) < min_per_side or len(after_costs) < min_per_side:
        return CodificationDelta(
            verdict="refused",
            n_before=len(before_costs),
            n_after=len(after_costs),
            median_before=_median(before_costs),
            median_after=_median(after_costs),
            ratio=None,
            ci_low=None,
            ci_high=None,
            basis=(
                f"refused: {len(before_costs)} sessions before / {len(after_costs)} after, "
                f"below the {min_per_side}-per-side minimum"
            ),
        )

    mb = _median(before_costs)
    ma = _median(after_costs)
    ratio = (ma / mb) if mb else None
    lo, hi = _bootstrap_ratio_ci(before_costs, after_costs)

    if ratio is None or lo != lo or hi != hi:  # NaN guard
        verdict = "null"
    elif hi < 1.0:
        verdict = "cheaper"
    elif lo > 1.0:
        verdict = "dearer"
    else:
        verdict = "null"

    return CodificationDelta(
        verdict=verdict,
        n_before=len(before_costs),
        n_after=len(after_costs),
        median_before=mb,
        median_after=ma,
        ratio=ratio,
        ci_low=lo,
        ci_high=hi,
        basis=(
            f"clustered by {method.name} (precision {method.precision:.2f}); "
            f"median session cost {mb:.2f} -> {ma:.2f} across the codification event, "
            f"ratio {ratio:.2f} (95% bootstrap CI [{lo:.2f}, {hi:.2f}], "
            f"n={len(before_costs)}/{len(after_costs)})"
            if ratio is not None
            else "no usable ratio: median cost before the event was zero"
        ),
    )


# --- Known confounds ---------------------------------------------------------

#: A per-session cost comparison across time is only meaningful if the model mix
#: held. On the validation corpus, every apparently-significant delta was fully
#: explained by the harness's model-routing changing under the comparison, in
#: BOTH directions. Any caller comparing costs across a date must hold this
#: constant first.
MODEL_MIX_DOMINANCE = 0.90


def model_mix_is_stable(
    before_shares: Sequence[float],
    after_shares: Sequence[float],
    *,
    dominance: float = MODEL_MIX_DOMINANCE,
) -> bool:
    """True when one model dominates BOTH sides of the comparison.

    ``*_shares`` are per-session shares of LLM calls served by the dominant
    model. This is a necessary, not sufficient, control: it removes the confound
    that was measured to explain the whole signal, not every confound there is.
    """
    if not before_shares or not after_shares:
        return False
    return _median(before_shares) >= dominance and _median(after_shares) >= dominance
