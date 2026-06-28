"""Delivery — how the *CLI* produces a summary (DEC-027/029): ``claude -p`` (local, headless) or ``api``.

The MCP path never comes here (Claude rewrites in-session). `summarize_via` is the one-shot the
CLI — and the future Lens UI — both call: prep → deliver(mode) → check → stage. ToS clean by
construction: ``claude -p`` is sanctioned headless use of the user's own Claude Code (DEC-011); the
``api`` path uses the user's OWN ``TJ_ANTHROPIC_API_KEY`` (DEC-009/029) — a subscription credential
never makes a raw outbound call.
"""
from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import session
from tokenjam.core.summarize.estimate import DEFAULT_TARGET_RATIO
from tokenjam.core.summarize.session import CheckVerdict

# api delivery (DEC-029): Anthropic-only for v1, raw httpx (no heavy SDK), generous timeout.
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_API_TIMEOUT_S = 120.0          # generous — a hung socket shouldn't wedge forever (DEC-029)
_API_MAX_TOKENS = 8192          # a summary is shorter than its source; ample for prompt files
# Allowlist: only a clean completion counts as a summary. One element for now — extend deliberately
# (our request sends no tools / stop_sequences, so end_turn is the sole legitimate "done" signal).
_API_OK_STOP_REASONS = ("end_turn",)
_CLAUDE_TIMEOUT_S = 300.0       # claude -p is a full agent; generous so a stuck call can't wedge the CLI (#2)


class DeliveryError(Exception):
    """A delivery handler couldn't produce a summary (carries a house-voice message)."""


@dataclass(frozen=True)
class DeliveryResult:
    """What a delivery handler returns: the summary, plus priced rewrite usage when there is one.

    ``rewrite_usd`` / ``rewrite_tokens`` are populated only by ``api`` when provider ``usage`` is
    present. The dollar amount uses known model rates when available, otherwise TokenJam's
    default-rate estimate; ``Amortization.rates_known`` tells the user which path happened.
    ``claude`` / manual / in-session have no marginal $ here, and API responses missing usage have
    unknown cost, so both are None.
    """
    summary: str
    rewrite_usd: float | None = None
    rewrite_tokens: int | None = None


@dataclass(frozen=True)
class Amortization:
    """The "pays for itself" economics of an ``api`` rewrite (DEC-029).

    ``rewrite_usd`` is provider usage priced with known rates when ``rates_known`` is true, else
    TokenJam's default-rate estimate. ``saving_usd_per_call`` is an ESTIMATE (the ``chars/4``
    ``est_tokens_saved`` priced at the same model's input rate) and is zero when the rewrite failed
    the structure gate and was not staged. ``break_even_calls`` = ``ceil(rewrite / saving)`` — None
    when there's no staged saving to amortize.
    """
    model: str
    rewrite_usd: float
    saving_usd_per_call: float
    break_even_calls: int | None
    rates_known: bool                  # False → the $ used DEFAULT rates (model not in the pricing table)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "rewrite_usd": self.rewrite_usd,
            "saving_usd_per_call": self.saving_usd_per_call,
            "break_even_calls": self.break_even_calls,
            "rates_known": self.rates_known,
        }


@dataclass(frozen=True)
class RunResult:
    """What ``summarize_via`` hands the CLI: the verdict + (for ``api``) the amortization line.

    ``verdict`` is ``None`` when ``path`` was below the prose gate — ``skipped_note`` then carries
    the reason, captured from the single ``prepare()`` (the caller never re-preps → no TOCTOU window).
    ``cost_unknown`` is True only for an ``api`` rewrite whose response carried no usage — distinct
    from ``claude-p`` (genuinely free), so the CLI says "cost unknown" rather than showing nothing.
    """
    verdict: CheckVerdict | None
    amortization: Amortization | None = None
    skipped_note: str | None = None
    cost_unknown: bool = False


def _via_claude(wrapped_prompt: str, system_rules: str) -> DeliveryResult:
    """Run the user's local ``claude -p`` headless: combined prompt on stdin, stdout = the summary."""
    prompt = f"{system_rules}\n\n{wrapped_prompt}"
    try:
        proc = subprocess.run(["claude", "-p"], input=prompt, capture_output=True, text=True,
                              timeout=_CLAUDE_TIMEOUT_S)
    except FileNotFoundError as e:
        raise DeliveryError(
            "Claude Code isn't installed — install it, use `--via api`, or enter manual mode "
            "(`tj summarize prep` then `check`).") from e
    except subprocess.TimeoutExpired as e:
        raise DeliveryError(
            f"`claude -p` timed out after {int(_CLAUDE_TIMEOUT_S)}s — it may be blocked on auth, a "
            "permission prompt, or an update. Try manual mode or `--via api`.") from e
    if proc.returncode != 0:
        raise DeliveryError(
            f"`claude -p` failed (exit {proc.returncode}): {proc.stderr.strip() or '(no stderr)'}")
    out = proc.stdout.strip()
    if not out:
        raise DeliveryError("`claude -p` returned nothing.")
    return DeliveryResult(summary=out)


def _via_api(config: TjConfig, wrapped_prompt: str, system_rules: str) -> DeliveryResult:
    """Rewrite via the Anthropic API with the user's OWN key (DEC-029).

    Key = ``TJ_ANTHROPIC_API_KEY`` (its presence is the authorization — a global ``ANTHROPIC_API_KEY``
    is deliberately ignored). Model = the required ``[summarize] api_model`` (no default). Provider
    ``usage`` becomes the rewrite's priced cost via the shared pricing engine (known model rates
    when available, otherwise TokenJam's default-rate estimate). If usage is missing, cost stays
    unknown and the caller uses the same no-amortization fallback as ``claude -p``.
    """
    model = config.summarize.api_model
    if not model:
        raise DeliveryError(
            "`--via api` needs a model — set `[summarize] api_model` in your config (no default: only "
            "frontier models are validated to preserve structure; a weaker one just fails the check). "
            'e.g. api_model = "claude-opus-4-8".')
    key = os.environ.get("TJ_ANTHROPIC_API_KEY")
    if not key:
        raise DeliveryError(
            "`--via api` needs your key — set `TJ_ANTHROPIC_API_KEY` (this authorizes TJ to call "
            "Anthropic on your account), or use `--via claude-p` / manual mode.")

    import httpx                                 # base dep; lazy import keeps it off the non-api paths
    from tokenjam.core.cost import calculate_cost

    try:
        resp = httpx.post(
            _ANTHROPIC_URL, timeout=_API_TIMEOUT_S,
            headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION,
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": _API_MAX_TOKENS, "system": system_rules,
                  "messages": [{"role": "user", "content": wrapped_prompt}]},
        )
    except httpx.TimeoutException as e:
        raise DeliveryError(
            f"`--via api` timed out after {int(_API_TIMEOUT_S)}s calling Anthropic.") from e
    except httpx.HTTPError as e:
        raise DeliveryError(f"`--via api` request failed: {e}") from e
    if resp.status_code != 200:
        raise DeliveryError(
            f"Anthropic API returned {resp.status_code}: {resp.text[:200].strip() or '(no body)'}")

    try:
        data = resp.json()
    except ValueError as e:                         # 200 with an unparseable body
        raise DeliveryError("Anthropic API returned a 200 with an unparseable JSON body.") from e
    if not isinstance(data, dict):                  # unexpected shape — guard the .get() calls below
        raise DeliveryError(
            f"Anthropic API returned unexpected JSON (a {type(data).__name__}, not an object).")
    stop_reason = data.get("stop_reason")           # the API's own ground-truth signal (not inferred)
    if stop_reason == "max_tokens":                 # common case → a tailored, actionable message
        raise DeliveryError(
            f"Anthropic hit max_tokens ({_API_MAX_TOKENS}) — the summary was truncated; refusing to "
            "stage a partial rewrite. The prompt may be too long to summarize in one call "
            "(this api call was still billed).")
    if stop_reason not in _API_OK_STOP_REASONS:     # allowlist — refusal / pause_turn / tool_use /
        raise DeliveryError(                        # stop_sequence / absent are NOT clean summaries
            f"Anthropic didn't complete normally (stop_reason={stop_reason!r}); refusing to stage a "
            "non-summary response (this api call was still billed). Only a clean completion "
            "(end_turn) is accepted on this path.")
    summary = "".join(
        b.get("text", "") for b in data.get("content", []) if isinstance(b, dict) and b.get("type") == "text"
    ).strip()
    if not summary:
        raise DeliveryError("Anthropic API returned no text.")

    usage = data.get("usage")
    rewrite_usd: float | None = None
    rewrite_tokens: int | None = None
    if isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
        try:
            in_tok = int(usage["input_tokens"])
            out_tok = int(usage["output_tokens"])
        except (TypeError, ValueError):
            pass
        else:
            rewrite_usd = calculate_cost("anthropic", model, in_tok, out_tok)
            rewrite_tokens = in_tok + out_tok
            # DEF-009: this provider usage can later be recorded as a NormalizedSpan via
            # IngestPipeline.process() (on-demand DB open) so summarize's own spend shows in `tj cost`.
            # Deferred until the surface locks; summarize stays DB-free for now.
    return DeliveryResult(summary=summary, rewrite_usd=rewrite_usd, rewrite_tokens=rewrite_tokens)


def deliver(config: TjConfig, mode: str, wrapped_prompt: str, system_rules: str) -> DeliveryResult:
    """Produce a summary via the chosen delivery mode — ``claude-p`` or ``api``."""
    if mode == "claude-p":
        return _via_claude(wrapped_prompt, system_rules)
    if mode == "api":
        return _via_api(config, wrapped_prompt, system_rules)
    raise DeliveryError(f"unknown delivery mode {mode!r}")


def summarize_via(
    config: TjConfig, path: str, mode: str, *, ratio: float = DEFAULT_TARGET_RATIO,
    on_progress: Callable[[str], None] | None = None,
) -> RunResult:
    """One-shot for the CLI / future UI: prep → deliver(mode) → check → stage (+ amortize for ``api``).

    Returns a ``RunResult``; when ``path`` is below the prose gate it has ``verdict=None`` and
    ``skipped_note`` set (from the single ``prepare()`` — the caller never re-preps). Raises
    ``DeliveryError`` if the model call fails and ``SummarizeRefused`` if the file changed since prep
    — both *before* anything is staged. ``on_progress``, when given, is called with a short status
    string at each phase so the CLI isn't silent during the (slow) model call.
    """
    def _p(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)

    _p("Wrapping structure")
    prep = session.prepare(path=path, ratio=ratio)
    if not prep.wrapped_prompt:                 # below the worth-it prose gate
        return RunResult(verdict=None, skipped_note=prep.note)
    _p(f"Rewriting via {mode} — this can take a few seconds")
    delivered = deliver(config, mode, prep.wrapped_prompt, prep.system_rules)
    _p("Verifying structure + staging")
    verdict = session.check(config, path, delivered.summary, prep.source_sha256, produced_by=mode)

    amortization: Amortization | None = None
    if delivered.rewrite_usd is not None:
        # Priced rewrite usage ÷ Estimate saving (DEC-029): the saving is est_tokens_saved priced at
        # the same model's input rate, via the shared cost engine (consistent fallback with the
        # rewrite cost). Failed checks still cost money, but have no staged saving to amortize.
        from tokenjam.core.cost import calculate_cost
        from tokenjam.core.pricing import get_rates
        model = config.summarize.api_model
        assert model is not None  # rewrite_usd is set only on the api path, which required a model
        rates_known = get_rates("anthropic", model) is not None   # else the $ used default rates (#4)
        if verdict.structure_ok:
            saving = calculate_cost("anthropic", model, verdict.est_tokens_saved, 0)
            break_even = math.ceil(delivered.rewrite_usd / saving) if saving > 0 else None
        else:
            saving = 0.0
            break_even = None
        amortization = Amortization(
            model=model, rewrite_usd=delivered.rewrite_usd,
            saving_usd_per_call=saving, break_even_calls=break_even, rates_known=rates_known,
        )
    # api with no usage in the response → cost can't be computed; flag it so the CLI says so
    # (claude-p / manual have no marginal $ and stay silent).
    cost_unknown = mode == "api" and delivered.rewrite_usd is None
    return RunResult(verdict=verdict, amortization=amortization, cost_unknown=cost_unknown)
