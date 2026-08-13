"""Stand-ins for the model on the summarize delivery paths.

The source is sent to the rewriter fenced in a per-call `<tj-source nonce=...>` envelope, and a
model that FOLLOWED the contract returns its rewrite inside the same envelope with every
`<tj-keep>` marker echoed. These helpers produce exactly that.

Worth having in one place: a fake that returns a six-word stub outside the envelope is a fake of a
HIJACKED model, and a happy-path test built on one silently stops testing the happy path the day
the gate learns to catch hijacks. Tests that mean to exercise a refusal build the bad output
explicitly, at the site that asserts the refusal.
"""
from __future__ import annotations

import re

from tokenjam.core.summarize import wrap

MARKER_RE = re.compile(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', re.DOTALL)
_NONCE_RE = re.compile(rf'<{wrap.SOURCE_TAG} nonce="([0-9a-f]+)">')

DEFAULT_PROSE = "Be careful; never skip a required step."


def nonce_of(sent: str) -> str | None:
    """The envelope nonce in what the rewriter was sent (None when it wasn't enveloped)."""
    m = _NONCE_RE.search(sent)
    return m.group(1) if m else None


def envelope_like(sent: str, body: str) -> str:
    """Return `body` fenced in the same envelope `sent` arrived in (unchanged if it had none).

    For fakes that build their own rewrite text but still need to answer as a compliant model.
    """
    n = nonce_of(sent)
    return wrap.envelope(body, n) if n is not None else body


def compliant_summary(sent: str, new_prose: str = DEFAULT_PROSE, *, ratio: float = 0.5) -> str:
    """What a model that obeyed the contract returns for the prompt `sent`.

    Echoes every marker, wraps the result in the same envelope, and repeats `new_prose` until it is
    about `ratio` of the source's prose. The padding is load-bearing: `session.check` refuses a
    rewrite far under its word budget, so a fixed one-line stub would read as a refusal rather than
    as a compression.
    """
    body_in = sent
    n = nonce_of(sent)
    if n is not None:
        body_in, _ = wrap.strip_envelope(sent, n)
    markers = MARKER_RE.findall(body_in)
    source_prose_words = wrap.word_count(MARKER_RE.sub("", body_in))

    want = max(1, int(ratio * source_prose_words))
    per = max(1, wrap.word_count(new_prose))
    prose = " ".join([new_prose] * max(1, -(-want // per)))    # ceil-div repeats

    out = prose + " " + " ".join(markers)
    return wrap.envelope(out, n) if n is not None else out
