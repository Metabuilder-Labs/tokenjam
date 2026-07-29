"""Which BUILD produced a stored analyzer result.

WHY THIS EXISTS. The analyzer stores are caches, and nothing invalidates them
on upgrade. Each carries a ``computed_at``, which answers HOW OLD — and every
surface renders it as ``computed <age> ago``, which readers take to mean WHICH
BUILD. Those are different questions, and they come apart at exactly the moment
it matters most: after upgrading tokenjam, the next pass stamps a fresh
timestamp over figures the replaced binary may have produced. The audience for
that sentence is precisely the user who upgraded to get a fix and will conclude
it did not work. A card with no timestamp at all would invite the suspicion that
saves them; a recent one earns trust it has not got.

So the producing build travels WITH the result, and a surface can qualify the
freshness claim instead of vouching for something it does not know.

ONE derivation, in one place. Two stores stamp it (``report_store`` and
``relearn_store``) and the routes publish it beside the build that is SERVING
the figures; two copies of "what version am I" are free to disagree the moment
one of them is imported differently, and a provenance label that disagrees with
itself is worse than none.
"""
from __future__ import annotations

#: What an unknown build reads as on the wire. A real value, never ``None`` —
#: a surface has to be able to distinguish "produced by a build that did not
#: stamp itself" from "the key is missing because the payload is malformed".
UNKNOWN = "unknown"


def tj_build() -> str:
    """The running build's version string. Never raises.

    A provenance label must not be able to sink the write it labels, so every
    failure mode collapses to :data:`UNKNOWN` — which a surface renders as an
    unqualified build rather than as agreement.
    """
    try:
        from tokenjam import __version__

        return str(__version__) or UNKNOWN
    except Exception:  # noqa: BLE001 - see docstring
        return UNKNOWN
