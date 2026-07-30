"""Branded welcome banner for tj.

Hand-built ASCII wordmark — deliberately no figlet / pyfiglet / third-party
banner code or font licensing. Shown at the top of `tj onboard` and bare `tj`
so the first thing a new user sees is a branded moment (wordmark + version +
one-line value prop), not a bare budget prompt (#240).
"""
from __future__ import annotations

from tokenjam import __version__
from tokenjam.utils.formatting import console

# In-house wordmark. Kept narrow (< 50 cols) and a few rows tall so it reads as
# a branded moment without dominating the terminal. Every glyph is hand-placed;
# do not regenerate from a figlet font (licensing — #240).
_WORDMARK = r"""
 _____    _              _
|_   _|__| |_____ _ _   | |__ _ _ __
  | |/ _ \ / / -_) ' \  | / _` | '  \
  |_|\___/_\_\___|_||_|_/ \__,_|_|_|_|
                     |__/
"""

# One-line value prop. Must stay honest (Critical Rule 14) — describes what tj
# is, never promises a specific saving. Trimmed to match the landing page (#643).
_TAGLINE = "token efficiency for AI agents"


def print_welcome_banner() -> None:
    """Print the centered orange wordmark + ``TokenJam vX.Y.Z`` + value prop.

    Shared by the bare-``tj`` home screen and every ``tj onboard`` path so the
    branded moment is rendered in exactly one place (#643): centered, the
    wordmark in brand orange, the tagline trimmed to one line.
    """
    from rich.align import Align
    from rich.text import Text

    # Align the whole banner block on the wordmark's own width, not the
    # terminal's, so the version/tagline sit under the logo rather than being
    # centered independently across a wide terminal.
    wordmark = Text(_WORDMARK.strip("\n"), style="brand")
    console.print(Align.center(wordmark))
    console.print(Align.center(
        Text.from_markup(f"[label]TokenJam[/label] [muted]v{__version__}[/muted]")
    ))
    console.print(Align.center(Text(_TAGLINE, style="muted")))
    console.print()
