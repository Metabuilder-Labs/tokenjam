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
# is, never promises a specific saving.
_TAGLINE = "cost-optimization for AI agents · local-first, OTel-native · no signup"


def print_welcome_banner() -> None:
    """Print the wordmark + ``TokenJam vX.Y.Z`` + one-line value prop."""
    console.print(f"[bold cyan]{_WORDMARK.strip(chr(10))}[/bold cyan]")
    console.print(
        f"  [bold]TokenJam[/bold] [dim]v{__version__}[/dim]"
    )
    console.print(f"  [dim]{_TAGLINE}[/dim]")
    console.print()
