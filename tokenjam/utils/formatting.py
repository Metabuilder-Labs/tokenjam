from rich.console import Console
from rich.table import Table
from rich import box

# Re-exported so Rich-context callers keep importing from one place; the bodies
# live in the stdlib-only `humanize` module the zero-token statusline depends on.
from tokenjam.utils.humanize import display_path, format_tokens  # noqa: F401

console = Console()
err_console = Console(stderr=True)


def severity_colour(severity: str) -> str:
    return {"critical": "red", "warning": "yellow", "info": "blue"}.get(severity, "white")


def status_icon(status: str) -> str:
    return {
        "ok": "\u2713",
        "error": "\u2717",
        "active": "\u25cf",
        "idle": "\u25cb",
        "completed": "\u25cf",
        "stale": "\u25cb",
    }.get(status, "?")


def format_cost(usd: float) -> str:
    """Format a USD amount for display (#96).

    Under $100: 2 decimal places ("$12.50"). At or above $100: whole dollars
    with thousands separators ("$1,042") — sub-dollar precision on large
    figures (e.g. "$29488.0100") reads as false precision, not accuracy.
    """
    if usd >= 100:
        return f"${usd:,.0f}"
    return f"${usd:,.2f}"


def make_table(*headers: str, box_style=box.SIMPLE) -> Table:
    """Create a pre-styled Rich table."""
    t = Table(box=box_style, show_header=True, header_style="bold dim")
    for h in headers:
        t.add_column(h)
    return t
