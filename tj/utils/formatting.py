from rich.console import Console
from rich.table import Table
from rich import box

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
    if usd < 0.001:
        return f"${usd:.6f}"
    return f"${usd:.4f}"


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def make_table(*headers: str, box_style=box.SIMPLE) -> Table:
    """Create a pre-styled Rich table."""
    t = Table(box=box_style, show_header=True, header_style="bold dim")
    for h in headers:
        t.add_column(h)
    return t
