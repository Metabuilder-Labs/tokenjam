"""Shared `--json` option plumbing for CLI commands.

The root `tj` group defines a global `--json` (`tj --json <cmd>`), stashed on
`ctx.obj["output_json"]`. Click params never inherit from an ancestor Context
automatically, so a subcommand that defines its own local `--json` and reads
only its own parameter silently ignores the global one — `tj --json status`
would print human text while `tj status --json` prints JSON. Every command
that supports JSON output should wire its option through the pair below
instead of a bare `@click.option("--json", ...)`, so the two spellings are
always equivalent.

Usage::

    from tokenjam.cli.json_option import json_option, resolve_output_json

    @cmd_group.command("foo")
    @json_option
    @click.pass_context
    def cmd_foo(ctx: click.Context, output_json_flag: bool) -> None:
        output_json = resolve_output_json(ctx, output_json_flag)
        ...

A command with no JSON support at all should keep using plain `click.option`
calls for its own flags (or none) — don't reach for this pair unless the
command actually renders a JSON branch.
"""
from __future__ import annotations

import click

# Reusable option decorator: `click.option(...)` builds a fresh Option
# instance per command it's applied to, so sharing this one object across
# every command file is safe and is the standard way to DRY up a repeated
# click.option() call.
json_option = click.option(
    "--json", "output_json_flag", is_flag=True,
    help="Emit machine-readable JSON.",
)


def resolve_output_json(ctx: click.Context, output_json_flag: bool) -> bool:
    """Whether JSON output was requested, locally or via the global flag.

    Either `tj --json <cmd>` or `tj <cmd> --json` enables JSON — there is no
    way to use one to force the other back off for a single invocation.
    """
    global_flag = bool(ctx.obj.get("output_json")) if ctx.obj else False
    return output_json_flag or global_flag
