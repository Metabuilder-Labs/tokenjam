from __future__ import annotations

import click

from tokenjam.cli.cmd_onboard import _derive_project_name


@click.command("otel-resource-attrs")
@click.pass_context
def cmd_otel_resource_attrs(ctx: click.Context) -> None:
    """Print this project's OTel resource attributes on a single line.

    Emits e.g. ``service.name=claude-code-myrepo,service.namespace=myproject``.

    The per-terminal ``claude`` shell wrapper installed by
    ``tj onboard --claude-code`` shells out to this command and appends a
    ``service.instance.id=<terminal>`` so concurrent terminals show as
    distinct tiles on the dashboard while keeping the project's
    ``service.name`` / ``service.namespace``.

    Output goes to stdout as a single bare line with no markup so it is safe
    to embed in ``OTEL_RESOURCE_ATTRIBUTES="$(tj otel-resource-attrs),..."``.
    """
    # Same derivation tj onboard uses: git remote repo name, else cwd dir name.
    # So agent id == service.name == claude-code-<repo>.
    agent_id = f"claude-code-{_derive_project_name()}"
    attrs = f"service.name={agent_id}"

    # Append service.namespace only when this repo's agent has a project set in
    # config (written by `tj onboard --claude-code`). No config / no project =>
    # service.name only.
    config = ctx.obj["config"]
    agent_cfg = config.agents.get(agent_id)
    if agent_cfg is not None and agent_cfg.project:
        attrs += f",service.namespace={agent_cfg.project}"

    click.echo(attrs)
