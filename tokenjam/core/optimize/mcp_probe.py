"""Measure what one MCP server's tool schemas actually cost to inject.

The number this replaces was an assumption, and its own docstring said so. A
flat 25,000 tokens was charged to EVERY configured server, every call, whatever
that server exposed — while the single in-repo source for "~25K" described the
tax for ALL of a session's attached servers COMBINED. So the headline dollar
figure was linear in a constant nobody had measured, multiplied by the number of
servers the user happened to have configured. A user with six servers was shown
roughly six times the tax the cited source described for their whole session.

This module answers the question directly instead: start the server, ask it what
tools it has, serialize those schemas the way they are injected, and count that.

**What "start it" means, and its limits.** For a stdio server this is a real
subprocess on the user's machine. Three properties are load-bearing:

* **Read-only AT THE PROTOCOL LAYER, which is not the same as harmless.**
  Exactly two requests are ever sent — ``initialize`` and ``tools/list`` — plus
  the ``notifications/initialized`` the protocol requires between them.
  ``tools/call``, ``resources/read`` and ``prompts/get`` are never issued, so no
  REQUEST from this module asks a server to act on the world.

  Starting the process is a different matter and this module cannot make any
  promise about it. The command comes from the user's own config and does
  whatever it does on startup: an ``npx -y`` spec fetches from the network, a
  server may open connections, refresh an OAuth token, write a lock file or a
  cache. That is inherent to "start it and ask what tools it has" — there is no
  way to learn a server's real schema size without running it — so the honest
  statement is that the PROTOCOL surface is read-only and process startup is the
  user's own command, gated behind an off-by-default setting rather than
  described as safe.
* **Bounded and terminated, including the tree.** Every server gets one time
  budget covering spawn, handshake and listing. stdin closes, then the child's
  whole PROCESS GROUP is signalled, then killed if it does not go — the group
  rather than the child because an ``npx -y <pkg>`` spec makes npx the child and
  the real server its grandchild, so signalling the child alone leaves the
  server running. Children are also registered for an ``atexit`` sweep, so an
  interpreter killed between spawn and terminate (an MCP client timing out the
  tool call that triggered the pass is the realistic case) does not orphan the
  user's servers.
* **Never a fabricated answer.** A server that will not start, does not respond,
  or answers something unparseable returns a measurement with ``tokens = None``
  and a status saying which of those happened. There is no default to fall back
  to, because a default is exactly the defect this module exists to remove.

A remote (``http``/``sse``) server is deliberately NOT probed. There is nothing
local to start; measuring one would mean issuing an authenticated request to a
third party as a side effect of running an analyzer. It is reported
``unsupported`` — an honest gap, not a guess.

**Why the measurement is cached.** Starting a server is expensive and visible.
The measurement is stored on the server's ingested config record
(``core/agent_config``) against its spec hash, so an unchanged server is measured
once and re-read thereafter; changing its command, args or env invalidates it.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

from tokenjam.core import agent_config as ac
from tokenjam.utils.time_parse import utcnow

log = logging.getLogger(__name__)

#: Total wall-clock budget for ONE server: spawn, handshake and listing. A
#: server that needs longer than this to say what tools it has is one the agent
#: would also be waiting on, and the analyzer must not hang on it either.
PROBE_TIMEOUT_SECONDS = 20.0

#: Ceiling on how many servers a single analysis pass will start. The cache
#: means a steady-state run starts none; this bounds the first run, and the run
#: after someone adds a dozen servers at once.
MAX_SERVERS_PER_PASS = 24

#: How long a FAILED measurement is believed before the server is tried again. A
#: server can be transiently down, so an ``unreachable`` verdict is not permanent
#: the way a successful one is — but retrying on every pass would mean spawning
#: a failing process every time the analyzer runs.
UNREACHABLE_RETRY_AFTER = timedelta(hours=24)

#: The MCP protocol revision announced in the handshake. Servers negotiate down;
#: this is the version whose ``tools/list`` shape the serializer below assumes.
PROTOCOL_VERSION = "2024-11-05"

#: Prefix Claude Code gives an MCP tool in the ``tools`` array it sends. The
#: measured schema must carry the name the model actually receives, because the
#: name is part of what is serialized and a server with many long tool names
#: pays for them.
TOOL_NAME_PREFIX = "mcp__{server}__{tool}"


@dataclass(frozen=True)
class SchemaMeasurement:
    """What one server's schema injection was measured to cost.

    ``tokens is None`` is the only honest answer for a server that could not be
    measured, and every consumer must treat it as "excluded", never as zero. A
    zero would read as "this server is free", which is a stronger claim than any
    failed probe supports.
    """

    server: str
    #: Serialized size of the FULL tool-schema array, in tokens. ``None`` when
    #: unmeasured.
    tokens: int | None = None
    #: Serialized size of the deferred/ToolSearch-style listing — name and
    #: description per tool, no input schema. Measured off the same response, so
    #: the deferred lane stops being an assumption too.
    deferred_tokens: int | None = None
    tool_count: int = 0
    status: str = ac.MEASURE_SKIPPED
    #: Why, in words, when the measurement did not happen. Empty on success.
    detail: str = ""
    measured_at: datetime | None = None
    #: True when this came out of the store rather than from a fresh probe.
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status == ac.MEASURE_OK and self.tokens is not None


# --- Serialization ----------------------------------------------------------

def serialize_tool_schemas(server: str, tools: Sequence[Mapping[str, Any]]) -> str:
    """The tool array as it is injected, for ``server``'s tools.

    Shaped as the provider tool-definition array — namespaced name, description,
    input schema — because that is what rides in the request, and the name
    prefix and the full JSON Schema are a real part of what a server costs. A
    measurement of the raw ``tools/list`` payload instead would miss the prefix
    and include protocol scaffolding the model never sees.
    """
    payload = [
        {
            "name": TOOL_NAME_PREFIX.format(server=server, tool=str(tool.get("name") or "")),
            "description": str(tool.get("description") or ""),
            "input_schema": tool.get("inputSchema") or tool.get("input_schema") or {},
        }
        for tool in tools
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def serialize_deferred_listing(server: str, tools: Sequence[Mapping[str, Any]]) -> str:
    """The same tools as a DEFERRED listing: one name+description line each.

    What a ToolSearch-style listing actually puts in context when a server's
    schemas are not loaded. Measured off the same response as the full schema so
    the two lanes are on one basis rather than one measured and one assumed.
    """
    return "\n".join(
        f"{TOOL_NAME_PREFIX.format(server=server, tool=str(tool.get('name') or ''))}: "
        f"{str(tool.get('description') or '').strip()}"
        for tool in tools
    )


# --- The stdio probe --------------------------------------------------------

def _spec_transport(spec: Mapping[str, Any]) -> str:
    declared = str(spec.get("type") or spec.get("transport") or "").strip().lower()
    if declared:
        return declared
    if spec.get("command"):
        return "stdio"
    if spec.get("url"):
        return "http"
    return ""


def _child_env(spec: Mapping[str, Any]) -> dict[str, str]:
    """The probe child's environment: the current one plus the server's own.

    The parent environment is included because a server's command is routinely a
    ``npx``/``uvx`` shim that needs ``PATH`` and a home directory to resolve at
    all — stripping it would turn every working server into an ``unreachable``
    one, which is a fabricated finding in the other direction.
    """
    env = dict(os.environ)
    declared = spec.get("env")
    if isinstance(declared, dict):
        for key, value in declared.items():
            env[str(key)] = os.path.expandvars(str(value))
    return env


def _rpc(stream: Any, message: dict[str, Any]) -> None:
    stream.write(json.dumps(message) + "\n")
    stream.flush()


#: Sentinel the reader thread pushes when the child's stdout reaches EOF.
_EOF = object()


def _start_reader(proc: subprocess.Popen) -> "queue.Queue[Any]":
    """Drain the child's stdout on a daemon thread.

    THE reason this is not a plain ``readline()`` loop: ``readline`` blocks, so
    a deadline checked between reads is not a deadline at all — a server that
    accepts the handshake and then says nothing would hold the analyzer for as
    long as it felt like living, and the probe's whole time budget would be
    decorative. A test that starts a process sleeping for a minute is what
    surfaced it, and it is pinned by
    ``test_a_process_that_says_nothing_times_out_without_hanging``.

    Daemon, so a thread still parked on a read of a process that outlived its
    termination can never keep the interpreter alive.
    """
    lines: "queue.Queue[Any]" = queue.Queue()

    def _pump() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(_EOF)

    threading.Thread(target=_pump, daemon=True).start()
    return lines


def _read_response(
    lines: "queue.Queue[Any]", wanted_id: int, deadline: float,
) -> dict | None:
    """The response carrying ``wanted_id``, or None on timeout/EOF.

    Notifications and log messages share the stream, so anything without the
    wanted id is skipped rather than treated as the answer. A malformed line is
    skipped for the same reason: a server that logs to stdout is misbehaving but
    is not necessarily unmeasurable.
    """
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            item = lines.get(timeout=remaining)
        except queue.Empty:
            return None
        if item is _EOF:
            return None
        try:
            message = json.loads(item)
        except ValueError:
            continue
        if isinstance(message, dict) and message.get("id") == wanted_id:
            return message


#: POSIX gives a probe child its own process group so the whole tree can be
#: signalled; Windows has no equivalent here, so the group calls degrade to
#: signalling the direct child.
_CAN_GROUP_SIGNAL = hasattr(os, "killpg") and hasattr(os, "getpgid")

#: Probe children that have not been reaped yet. An interpreter dying between
#: spawn and terminate — the MCP client timing out the tool call that triggered
#: the pass is the realistic case — would otherwise leave the user's MCP servers
#: running with nothing left to shut them down.
_LIVE_CHILDREN: "set[subprocess.Popen]" = set()
_LIVE_CHILDREN_LOCK = threading.Lock()


def _register_child(proc: subprocess.Popen) -> None:
    with _LIVE_CHILDREN_LOCK:
        _LIVE_CHILDREN.add(proc)


def _forget_child(proc: subprocess.Popen) -> None:
    with _LIVE_CHILDREN_LOCK:
        _LIVE_CHILDREN.discard(proc)


def _signal_group(proc: subprocess.Popen, sig: int) -> bool:
    """Signal the child's whole process group, falling back to the child.

    A spec like ``npx -y <pkg>`` makes npx the direct child and the real server
    its grandchild, so signalling only the child leaves the server alive.
    """
    if _CAN_GROUP_SIGNAL:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            return True
        except (OSError, ProcessLookupError):
            return False
    try:
        proc.send_signal(sig)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False


def _sweep_live_children() -> None:
    """Kill anything still running at interpreter exit. Registered with atexit."""
    with _LIVE_CHILDREN_LOCK:
        stragglers = list(_LIVE_CHILDREN)
    for proc in stragglers:
        if proc.poll() is None:
            _signal_group(proc, signal.SIGKILL)


atexit.register(_sweep_live_children)


def _terminate(proc: subprocess.Popen) -> None:
    """Close the child down without leaving it behind, in escalating order.

    ORDER IS LOAD-BEARING, and getting it wrong reintroduces the hang the reader
    thread exists to prevent. ``stdout`` must NOT be closed while the pump
    thread is parked on a read of it: ``close()`` waits for that buffer's lock,
    so closing first blocks the caller for exactly as long as the silent server
    it was trying to escape. Signal the process, let its exit deliver EOF to the
    pump thread, and only then close the pipe.
    """
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    if proc.poll() is None:
        # The GROUP, not just the child — see `_signal_group`.
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            _signal_group(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, OSError):
                log.debug("MCP probe child did not exit after kill")
    try:
        if proc.stdout is not None:
            proc.stdout.close()
    except (OSError, RuntimeError, ValueError):
        pass
    _forget_child(proc)


def probe_stdio_server(
    name: str, spec: Mapping[str, Any], *, timeout: float = PROBE_TIMEOUT_SECONDS,
) -> SchemaMeasurement:
    """Start ``name``, ask for its tools, measure them, shut it down.

    Never raises: every failure mode returns a measurement whose ``tokens`` is
    ``None`` and whose ``detail`` says what went wrong, because a probe that
    raised would take the whole analyzer down over one broken server.
    """
    command = spec.get("command")
    if not command:
        return SchemaMeasurement(
            server=name, status=ac.MEASURE_UNSUPPORTED,
            detail="no launch command declared for this server.",
            measured_at=utcnow(),
        )
    args = [str(a) for a in (spec.get("args") or []) if a is not None]
    argv = [str(command), *args]
    deadline = time.monotonic() + timeout
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv comes from the user's own config
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_child_env(spec),
            cwd=str(spec.get("cwd")) if spec.get("cwd") else None,
            text=True,
            bufsize=1,
            # Own process group, so the ORPHAN case is recoverable. `_terminate`
            # signals the whole group rather than just the direct child: a spec
            # like `npx -y <pkg>` makes npx the child and the real server its
            # grandchild, so signalling the child alone leaves the server
            # running. And if this process is killed outright — an MCP client
            # timing out the tool call that triggered the pass, say — the
            # registry below is what the atexit sweep uses to take the group
            # down instead of leaving the user's servers running.
            start_new_session=_CAN_GROUP_SIGNAL,
        )
        _register_child(proc)
    except (OSError, ValueError) as exc:
        return SchemaMeasurement(
            server=name, status=ac.MEASURE_UNREACHABLE,
            detail=f"could not start `{argv[0]}` ({exc}).",
            measured_at=utcnow(),
        )

    lines = _start_reader(proc)
    try:
        _rpc(proc.stdin, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tokenjam-schema-probe", "version": "1"},
            },
        })
        if _read_response(lines, 1, deadline) is None:
            return SchemaMeasurement(
                server=name, status=ac.MEASURE_UNREACHABLE,
                detail="no response to the MCP initialize handshake.",
                measured_at=utcnow(),
            )
        _rpc(proc.stdin, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools: list[dict] = []
        cursor: str | None = None
        request_id = 2
        while True:
            params = {"cursor": cursor} if cursor else {}
            _rpc(proc.stdin, {
                "jsonrpc": "2.0", "id": request_id, "method": "tools/list",
                "params": params,
            })
            response = _read_response(lines, request_id, deadline)
            if response is None:
                return SchemaMeasurement(
                    server=name, status=ac.MEASURE_UNREACHABLE,
                    detail="no response to tools/list within the probe budget.",
                    measured_at=utcnow(),
                )
            if response.get("error"):
                return SchemaMeasurement(
                    server=name, status=ac.MEASURE_UNREACHABLE,
                    detail=f"tools/list returned an error ({response['error']}).",
                    measured_at=utcnow(),
                )
            result = response.get("result")
            if not isinstance(result, dict):
                return SchemaMeasurement(
                    server=name, status=ac.MEASURE_UNREACHABLE,
                    detail="tools/list returned no result object.",
                    measured_at=utcnow(),
                )
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(t for t in page if isinstance(t, dict))
            cursor = result.get("nextCursor") or None
            request_id += 1
            if not cursor or time.monotonic() >= deadline:
                break
        return measurement_from_tools(name, tools)
    except (OSError, ValueError, BrokenPipeError) as exc:
        return SchemaMeasurement(
            server=name, status=ac.MEASURE_UNREACHABLE,
            detail=f"the probe connection failed ({exc}).",
            measured_at=utcnow(),
        )
    finally:
        if proc is not None:
            _terminate(proc)


def measurement_from_tools(
    name: str, tools: Sequence[Mapping[str, Any]],
) -> SchemaMeasurement:
    """A measurement built from an already-obtained ``tools/list`` payload.

    Split out from the probe so the arithmetic is testable without starting a
    process, and so a caller that already holds a listing never has to start one.
    """
    full = serialize_tool_schemas(name, tools)
    deferred = serialize_deferred_listing(name, tools)
    return SchemaMeasurement(
        server=name,
        tokens=ac.tokens_for_chars(len(full)),
        deferred_tokens=ac.tokens_for_chars(len(deferred)),
        tool_count=len(tools),
        status=ac.MEASURE_OK,
        measured_at=utcnow(),
    )


def measure_server(
    name: str, spec: Mapping[str, Any], *, timeout: float = PROBE_TIMEOUT_SECONDS,
) -> SchemaMeasurement:
    """Measure ``name`` by whatever transport it declares, or say why not."""
    transport = _spec_transport(spec)
    if transport in ("", "stdio"):
        return probe_stdio_server(name, spec, timeout=timeout)
    return SchemaMeasurement(
        server=name, status=ac.MEASURE_UNSUPPORTED,
        detail=(
            f"`{transport}` transport: there is no local process to start, and "
            f"measuring it would mean issuing an authenticated request to a "
            f"third party as a side effect of an analysis run."
        ),
        measured_at=utcnow(),
    )


#: The measurer used when a caller names none. A module-level indirection on
#: purpose: it is the seam a test replaces to get deterministic sizes without
#: any subprocess, and the seam a caller replaces to supply a pre-recorded
#: measurement.
_default_measurer: Callable[[str, Mapping[str, Any]], SchemaMeasurement] = measure_server


def _cached(
    store: ac.AgentConfigStore, config_id: str, name: str, now: datetime,
) -> SchemaMeasurement | None:
    """A believable stored measurement for ``config_id``, or None.

    The store already drops a measurement whose spec hash moved, so a stored
    SUCCESS is believed indefinitely. A stored FAILURE is believed only for
    :data:`UNREACHABLE_RETRY_AFTER`, because "this server was down" is a
    statement about a moment, not about the server.
    """
    row = store.measurement_for(config_id)
    if not row.status:
        return None
    if row.status == ac.MEASURE_OK and row.tokens is not None:
        deferred = row.extra.get("deferred_tokens")
        return SchemaMeasurement(
            server=name, tokens=row.tokens,
            deferred_tokens=int(deferred) if deferred is not None else None,
            tool_count=int(row.extra.get("tool_count") or 0),
            status=row.status, measured_at=row.at, from_cache=True,
        )
    if row.at is not None and now - row.at < UNREACHABLE_RETRY_AFTER:
        return SchemaMeasurement(
            server=name, tokens=None, status=row.status, measured_at=row.at,
            from_cache=True,
            detail="a previous probe of this server did not succeed.",
        )
    return None


def resolve_schema_measurements(
    servers: Mapping[str, Any],
    *,
    store: ac.AgentConfigStore,
    enabled: bool = True,
    measurer: Callable[[str, Mapping[str, Any]], SchemaMeasurement] | None = None,
    max_servers: int = MAX_SERVERS_PER_PASS,
) -> dict[str, SchemaMeasurement]:
    """One measurement per configured server, from cache where possible.

    ``enabled=False`` takes no measurement at all and reports every server
    ``skipped``. It does NOT substitute a number: a run that chose not to measure
    has the same evidence as a run that failed to, and both must exclude the
    server from anything priced rather than quietly billing it a default.
    """
    now = utcnow()
    take = measurer if measurer is not None else _default_measurer
    out: dict[str, SchemaMeasurement] = {}
    probed = 0
    for name in sorted(servers):
        server = servers[name]
        config_id = getattr(server, "config_id", "") or ""
        cached = _cached(store, config_id, name, now) if config_id else None
        if cached is not None:
            out[name] = cached
            continue
        if not enabled:
            out[name] = SchemaMeasurement(
                server=name, status=ac.MEASURE_SKIPPED,
                detail="schema measurement is switched off for this run.",
            )
            continue
        if probed >= max_servers:
            out[name] = SchemaMeasurement(
                server=name, status=ac.MEASURE_SKIPPED,
                detail=(
                    f"this pass had already started {max_servers} server(s); "
                    f"the rest are measured on the next run."
                ),
            )
            continue
        probed += 1
        try:
            result = take(name, getattr(server, "spec", {}) or {})
        except Exception as exc:  # a broken server must not take the analyzer down
            result = SchemaMeasurement(
                server=name, status=ac.MEASURE_UNREACHABLE,
                detail=f"the probe raised ({exc}).", measured_at=now,
            )
        out[name] = result
        if config_id:
            store.record_measurement(
                config_id, tokens=result.tokens, status=result.status,
                at=result.measured_at or now,
                extra={
                    "deferred_tokens": result.deferred_tokens,
                    "tool_count": result.tool_count,
                },
            )
    return out


__all__ = [
    "MAX_SERVERS_PER_PASS",
    "PROBE_TIMEOUT_SECONDS",
    "PROTOCOL_VERSION",
    "TOOL_NAME_PREFIX",
    "UNREACHABLE_RETRY_AFTER",
    "SchemaMeasurement",
    "measure_server",
    "measurement_from_tools",
    "probe_stdio_server",
    "resolve_schema_measurements",
    "serialize_deferred_listing",
    "serialize_tool_schemas",
]
