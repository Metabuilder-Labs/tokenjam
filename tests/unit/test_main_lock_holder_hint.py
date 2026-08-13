"""`cli/main.py`'s DB-locked error used to name only the CONFIGURED
api.{host,port} — whatever THIS invocation's config resolves to, which can
differ from the config the process actually holding the lock booted with (a
different config file, a different port, or a daemon that has since crashed
and left the lock behind). `_lock_holder_hint` adds the ACTUAL holder, read
from `~/.local/share/tj/server.state`, when it's resolvable."""
from __future__ import annotations

from tokenjam.cli.main import _lock_holder_hint
from tokenjam.core.server_state import ServerState


def test_no_state_file_degrades_to_empty_string(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.server_state.read_server_state", lambda: None,
    )
    assert _lock_holder_hint() == ""


def test_names_the_live_holder(monkeypatch):
    state = ServerState(pid=4242, port=8787, config_path="/home/me/.config/tj/config.toml")
    monkeypatch.setattr(
        "tokenjam.core.server_state.read_server_state", lambda: state,
    )
    monkeypatch.setattr("tokenjam.core.server_state.is_pid_alive", lambda pid: True)
    monkeypatch.setattr("tokenjam.core.server_state.is_serve_process", lambda pid: True)

    hint = _lock_holder_hint()
    assert "4242" in hint
    assert "8787" in hint
    assert "/home/me/.config/tj/config.toml" in hint


def test_names_a_dead_holder_as_stale_rather_than_asserting_liveness(monkeypatch):
    """A crashed daemon can leave the DB locked with no live process to
    name — don't claim `tj serve` is running when the recorded PID is gone
    or has been recycled by something else."""
    state = ServerState(pid=9999, port=8787, config_path=None)
    monkeypatch.setattr(
        "tokenjam.core.server_state.read_server_state", lambda: state,
    )
    monkeypatch.setattr("tokenjam.core.server_state.is_pid_alive", lambda pid: False)
    monkeypatch.setattr("tokenjam.core.server_state.is_serve_process", lambda pid: False)

    hint = _lock_holder_hint()
    assert "9999" in hint
    assert "no longer a running" in hint
