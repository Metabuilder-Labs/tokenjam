import tempfile
from pathlib import Path

import pytest

from tokenjam.core.config import (
    find_config_file, load_config, _parse, _serialise, TjConfig, AgentConfig,
    BudgetConfig, DefaultsConfig, SensitiveAction, SecurityConfig, CaptureConfig,
    StorageConfig, resolve_effective_budget, validate_budget_value,
)


class TestFindConfigFile:
    def test_global_fallback_found(self, tmp_path, monkeypatch):
        """find_config_file() discovers ~/.config/tj/config.toml when no local config exists."""
        monkeypatch.chdir(tmp_path)
        global_config = tmp_path / ".config" / "tj" / "config.toml"
        global_config.parent.mkdir(parents=True)
        global_config.write_bytes(b'version = "1"\n\n[security]\ningest_secret = "global-secret"\n')
        import tokenjam.core.config as cfg_mod
        from tokenjam.core.config import find_config_file
        monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
            Path("tokenjam.toml"),
            Path(".tj/config.toml"),
            global_config,
        ])
        result = find_config_file()
        assert result is not None
        assert result == global_config

    def test_local_config_takes_priority_over_global(self, tmp_path, monkeypatch):
        """Local .tj/config.toml is preferred over the global config."""
        monkeypatch.chdir(tmp_path)
        local_config = tmp_path / ".tj" / "config.toml"
        local_config.parent.mkdir(parents=True)
        local_config.write_bytes(b'version = "1"\n\n[security]\ningest_secret = "local"\n')
        global_config = tmp_path / ".config" / "tj" / "config.toml"
        global_config.parent.mkdir(parents=True)
        global_config.write_bytes(b'version = "1"\n\n[security]\ningest_secret = "global"\n')
        import tokenjam.core.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
            Path("tokenjam.toml"),
            Path(".tj/config.toml"),
            global_config,
        ])
        result = find_config_file()
        assert result is not None
        # .tj/config.toml is a relative path so resolve to compare
        assert result == Path(".tj/config.toml")

    def test_returns_none_when_no_config_anywhere(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import tokenjam.core.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
            Path("tokenjam.toml"),
            Path(".tj/config.toml"),
            tmp_path / ".config" / "tj" / "config.toml",
        ])
        assert find_config_file() is None


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # SEARCH_PATHS is a module-level constant built at import time; patch it
        # directly so the real ~/.config/tj/config.toml is never found.
        import tokenjam.core.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "SEARCH_PATHS", [
            Path("tokenjam.toml"),
            Path(".tj/config.toml"),
            tmp_path / ".config" / "tj" / "config.toml",
        ])
        config = load_config()
        assert config.version == "1"
        assert config.storage.path == "~/.tj/telemetry.duckdb"
        assert config.security.ingest_secret == ""

    def test_loads_from_file(self, tmp_path):
        toml_content = b'version = "1"\n\n[storage]\npath = "/tmp/test.duckdb"\n'
        config_file = tmp_path / "tokenjam.toml"
        config_file.write_bytes(toml_content)
        config = load_config(str(config_file))
        assert config.storage.path == "/tmp/test.duckdb"

    def test_raises_on_missing_override(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/tj.toml")

    def test_binary_mode_required(self, tmp_path):
        # Verify the file is opened in binary mode by testing a valid TOML
        config_file = tmp_path / "tokenjam.toml"
        config_file.write_bytes(b'version = "2"\n')
        config = load_config(str(config_file))
        assert config.version == "2"


class TestParse:
    def test_empty_dict_returns_defaults(self):
        config = _parse({})
        assert config.version == "1"
        assert config.agents == {}
        # prompts (needed by trim / cache-recommend / reuse) and tool_inputs
        # (needed by script / verbosity's argument-shape clustering) both
        # default on; completions/tool_outputs stay off.
        assert config.capture.prompts is True
        assert config.capture.completions is False
        assert config.capture.tool_inputs is True
        assert config.capture.tool_outputs is False

    def test_agents_parsed(self):
        raw = {
            "agents": {
                "my-agent": {
                    "description": "Test agent",
                    "budget": {"daily_usd": 5.0, "session_usd": 1.0},
                    "sensitive_actions": [
                        {"name": "send_email", "severity": "critical"}
                    ],
                }
            }
        }
        config = _parse(raw)
        assert "my-agent" in config.agents
        agent = config.agents["my-agent"]
        assert agent.description == "Test agent"
        assert agent.budget.daily_usd == 5.0
        assert agent.budget.session_usd == 1.0
        assert len(agent.sensitive_actions) == 1
        assert agent.sensitive_actions[0].name == "send_email"
        assert agent.sensitive_actions[0].severity == "critical"

    def test_security_parsed(self):
        raw = {"security": {"ingest_secret": "my-secret", "max_attribute_bytes": 1024}}
        config = _parse(raw)
        assert config.security.ingest_secret == "my-secret"
        assert config.security.max_attribute_bytes == 1024

    def test_capture_parsed(self):
        raw = {"capture": {"prompts": True, "tool_outputs": True}}
        config = _parse(raw)
        assert config.capture.prompts is True
        assert config.capture.completions is False
        assert config.capture.tool_outputs is True

    def test_alerts_channels_parsed(self):
        raw = {
            "alerts": {
                "cooldown_seconds": 120,
                "channels": [
                    {"type": "stdout"},
                    {"type": "ntfy", "topic": "my-topic"},
                ],
            }
        }
        config = _parse(raw)
        assert config.alerts.cooldown_seconds == 120
        assert len(config.alerts.channels) == 2
        assert config.alerts.channels[1].topic == "my-topic"

    def test_default_alert_channel_is_stdout(self):
        config = _parse({})
        assert len(config.alerts.channels) == 1
        assert config.alerts.channels[0].type == "stdout"

    def test_api_auth_parsed(self):
        raw = {"api": {"port": 8080, "auth": {"enabled": True, "api_key": "key123"}}}
        config = _parse(raw)
        assert config.api.port == 8080
        assert config.api.auth.enabled is True
        assert config.api.auth.api_key == "key123"

    def test_drift_config_parsed(self):
        raw = {
            "agents": {
                "a1": {"drift": {"enabled": False, "token_threshold": 3.0}}
            }
        }
        config = _parse(raw)
        assert config.agents["a1"].drift.enabled is False
        assert config.agents["a1"].drift.token_threshold == 3.0
        assert config.agents["a1"].drift.baseline_sessions == 10  # default


class TestSerialise:
    def test_roundtrip(self):
        config = TjConfig(
            version="1",
            agents={
                "test": AgentConfig(
                    description="A test agent",
                    budget=BudgetConfig(daily_usd=5.0),
                    sensitive_actions=[SensitiveAction(name="rm_rf", severity="critical")],
                )
            },
            security=SecurityConfig(ingest_secret="secret123"),
            capture=CaptureConfig(prompts=True),
        )
        serialised = _serialise(config)
        restored = _parse(serialised)

        assert restored.version == "1"
        assert restored.agents["test"].description == "A test agent"
        assert restored.agents["test"].budget.daily_usd == 5.0
        assert restored.agents["test"].sensitive_actions[0].name == "rm_rf"
        assert restored.security.ingest_secret == "secret123"
        assert restored.capture.prompts is True

    def test_proxy_roundtrip(self):
        """[proxy] config round-trips through serialise/parse (#219)."""
        config = TjConfig(version="1")
        config.proxy.enabled = True
        config.proxy.killswitch = True
        config.proxy.port = 7392
        restored = _parse(_serialise(config))
        assert restored.proxy.enabled is True
        assert restored.proxy.killswitch is True
        assert restored.proxy.port == 7392
        assert restored.proxy.mode == "suggest"
        # Defaults when [proxy] is absent.
        assert _parse({"version": "1"}).proxy.enabled is False

    def test_policies_roundtrip(self):
        """[[policies]] enforcement-plane policies round-trip (#220)."""
        from tokenjam.core.config import PolicyConfig
        config = TjConfig(version="1", policies=[
            PolicyConfig(name="cap", kind="noop", mode="enforce",
                         target_provider="openai", params={"limit": 5}),
        ])
        restored = _parse(_serialise(config))
        assert len(restored.policies) == 1
        p = restored.policies[0]
        assert p.name == "cap"
        assert p.kind == "noop"
        assert p.mode == "enforce"
        assert p.target_provider == "openai"
        assert p.params == {"limit": 5}
        # Defaults when [[policies]] absent / malformed entries skipped.
        assert _parse({"version": "1"}).policies == []
        assert _parse({"version": "1", "policies": [{"name": "x"}]}).policies == []  # no kind

    def test_session_idle_minutes_roundtrip(self):
        config = TjConfig(version="1", session_idle_minutes=90)
        serialised = _serialise(config)
        # Maps to the [sessions] table, not a bare top-level scalar.
        assert serialised["sessions"]["idle_minutes"] == 90
        assert "session_idle_minutes" not in serialised
        restored = _parse(serialised)
        assert restored.session_idle_minutes == 90

    def test_session_idle_minutes_defaults_when_absent(self):
        restored = _parse({"version": "1"})
        assert restored.session_idle_minutes == 240

    def test_serialise_excludes_config_path(self):
        """Regression: config_path is a Path object which is not TOML
        serializable. _serialise() must exclude it. See v0.1.7 fix."""
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret="s"),
            config_path=Path("/some/path/tj.toml"),
        )
        serialised = _serialise(config)
        assert "config_path" not in serialised

    def test_write_config_after_load_roundtrip(self, tmp_path):
        """Regression: load_config sets config_path (a Path). Writing that
        config back must not crash with 'PosixPath is not TOML serializable'.
        See v0.1.7 fix."""
        from tokenjam.core.config import write_config

        toml_content = b'version = "1"\n\n[security]\ningest_secret = "abc"\n'
        config_file = tmp_path / "tokenjam.toml"
        config_file.write_bytes(toml_content)

        config = load_config(str(config_file))
        assert config.config_path is not None

        out_path = tmp_path / "out.toml"
        write_config(config, out_path)
        assert out_path.exists()
        reloaded = load_config(str(out_path))
        assert reloaded.security.ingest_secret == "abc"


class TestResolveEffectiveBudget:
    def test_agent_with_both_fields_uses_agent_values(self):
        config = TjConfig(
            version="1",
            defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=10.0, session_usd=2.0)),
            agents={"a": AgentConfig(budget=BudgetConfig(daily_usd=5.0, session_usd=1.0))},
        )
        eff = resolve_effective_budget("a", config)
        assert eff.daily_usd == 5.0
        assert eff.session_usd == 1.0

    def test_agent_with_partial_fields_merges_from_defaults(self):
        config = TjConfig(
            version="1",
            defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=10.0)),
            agents={"a": AgentConfig(budget=BudgetConfig(session_usd=1.0))},
        )
        eff = resolve_effective_budget("a", config)
        assert eff.daily_usd == 10.0
        assert eff.session_usd == 1.0

    def test_unknown_agent_uses_defaults(self):
        config = TjConfig(
            version="1",
            defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=10.0, session_usd=2.0)),
        )
        eff = resolve_effective_budget("unknown", config)
        assert eff.daily_usd == 10.0
        assert eff.session_usd == 2.0

    def test_no_defaults_no_agent_returns_none_both(self):
        config = TjConfig(version="1")
        eff = resolve_effective_budget("any", config)
        assert eff.daily_usd is None
        assert eff.session_usd is None

    def test_agent_explicit_none_falls_through_to_defaults(self):
        config = TjConfig(
            version="1",
            defaults=DefaultsConfig(budget=BudgetConfig(daily_usd=10.0)),
            agents={"a": AgentConfig(budget=BudgetConfig(daily_usd=None, session_usd=5.0))},
        )
        eff = resolve_effective_budget("a", config)
        assert eff.daily_usd == 10.0
        assert eff.session_usd == 5.0


class TestValidateBudgetValue:
    def test_positive_returns_value(self):
        assert validate_budget_value(5.0, "daily_usd") == 5.0

    def test_zero_returns_none(self):
        assert validate_budget_value(0.0, "daily_usd") is None

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_budget_value(-1.0, "daily_usd")
