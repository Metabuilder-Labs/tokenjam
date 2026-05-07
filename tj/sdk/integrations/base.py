from typing import Protocol, runtime_checkable


@runtime_checkable
class Integration(Protocol):
    """
    Formal interface for all framework and provider integrations.

    Convenience functions like patch_anthropic() instantiate and install
    the integration automatically. You can also install manually:

        integration = AnthropicIntegration()
        integration.install(tracer)

    tj doctor uses `integration.installed` to list active integrations
    and detect conflicts (two integrations patching the same method).
    """
    name: str
    installed: bool

    def install(self, tracer) -> None:
        """Register all hooks. Idempotent — safe to call multiple times."""
        ...

    def uninstall(self) -> None:
        """Remove all hooks. Called on process shutdown."""
        ...
