"""Unit tests for org-name derivation used as the dashboard's service.namespace."""
from __future__ import annotations

from unittest.mock import patch

from tokenjam.cli.cmd_onboard import _derive_org_name


def _fake_remote(url: str):
    """Patch subprocess.run so `git remote get-url origin` returns `url`."""
    class _Result:
        returncode = 0
        stdout = url + "\n"
    return patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=_Result())


def test_derive_org_name_https():
    with _fake_remote("https://github.com/Aquanodeio/harness.git"):
        assert _derive_org_name() == "aquanodeio"


def test_derive_org_name_ssh():
    with _fake_remote("git@github.com:Aquanodeio/harness.git"):
        assert _derive_org_name() == "aquanodeio"


def test_derive_org_name_no_dotgit_suffix():
    with _fake_remote("https://github.com/Aquanodeio/console"):
        assert _derive_org_name() == "aquanodeio"


def test_derive_org_name_no_remote_returns_empty():
    class _Result:
        returncode = 128
        stdout = ""
    with patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=_Result()):
        assert _derive_org_name() == ""
