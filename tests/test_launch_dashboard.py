from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent


def _load():
    """Load launch_dashboard.py as a module without executing __main__."""
    module_globals = runpy.run_path(
        str(ROOT / "launch_dashboard.py"),
        run_name="<test-loader>",
    )
    return module_globals


MOD = _load()
_open_browser = MOD["_open_browser"]
_port_open = MOD["_port_open"]


def test_open_browser_none_does_nothing():
    with patch("subprocess.run") as run, patch("webbrowser.open") as web:
        _open_browser("http://x", "none")
    run.assert_not_called()
    web.assert_not_called()


def test_open_browser_safari_on_mac_uses_open_dash_a():
    with patch("platform.system", return_value="Darwin"), \
         patch("subprocess.run") as run:
        _open_browser("http://x", "safari")
    run.assert_called_once()
    assert run.call_args[0][0][:3] == ["open", "-a", "Safari"]


def test_open_browser_default_on_mac_picks_safari():
    with patch("platform.system", return_value="Darwin"), \
         patch("subprocess.run") as run:
        _open_browser("http://x", "default")
    run.assert_called_once()
    assert "Safari" in run.call_args[0][0]


def test_open_browser_default_on_windows_uses_webbrowser():
    with patch("platform.system", return_value="Windows"), \
         patch("subprocess.run") as run, \
         patch("webbrowser.open") as web:
        _open_browser("http://x", "default")
    run.assert_not_called()
    web.assert_called_once_with("http://x")


def test_open_browser_safari_on_linux_falls_back():
    with patch("platform.system", return_value="Linux"), \
         patch("webbrowser.open") as web:
        _open_browser("http://x", "safari")
    web.assert_called_once_with("http://x")


def test_port_open_returns_false_for_closed_port():
    # Port 1 is virtually never open on a user machine.
    assert _port_open("127.0.0.1", 1) is False
