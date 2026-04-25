"""DesktopNotifier and MultiNotifier tests.

We patch subprocess.run + shutil.which so the tests work on any host
regardless of whether osascript / notify-send / powershell are
actually installed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot.notifications import (
    DesktopNotifier,
    MultiNotifier,
    NullNotifier,
    TelegramNotifier,
)


# ---------------------------------------------------------------------------
# Backend selection.
# ---------------------------------------------------------------------------
def test_macos_uses_osascript():
    with patch("bot.notifications.platform.system", return_value="Darwin"), \
         patch("bot.notifications.shutil.which", return_value="/usr/bin/osascript"), \
         patch("bot.notifications.subprocess.run") as run:
        n = DesktopNotifier()
        assert n.enabled is True
        n.send("hello", title="Bot")
    args = run.call_args[0][0]
    assert args[0] == "osascript"
    assert "-e" in args
    assert "display notification" in args[-1]
    assert '"Bot"' in args[-1]


def test_linux_uses_notify_send():
    with patch("bot.notifications.platform.system", return_value="Linux"), \
         patch("bot.notifications.shutil.which", return_value="/usr/bin/notify-send"), \
         patch("bot.notifications.subprocess.run") as run:
        n = DesktopNotifier()
        assert n.enabled is True
        n.send("hello", title="Bot")
    args = run.call_args[0][0]
    assert args[0] == "notify-send"
    assert "Bot" in args
    assert "hello" in args


def test_windows_uses_powershell():
    with patch("bot.notifications.platform.system", return_value="Windows"), \
         patch("bot.notifications.shutil.which", return_value="powershell.exe"), \
         patch("bot.notifications.subprocess.run") as run:
        n = DesktopNotifier()
        assert n.enabled is True
        n.send("hello", title="Bot")
    args = run.call_args[0][0]
    assert args[0] == "powershell"
    assert "-NoProfile" in args


def test_unknown_platform_disabled():
    with patch("bot.notifications.platform.system", return_value="FreeBSD"):
        n = DesktopNotifier()
    assert n.enabled is False


def test_disabled_notifier_send_is_noop():
    with patch("bot.notifications.platform.system", return_value="FreeBSD"), \
         patch("bot.notifications.subprocess.run") as run:
        DesktopNotifier().send("hello")
    run.assert_not_called()


def test_missing_backend_cli_disables():
    """If we're on macOS but osascript isn't installed (locked-down box),
    enabled should be False - never crash."""
    with patch("bot.notifications.platform.system", return_value="Darwin"), \
         patch("bot.notifications.shutil.which", return_value=None):
        n = DesktopNotifier()
    assert n.enabled is False


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------
def test_subprocess_failure_is_swallowed():
    """A failed pop-up must never crash the trading loop."""
    def boom(*args, **kwargs):
        raise RuntimeError("display server gone")

    with patch("bot.notifications.platform.system", return_value="Darwin"), \
         patch("bot.notifications.shutil.which", return_value="/usr/bin/osascript"), \
         patch("bot.notifications.subprocess.run", side_effect=boom):
        n = DesktopNotifier()
        # Should not raise.
        n.send("hello")


def test_applescript_escapes_quotes_and_backslashes():
    # Embed both characters in the title and body and verify they reach
    # osascript in escaped form so the script doesn't error out.
    with patch("bot.notifications.platform.system", return_value="Darwin"), \
         patch("bot.notifications.shutil.which", return_value="/usr/bin/osascript"), \
         patch("bot.notifications.subprocess.run") as run:
        DesktopNotifier().send('hi "world"\\path', title='Bot "X"')
    script = run.call_args[0][0][-1]
    # Both characters were escaped.
    assert '\\"' in script
    assert "\\\\" in script


# ---------------------------------------------------------------------------
# MultiNotifier fan-out.
# ---------------------------------------------------------------------------
def test_multi_notifier_fans_out_to_each_channel():
    a, b = MagicMock(), MagicMock()
    MultiNotifier([a, b]).send("hello")
    a.send.assert_called_once_with("hello")
    b.send.assert_called_once_with("hello")


def test_multi_notifier_passes_title_to_desktop_only():
    desktop = MagicMock(spec=DesktopNotifier)
    telegram = MagicMock(spec=TelegramNotifier)
    MultiNotifier([desktop, telegram]).send("hello", title="X")
    desktop.send.assert_called_once_with("hello", title="X")
    # Telegram doesn't accept a title kwarg - we must not have passed one.
    telegram.send.assert_called_once_with("hello")


def test_multi_notifier_isolates_per_channel_failure():
    """One failing channel must not stop the others."""
    failing = MagicMock()
    failing.send.side_effect = RuntimeError("network gone")
    ok = MagicMock()
    MultiNotifier([failing, ok]).send("hello")
    ok.send.assert_called_once_with("hello")


def test_multi_notifier_drops_none_channels():
    a = MagicMock()
    n = MultiNotifier([None, a, None])
    n.send("hello")
    a.send.assert_called_once_with("hello")
