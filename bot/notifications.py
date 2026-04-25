"""Notification channels.

Three flavours, all sharing a one-method ``send(message)`` interface so
the engine can fan out without caring which is configured:

  * :class:`NullNotifier`    - silent (default)
  * :class:`TelegramNotifier` - remote, requires bot token + chat ID
  * :class:`DesktopNotifier`  - native OS pop-up (macOS / Linux / Windows)
  * :class:`MultiNotifier`    - fan out to several at once

Desktop notifications use platform-native commands so we don't pull in
any extra dependencies:

  * macOS:   ``osascript -e 'display notification ...'``
  * Linux:   ``notify-send`` (libnotify)
  * Windows: PowerShell ``[Windows.UI.Notifications.ToastNotificationManager]``

Every backend swallows its own errors. A failed notification must never
crash the trading loop.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import sys
from typing import Iterable

import requests

log = logging.getLogger("bot.notifications")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id)

    def send(self, message: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message},
                timeout=5,
            )
        except Exception as exc:
            log.warning("telegram send failed: %s", exc)


class NullNotifier:
    def send(self, message: str) -> None:  # pragma: no cover - trivial
        return


class DesktopNotifier:
    """Native OS desktop notification.

    Detects the platform once at construction time and picks the right
    backend. ``enabled`` is False when the platform isn't supported or
    the required CLI is missing - the engine can still call .send() and
    it will be a silent no-op.

    Notifications are intentionally kept simple: a short title and a
    short body. Avoid the temptation to embed full prices or risk
    parameters - those belong in the dashboard, not in a banner.
    """

    DEFAULT_TITLE = "Trading Bot"

    def __init__(self, app_name: str = DEFAULT_TITLE) -> None:
        self.app_name = app_name
        self.system = platform.system()
        self.enabled = self._detect_backend()

    def _detect_backend(self) -> bool:
        if self.system == "Darwin":
            return shutil.which("osascript") is not None
        if self.system == "Linux":
            return shutil.which("notify-send") is not None
        if self.system == "Windows":
            return shutil.which("powershell") is not None or shutil.which("powershell.exe") is not None
        return False

    def send(self, message: str, title: str | None = None) -> None:
        if not self.enabled:
            return
        title = title or self.app_name
        try:
            if self.system == "Darwin":
                self._send_macos(title, message)
            elif self.system == "Linux":
                self._send_linux(title, message)
            elif self.system == "Windows":
                self._send_windows(title, message)
        except Exception as exc:
            log.debug("desktop notification failed: %s", exc)

    @staticmethod
    def _escape_applescript(s: str) -> str:
        # AppleScript strings escape backslash and double-quote.
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def _send_macos(self, title: str, message: str) -> None:
        title_e = self._escape_applescript(title)
        msg_e = self._escape_applescript(message)
        script = f'display notification "{msg_e}" with title "{title_e}"'
        subprocess.run(
            ["osascript", "-e", script],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _send_linux(self, title: str, message: str) -> None:
        subprocess.run(
            ["notify-send", "--app-name", self.app_name, title, message],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _send_windows(self, title: str, message: str) -> None:
        # Use the BurntToast-free WinRT path; works on Win10+ default install.
        # PowerShell single-quote escape is just doubling the quote.
        title_e = title.replace("'", "''")
        msg_e = message.replace("'", "''")
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime] | Out-Null;"
            "[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime] | Out-Null;"
            f"$xml = '<toast><visual><binding template=\"ToastText02\"><text id=\"1\">{title_e}</text><text id=\"2\">{msg_e}</text></binding></visual></toast>';"
            "$doc = New-Object Windows.Data.Xml.Dom.XmlDocument; $doc.LoadXml($xml);"
            "$toast = New-Object Windows.UI.Notifications.ToastNotification $doc;"
            f"[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{self.app_name}').Show($toast);"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


class MultiNotifier:
    """Fan out a single send() to several backends. Used by the engine so
    the same trade event can pop up on the desktop AND send a Telegram
    message without duplicating the call site.
    """

    def __init__(self, channels: Iterable[object]) -> None:
        self.channels = [c for c in channels if c is not None]

    def send(self, message: str, title: str | None = None) -> None:
        for ch in self.channels:
            try:
                if isinstance(ch, DesktopNotifier):
                    ch.send(message, title=title)
                else:
                    ch.send(message)
            except Exception as exc:
                log.debug("notifier %s failed: %s", type(ch).__name__, exc)
