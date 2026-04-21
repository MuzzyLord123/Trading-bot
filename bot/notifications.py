from __future__ import annotations

import logging

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
