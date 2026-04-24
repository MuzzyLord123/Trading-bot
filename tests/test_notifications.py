from __future__ import annotations

from unittest.mock import patch

import pytest

from bot.notifications import NullNotifier, TelegramNotifier


def test_null_notifier_is_quiet():
    # No exception, no side effects.
    NullNotifier().send("anything")


def test_telegram_notifier_disabled_without_credentials():
    assert TelegramNotifier("", "").enabled is False
    assert TelegramNotifier("token", "").enabled is False
    assert TelegramNotifier("", "chat").enabled is False


def test_telegram_notifier_enabled_with_both_credentials():
    assert TelegramNotifier("token", "chat").enabled is True


def test_disabled_notifier_does_not_hit_the_network():
    with patch("bot.notifications.requests.post") as post:
        TelegramNotifier("", "").send("hello")
    post.assert_not_called()


def test_enabled_notifier_posts_to_correct_url():
    with patch("bot.notifications.requests.post") as post:
        TelegramNotifier("abc", "123").send("hi")
    post.assert_called_once()
    url = post.call_args[0][0]
    assert url == "https://api.telegram.org/botabc/sendMessage"
    body = post.call_args.kwargs["json"]
    assert body == {"chat_id": "123", "text": "hi"}


def test_network_failure_is_swallowed():
    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    with patch("bot.notifications.requests.post", side_effect=boom):
        # Should not raise.
        TelegramNotifier("abc", "123").send("hi")


def test_timeout_is_set_on_post():
    with patch("bot.notifications.requests.post") as post:
        TelegramNotifier("abc", "123").send("hi")
    assert post.call_args.kwargs["timeout"] == 5
