"""
Notification utilities for the crypto trading bot.

This module defines a simple interface for sending alerts to Discord via
webhooks. The `notify` function will read the webhook URL from your
``.env`` file and post a short message. If the webhook isn't configured
the message will be printed to stdout instead.

Dependencies: ``requests`` and ``python-dotenv``.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Discord webhook URL
WEBHOOK: str = (os.getenv("DISCORD_WEBHOOK") or "").strip()


def notify(text: str) -> bool:
    """Send a short alert to Discord.

    Args:
        text: The message to send. Messages longer than 1900 characters
            will be truncated to avoid hitting Discord's 2 000‑character
            limit.

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    if not WEBHOOK:
        # No webhook configured – print the message and return False.
        print("[Discord] No DISCORD_WEBHOOK in .env")
        print(text)
        return False
    try:
        payload = {"content": text[:1900]}
        r = requests.post(WEBHOOK, json=payload, timeout=15)
        ok = (r.status_code == 204)
        print("[Discord]", "sent" if ok else f"failed {r.status_code} {r.text[:120]}")
        return ok
    except Exception as e:
        print("[Discord] exception:", e)
        return False