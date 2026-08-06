"""Add Windows-compatible initialization hook that patches the asyncio event loop
policy for DatagramTransport support."""

import sys
import asyncio


def _patch_win_event_loop():
    if sys.platform == "win32":
        try:
            policy = asyncio.WindowsSelectorEventLoopPolicy()
            asyncio.set_event_loop_policy(policy)
        except AttributeError:
            pass


def initialize():
    _patch_win_event_loop()
