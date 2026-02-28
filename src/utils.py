"""Utility functions for PaperLens."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any

_executor: concurrent.futures.ThreadPoolExecutor | None = None


def run_sync(coro: Coroutine[Any, Any, Any], timeout: int = 30) -> Any:
    """Run an async coroutine from sync context.

    Handles both cases:
    - No running event loop: uses asyncio.run() directly
    - Inside async context: runs in a separate thread via ThreadPoolExecutor
    """
    global _executor
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        future = _executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)
    else:
        return asyncio.run(coro)
