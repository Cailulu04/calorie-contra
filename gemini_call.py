"""Run blocking Gemini SDK calls with a timeout (avoids hung requests)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_executor = ThreadPoolExecutor(max_workers=4)
DEFAULT_GEMINI_TIMEOUT_SEC = 55.0


async def run_sync_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float = DEFAULT_GEMINI_TIMEOUT_SEC,
    **kwargs: Any,
) -> Any:
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))
    return await asyncio.wait_for(fut, timeout=timeout)


async def generate_content_with_timeout(
    model,
    contents,
    *,
    timeout: float = DEFAULT_GEMINI_TIMEOUT_SEC,
    **generation_kwargs,
):
    def _call():
        return model.generate_content(contents, **generation_kwargs)

    return await run_sync_with_timeout(_call, timeout=timeout)
