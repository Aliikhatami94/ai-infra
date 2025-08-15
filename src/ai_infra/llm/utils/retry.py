from __future__ import annotations
import asyncio, random
from typing import Any, Callable

async def with_retry(afn: Callable[[], Any], *, max_tries: int = 3, base: float = 0.5, jitter: float = 0.2):
    """Exponential backoff around an awaited call factory."""
    last: BaseException | None = None
    for i in range(max_tries):
        try:
            return await afn()
        except Exception as e:  # defensive
            last = e
            if i == max_tries - 1:
                break
            await asyncio.sleep(base * (2 ** i) + random.random() * jitter)
    raise last if last else RuntimeError("Retry failed with unknown error")