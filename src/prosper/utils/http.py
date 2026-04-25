"""HTTP utilities with retry/backoff helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

import requests
from requests import Response
from rich.console import Console

console = Console()


def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    backoff_base: float = 2.0,
    initial_delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (requests.RequestException,),
) -> Any:
    """Retry a function with exponential backoff."""

    last_exception: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                console.print(
                    f"[yellow]Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}[/yellow]"
                )
                time.sleep(delay)
                delay *= backoff_base
            else:
                console.print(f"[red]All retries failed: {e}[/red]")

    if last_exception is None:
        raise RuntimeError("Unexpected retry failure")
    raise last_exception


def handle_rate_limit(response: Response, backoff_base: float = 2.0) -> float:
    """
    Return a suggested delay (seconds) based on Binance rate limit headers.

    Works for both 429 responses and near-limit weight headers.
    """

    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return backoff_base**3

    used_weight_1m = response.headers.get("X-Mbx-Used-Weight-1m")
    if used_weight_1m:
        try:
            weight = int(used_weight_1m)
            if weight > 1000:  # conservative threshold
                return backoff_base**2
        except ValueError:
            pass

    return 0.0
