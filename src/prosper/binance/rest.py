"""Binance REST API client."""

from __future__ import annotations

import time
from typing import Any

import requests
from rich.console import Console

from prosper.config import get_settings
from prosper.utils.http import handle_rate_limit

console = Console()


class BinanceRESTClient:
    """Client for Binance REST API."""

    def __init__(self, base_url: str | None = None, timeout: int = 30) -> None:
        """
        Initialize Binance REST client.

        Args:
            base_url: Base URL for Binance API (defaults to config)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.base_url = base_url or settings.binance_api_base
        self.timeout = timeout
        self.session = requests.Session()

    def __enter__(self) -> BinanceRESTClient:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.session.close()

    def get_exchange_info(self) -> dict[str, Any]:
        """
        Get exchange information including trading symbols.

        Returns:
            Exchange info dictionary

        """
        settings = get_settings()
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"

        last_err: Exception | None = None
        for attempt in range(settings.download_retry_max + 1):
            try:
                response = self.session.get(endpoint, timeout=self.timeout)
                delay = handle_rate_limit(response, settings.api_rate_limit_backoff_base)
                if delay > 0:
                    time.sleep(delay)

                if response.status_code == 429 and attempt < settings.download_retry_max:
                    continue

                response.raise_for_status()
                return response.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= settings.download_retry_max:
                    break
                time.sleep(settings.download_retry_backoff**attempt)

        if last_err is not None:
            raise last_err
        raise RuntimeError("get_exchange_info failed")

    def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        """
        Get klines (candlestick data) from Binance.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            interval: Kline interval (1m, 5m, 1h, 1d, etc.)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            limit: Maximum number of klines (max 1000)

        Returns:
            List of kline arrays

        """
        settings = get_settings()
        limit = min(limit, 1000)  # Binance max limit

        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        endpoint = f"{self.base_url}/api/v3/klines"
        last_err: Exception | None = None
        for attempt in range(settings.download_retry_max + 1):
            try:
                response = self.session.get(endpoint, params=params, timeout=self.timeout)
                delay = handle_rate_limit(response, settings.api_rate_limit_backoff_base)
                if delay > 0:
                    time.sleep(delay)

                if response.status_code == 429 and attempt < settings.download_retry_max:
                    continue

                response.raise_for_status()
                return response.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= settings.download_retry_max:
                    break
                time.sleep(settings.download_retry_backoff**attempt)

        if last_err is not None:
            raise last_err
        raise RuntimeError("get_klines failed")

    def get_symbols_spot(self, quote_asset: str | None = None) -> list[str]:
        """
        Get list of SPOT trading symbols.

        Args:
            quote_asset: Filter by quote asset (e.g., "USDT"). If None, returns all.

        Returns:
            List of symbol strings
        """
        exchange_info = self.get_exchange_info()
        symbols: list[str] = []

        for symbol_info in exchange_info.get("symbols", []):
            if symbol_info.get("status") != "TRADING":
                continue
            # Spot exchangeInfo: accept SPOT type or any symbol (legacy API may omit type)
            sym_type = symbol_info.get("type")
            if sym_type is not None and sym_type != "SPOT":
                continue

            symbol = symbol_info.get("symbol", "")
            if not symbol:
                continue
            if quote_asset:
                # Support both symbol suffix and explicit quoteAsset
                quote = symbol_info.get("quoteAsset", "")
                if symbol.endswith(quote_asset) or quote == quote_asset:
                    symbols.append(symbol)
            else:
                symbols.append(symbol)

        return sorted(symbols)
