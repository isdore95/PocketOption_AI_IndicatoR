from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    INTERVAL,
    MINIMUM_CANDLES,
    PERIOD,
    REQUIRE_COMPLETE_DATA,
)


def _clean_column_name(column: object) -> str:
    """Convert a column name into a simple string."""
    if isinstance(column, tuple):
        return str(column[0])
    return str(column)


def _normalise_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Yahoo Finance output into standard OHLCV columns.

    The function handles both normal columns and MultiIndex columns.
    """
    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            _clean_column_name(column).lower()
            for column in result.columns
        ]
    else:
        result.columns = [
            str(column).lower()
            for column in result.columns
        ]

    required_columns = ["open", "high", "low", "close", "volume"]

    for column in required_columns:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[required_columns].copy()

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )

    result = result[~result.index.duplicated(keep="last")]

    return result.sort_index()


def get_market_data(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
    minimum_candles: int = MINIMUM_CANDLES,
) -> pd.DataFrame:
    """
    Download recent market candles for one symbol.

    Returns:
        A clean OHLCV DataFrame.

    Raises:
        ValueError: if there is not enough usable market data.
        RuntimeError: if Yahoo Finance cannot be reached or returns no data.
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    if minimum_candles < 1:
        raise ValueError("minimum_candles must be greater than zero.")

    try:
        data = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Market-data request failed for {symbol}: {exc}"
        ) from exc

    cleaned = _normalise_dataframe(data)

    if cleaned.empty:
        raise RuntimeError(
            f"No market data was returned for {symbol}."
        )

    if len(cleaned) < minimum_candles:
        raise ValueError(
            f"Not enough candles for {symbol}. "
            f"Required: {minimum_candles}, "
            f"received: {len(cleaned)}."
        )

    if REQUIRE_COMPLETE_DATA:
        if cleaned[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError(
                f"Missing OHLC values detected for {symbol}."
            )

    return cleaned


def get_latest_candle(
    symbol: str,
    period: str = PERIOD,
    interval: str = INTERVAL,
) -> pd.Series:
    """Return the most recent completed candle available."""
    data = get_market_data(
        symbol=symbol,
        period=period,
        interval=interval,
        minimum_candles=MINIMUM_CANDLES,
    )

    if data.empty:
        raise RuntimeError(
            f"No candle data available for {symbol}."
        )

    return data.iloc[-1]


def get_price(symbol: str) -> float:
    """Return the latest available close price."""
    candle = get_latest_candle(symbol)

    price = float(candle["close"])

    if price <= 0:
        raise ValueError(
            f"Invalid latest price for {symbol}: {price}"
        )

    return price


def test_market_data(
    symbol: str = "EURUSD=X",
) -> bool:
    """
    Basic market-data health check.

    This is useful later when we run the project automatically.
    """
    data = get_market_data(symbol)

    required_columns = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if not required_columns.issubset(data.columns):
        return False

    if data.empty:
        return False

    return True


def retry_market_data(
    symbol: str,
    attempts: int = 3,
    delay_seconds: int = 5,
) -> pd.DataFrame:
    """
    Retry market-data retrieval if a temporary failure occurs.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")

    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            return get_market_data(symbol)
        except Exception as exc:
            last_error = exc

            if attempt < attempts - 1:
                time.sleep(delay_seconds)

    raise RuntimeError(
        f"Unable to retrieve market data for {symbol} "
        f"after {attempts} attempts: {last_error}"
    )


if __name__ == "__main__":
    print("PocketOption AI Indicator - Market Data Test")
    print("=" * 50)

    test_symbol = "EURUSD=X"

    try:
        data = get_market_data(test_symbol)

        print(f"Symbol: {test_symbol}")
        print(f"Candles received: {len(data)}")
        print(f"Latest close: {data['close'].iloc[-1]}")
        print()
        print("Latest candle:")
        print(data.iloc[-1])
        print()
        print("Market data test: PASSED")

    except Exception as error:
        print()
        print("Market data test: FAILED")
        print(f"Reason: {error}")
