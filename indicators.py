from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

from config import (
    ATR_PERIOD,
    BOLLINGER_PERIOD,
    BOLLINGER_STD,
    EMA_FAST,
    EMA_SLOW,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
)


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators used by the strategy.

    Required input columns:
        open, high, low, close, volume

    Returns:
        A new DataFrame containing the original market data
        plus all calculated indicators.
    """

    if data is None or data.empty:
        raise ValueError("Market data is empty.")

    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing required market columns: {sorted(missing)}"
        )

    result = data.copy()

    # Make sure price columns are numeric.
    for column in ["open", "high", "low", "close"]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=["open", "high", "low", "close"]
    )

    if len(result) < EMA_SLOW + 10:
        raise ValueError(
            "Not enough candles to calculate indicators."
        )

    # ==========================================
    # EMA
    # ==========================================

    result["ema_fast"] = EMAIndicator(
        close=result["close"],
        window=EMA_FAST,
        fillna=False,
    ).ema_indicator()

    result["ema_slow"] = EMAIndicator(
        close=result["close"],
        window=EMA_SLOW,
        fillna=False,
    ).ema_indicator()

    # ==========================================
    # RSI
    # ==========================================

    result["rsi"] = RSIIndicator(
        close=result["close"],
        window=RSI_PERIOD,
        fillna=False,
    ).rsi()

    # ==========================================
    # MACD
    # ==========================================

    macd = MACD(
        close=result["close"],
        window_fast=MACD_FAST,
        window_slow=MACD_SLOW,
        window_sign=MACD_SIGNAL,
        fillna=False,
    )

    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_histogram"] = macd.macd_diff()

    # ==========================================
    # BOLLINGER BANDS
    # ==========================================

    bollinger = BollingerBands(
        close=result["close"],
        window=BOLLINGER_PERIOD,
        window_dev=BOLLINGER_STD,
        fillna=False,
    )

    result["bb_middle"] = bollinger.bollinger_mavg()
    result["bb_upper"] = bollinger.bollinger_hband()
    result["bb_lower"] = bollinger.bollinger_lband()
    result["bb_width"] = bollinger.bollinger_wband()

    # ==========================================
    # ATR
    # ==========================================

    atr = AverageTrueRange(
        high=result["high"],
        low=result["low"],
        close=result["close"],
        window=ATR_PERIOD,
        fillna=False,
    )

    result["atr"] = atr.average_true_range()

    # ==========================================
    # CANDLE INFORMATION
    # ==========================================

    result["candle_body"] = (
        result["close"] - result["open"]
    )

    result["candle_range"] = (
        result["high"] - result["low"]
    )

    result["bullish_candle"] = (
        result["close"] > result["open"]
    )

    result["bearish_candle"] = (
        result["close"] < result["open"]
    )

    # ==========================================
    # TREND INFORMATION
    # ==========================================

    result["trend_up"] = (
        result["ema_fast"] > result["ema_slow"]
    )

    result["trend_down"] = (
        result["ema_fast"] < result["ema_slow"]
    )

    # ==========================================
    # MACD MOMENTUM
    # ==========================================

    result["macd_bullish"] = (
        result["macd"] > result["macd_signal"]
    )

    result["macd_bearish"] = (
        result["macd"] < result["macd_signal"]
    )

    # ==========================================
    # RSI CONDITIONS
    # ==========================================

    result["rsi_bullish_zone"] = (
        (result["rsi"] >= 50)
        & (result["rsi"] < 70)
    )

    result["rsi_bearish_zone"] = (
        (result["rsi"] <= 50)
        & (result["rsi"] > 30)
    )

    result["rsi_overbought"] = result["rsi"] >= 70

    result["rsi_oversold"] = result["rsi"] <= 30

    # ==========================================
    # BOLLINGER CONDITIONS
    # ==========================================

    result["above_bb_middle"] = (
        result["close"] > result["bb_middle"]
    )

    result["below_bb_middle"] = (
        result["close"] < result["bb_middle"]
    )

    result["near_bb_upper"] = (
        result["close"] >= result["bb_upper"] * 0.998
    )

    result["near_bb_lower"] = (
        result["close"] <= result["bb_lower"] * 1.002
    )

    # ==========================================
    # REMOVE INVALID INDICATOR ROWS
    # ==========================================

    indicator_columns = [
        "ema_fast",
        "ema_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "atr",
    ]

    result = result.dropna(
        subset=indicator_columns
    )

    if result.empty:
        raise ValueError(
            "No valid rows remain after indicator calculation."
        )

    return result


def get_latest_indicators(
    data: pd.DataFrame,
) -> pd.Series:
    """Calculate indicators and return the latest row."""

    calculated = calculate_indicators(data)

    if calculated.empty:
        raise ValueError(
            "Indicator calculation returned no data."
        )

    return calculated.iloc[-1]


def validate_indicators(
    data: pd.DataFrame,
) -> bool:
    """Check that the indicator engine is producing valid values."""

    calculated = calculate_indicators(data)

    if calculated.empty:
        return False

    latest = calculated.iloc[-1]

    required = [
        "ema_fast",
        "ema_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_middle",
        "bb_upper",
        "bb_lower",
        "atr",
    ]

    for column in required:
        value = latest[column]

        if pd.isna(value):
            return False

    return True


if __name__ == "__main__":
    from market import get_market_data

    print("PocketOption AI Indicator - Indicator Test")
    print("=" * 50)

    symbol = "EURUSD=X"

    try:
        market_data = get_market_data(symbol)

        indicators = calculate_indicators(market_data)

        latest = indicators.iloc[-1]

        print(f"Symbol: {symbol}")
        print(f"Close: {latest['close']}")
        print(f"EMA {EMA_FAST}: {latest['ema_fast']}")
        print(f"EMA {EMA_SLOW}: {latest['ema_slow']}")
        print(f"RSI: {latest['rsi']}")
        print(f"MACD: {latest['macd']}")
        print(f"MACD Signal: {latest['macd_signal']}")
        print(f"BB Upper: {latest['bb_upper']}")
        print(f"BB Middle: {latest['bb_middle']}")
        print(f"BB Lower: {latest['bb_lower']}")
        print(f"ATR: {latest['atr']}")
        print()
        print("Indicator test: PASSED")

    except Exception as error:
        print()
        print("Indicator test: FAILED")
        print(f"Reason: {error}")
