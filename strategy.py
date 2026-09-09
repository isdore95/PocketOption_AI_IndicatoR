from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from config import (
    CALL_SIGNAL,
    EMA_FAST,
    EMA_SLOW,
    MIN_CONFIDENCE,
    PUT_SIGNAL,
    WAIT_SIGNAL,
)
from indicators import calculate_indicators


@dataclass
class SignalResult:
    """
    Represents one complete trading-signal decision.
    """

    symbol: str
    signal: str
    confidence: int
    price: float
    reasons: List[str]
    warnings: List[str]

    def to_dict(self) -> dict:
        """Convert the signal into a normal dictionary."""

        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "price": self.price,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


def _safe_float(value: object) -> float:
    """Convert a value to float and reject invalid values."""

    number = float(value)

    if pd.isna(number):
        raise ValueError("Indicator contains NaN.")

    return number


def _validate_latest_row(row: pd.Series) -> None:
    """Make sure all values required by the strategy exist."""

    required = [
        "close",
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

    missing = []

    for column in required:
        if column not in row.index:
            missing.append(column)
            continue

        if pd.isna(row[column]):
            missing.append(column)

    if missing:
        raise ValueError(
            "Missing or invalid strategy values: "
            + ", ".join(missing)
        )


def _calculate_signal_score(
    row: pd.Series,
) -> tuple[int, int, List[str], List[str]]:
    """
    Score bullish and bearish confirmations.

    Maximum raw score:
        100 points.

    The strategy deliberately uses WAIT when the evidence
    is not strong enough.
    """

    bullish_score = 0
    bearish_score = 0

    bullish_reasons: List[str] = []
    bearish_reasons: List[str] = []

    warnings: List[str] = []

    close = _safe_float(row["close"])
    ema_fast = _safe_float(row["ema_fast"])
    ema_slow = _safe_float(row["ema_slow"])

    rsi = _safe_float(row["rsi"])

    macd = _safe_float(row["macd"])
    macd_signal = _safe_float(row["macd_signal"])
    macd_histogram = _safe_float(row["macd_histogram"])

    bb_middle = _safe_float(row["bb_middle"])
    bb_upper = _safe_float(row["bb_upper"])
    bb_lower = _safe_float(row["bb_lower"])
    bb_width = _safe_float(row["bb_width"])

    atr = _safe_float(row["atr"])

    bullish_candle = bool(row.get("bullish_candle", False))
    bearish_candle = bool(row.get("bearish_candle", False))

    # ==========================================
    # 1. EMA TREND
    # ==========================================

    if ema_fast > ema_slow:
        bullish_score += 25
        bullish_reasons.append(
            f"EMA {EMA_FAST} is above EMA {EMA_SLOW}"
        )

    elif ema_fast < ema_slow:
        bearish_score += 25
        bearish_reasons.append(
            f"EMA {EMA_FAST} is below EMA {EMA_SLOW}"
        )

    else:
        warnings.append("EMAs are equal; trend is unclear.")

    # ==========================================
    # 2. PRICE VS EMA TREND
    # ==========================================

    if close > ema_fast and close > ema_slow:
        bullish_score += 10
        bullish_reasons.append(
            "Price is above both trend EMAs"
        )

    elif close < ema_fast and close < ema_slow:
        bearish_score += 10
        bearish_reasons.append(
            "Price is below both trend EMAs"
        )

    # ==========================================
    # 3. RSI MOMENTUM
    # ==========================================

    if 50 <= rsi < 70:
        bullish_score += 15
        bullish_reasons.append(
            f"RSI bullish momentum zone ({rsi:.1f})"
        )

    elif 30 < rsi <= 50:
        bearish_score += 15
        bearish_reasons.append(
            f"RSI bearish momentum zone ({rsi:.1f})"
        )

    elif rsi >= 70:
        warnings.append(
            f"RSI is overbought ({rsi:.1f}); CALL risk is elevated."
        )

        # Strong overbought readings do not automatically mean PUT.
        # We only give a small bearish score.
        bearish_score += 5
        bearish_reasons.append(
            f"RSI is overbought ({rsi:.1f})"
        )

    elif rsi <= 30:
        warnings.append(
            f"RSI is oversold ({rsi:.1f}); PUT risk is elevated."
        )

        # Strong oversold readings do not automatically mean CALL.
        # We only give a small bullish score.
        bullish_score += 5
        bullish_reasons.append(
            f"RSI is oversold ({rsi:.1f})"
        )

    # ==========================================
    # 4. MACD
    # ==========================================

    if macd > macd_signal and macd_histogram > 0:
        bullish_score += 20
        bullish_reasons.append(
            "MACD confirms bullish momentum"
        )

    elif macd < macd_signal and macd_histogram < 0:
        bearish_score += 20
        bearish_reasons.append(
            "MACD confirms bearish momentum"
        )

    elif macd > macd_signal:
        bullish_score += 10
        bullish_reasons.append(
            "MACD is above its signal line"
        )

    elif macd < macd_signal:
        bearish_score += 10
        bearish_reasons.append(
            "MACD is below its signal line"
        )

    # ==========================================
    # 5. BOLLINGER BAND POSITION
    # ==========================================

    if close > bb_middle and close < bb_upper:
        bullish_score += 10
        bullish_reasons.append(
            "Price is above Bollinger middle band"
        )

    elif close < bb_middle and close > bb_lower:
        bearish_score += 10
        bearish_reasons.append(
            "Price is below Bollinger middle band"
        )

    elif close >= bb_upper:
        warnings.append(
            "Price is at/above the upper Bollinger Band."
        )

    elif close <= bb_lower:
        warnings.append(
            "Price is at/below the lower Bollinger Band."
        )

    # ==========================================
    # 6. CURRENT CANDLE
    # ==========================================

    if bullish_candle:
        bullish_score += 10
        bullish_reasons.append(
            "Latest candle closed bullish"
        )

    elif bearish_candle:
        bearish_score += 10
        bearish_reasons.append(
            "Latest candle closed bearish"
        )

    # ==========================================
    # 7. VOLATILITY FILTER
    # ==========================================

    if atr <= 0:
        warnings.append(
            "ATR is invalid or zero; market volatility cannot be evaluated."
        )

    if bb_width <= 0:
        warnings.append(
            "Bollinger Band width is invalid or zero."
        )

    # ==========================================
    # NORMALIZE RAW SCORES TO 100
    # ==========================================

    bullish_score = min(bullish_score, 100)
    bearish_score = min(bearish_score, 100)

    return (
        bullish_score,
        bearish_score,
        bullish_reasons + bearish_reasons,
        warnings,
    )


def calculate_signal(
    data: pd.DataFrame,
    symbol: str = "UNKNOWN",
) -> SignalResult:
    """
    Calculate the latest CALL, PUT, or WAIT signal.

    The strategy uses only the supplied historical candle data.
    It does not use future candles.
    """

    if data is None or data.empty:
        raise ValueError("Cannot calculate a signal from empty data.")

    calculated = calculate_indicators(data)

    if calculated.empty:
        raise ValueError(
            "Indicator calculation returned no usable data."
        )

    latest = calculated.iloc[-1]

    _validate_latest_row(latest)

    (
        bullish_score,
        bearish_score,
        all_reasons,
        warnings,
    ) = _calculate_signal_score(latest)

    # Separate reasons by direction.
    bullish_reasons = []
    bearish_reasons = []

    for reason in all_reasons:
        if (
            "bullish" in reason.lower()
            or "above" in reason.lower()
            or "closed bullish" in reason.lower()
            or "oversold" in reason.lower()
        ):
            bullish_reasons.append(reason)

        if (
            "bearish" in reason.lower()
            or "below" in reason.lower()
            or "closed bearish" in reason.lower()
            or "overbought" in reason.lower()
        ):
            bearish_reasons.append(reason)

    price = _safe_float(latest["close"])

    # ==========================================
    # SIGNAL DECISION
    # ==========================================

    difference = abs(bullish_score - bearish_score)

    # If both sides are weak, WAIT.
    if (
        bullish_score < MIN_CONFIDENCE
        and bearish_score < MIN_CONFIDENCE
    ):
        signal = WAIT_SIGNAL
        confidence = max(
            bullish_score,
            bearish_score,
        )

        reasons = [
            "No direction reached the minimum confidence threshold.",
            f"Bullish score: {bullish_score}/100.",
            f"Bearish score: {bearish_score}/100.",
        ]

    # If both directions are strong, require separation.
    elif (
        bullish_score >= MIN_CONFIDENCE
        and bearish_score >= MIN_CONFIDENCE
    ):
        if difference < 15:
            signal = WAIT_SIGNAL
            confidence = max(
                bullish_score,
                bearish_score,
            )

            reasons = [
                "Bullish and bearish evidence are too close.",
                f"Bullish score: {bullish_score}/100.",
                f"Bearish score: {bearish_score}/100.",
            ]

        elif bullish_score > bearish_score:
            signal = CALL_SIGNAL
            confidence = bullish_score
            reasons = bullish_reasons

        else:
            signal = PUT_SIGNAL
            confidence = bearish_score
            reasons = bearish_reasons

    # Bullish signal.
    elif bullish_score >= MIN_CONFIDENCE:
        signal = CALL_SIGNAL
        confidence = bullish_score
        reasons = bullish_reasons

    # Bearish signal.
    elif bearish_score >= MIN_CONFIDENCE:
        signal = PUT_SIGNAL
        confidence = bearish_score
        reasons = bearish_reasons

    # Fallback.
    else:
        signal = WAIT_SIGNAL
        confidence = max(
            bullish_score,
            bearish_score,
        )

        reasons = [
            "Market conditions are not strong enough.",
        ]

    # Add score information to every decision.
    reasons = [
        f"Bullish score: {bullish_score}/100.",
        f"Bearish score: {bearish_score}/100.",
        *reasons,
    ]

    return SignalResult(
        symbol=symbol,
        signal=signal,
        confidence=int(max(0, min(100, confidence))),
        price=price,
        reasons=reasons,
        warnings=warnings,
    )


def get_signal(
    data: pd.DataFrame,
    symbol: str = "UNKNOWN",
) -> dict:
    """
    Convenience function returning a dictionary.

    This will make it easier for the Telegram and application
    modules to consume the strategy later.
    """

    result = calculate_signal(
        data=data,
        symbol=symbol,
    )

    return result.to_dict()


def format_signal(result: SignalResult) -> str:
    """Create a readable text representation of a signal."""

    lines = [
        "POCKETOPTION AI INDICATOR",
        "=" * 32,
        f"PAIR: {result.symbol}",
        f"SIGNAL: {result.signal}",
        f"CONFIDENCE SCORE: {result.confidence}%",
        f"PRICE: {result.price:.6f}",
        "",
        "REASONS:",
    ]

    for reason in result.reasons:
        lines.append(f"- {reason}")

    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")

        for warning in result.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def test_strategy(
    data: pd.DataFrame,
    symbol: str = "TEST",
) -> bool:
    """
    Basic strategy health check.

    Returns True when a valid CALL, PUT, or WAIT result
    is produced without invalid confidence values.
    """

    result = calculate_signal(
        data=data,
        symbol=symbol,
    )

    valid_signals = {
        CALL_SIGNAL,
        PUT_SIGNAL,
        WAIT_SIGNAL,
    }

    if result.signal not in valid_signals:
        return False

    if not 0 <= result.confidence <= 100:
        return False

    if result.price <= 0:
        return False

    return True


if __name__ == "__main__":
    from market import get_market_data

    print("PocketOption AI Indicator - Strategy Test")
    print("=" * 50)

    test_symbol = "EURUSD=X"

    try:
        market_data = get_market_data(test_symbol)

        result = calculate_signal(
            data=market_data,
            symbol=test_symbol,
        )

        print(format_signal(result))
        print()
        print("Strategy test: PASSED")

    except Exception as error:
        print()
        print("Strategy test: FAILED")
        print(f"Reason: {error}")
