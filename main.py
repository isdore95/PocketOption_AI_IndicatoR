from __future__ import annotations

from datetime import datetime
from typing import Optional

from config import (
    BOT_NAME,
    DEFAULT_EXPIRY_MINUTES,
    INTERVAL,
    MIN_CONFIDENCE,
    SYMBOLS,
    TIMEZONE,
    VERSION,
    validate_configuration,
)
from market import get_market_data
from strategy import SignalResult, calculate_signal, format_signal


def get_display_time() -> str:
    """
    Return the current system time for display.

    The project timezone is documented as Africa/Lagos.
    GitHub Actions can later be configured to use the same
    timezone when automated execution is added.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def analyze_symbol(symbol: str) -> Optional[SignalResult]:
    """
    Download market data and calculate one signal.

    A failure on one pair does not stop the entire scanner.
    """

    print()
    print("-" * 60)
    print(f"Analyzing {symbol}...")

    try:
        market_data = get_market_data(symbol)

        result = calculate_signal(
            data=market_data,
            symbol=symbol,
        )

        return result

    except Exception as error:
        print(f"ERROR: {symbol}")
        print(f"Reason: {error}")
        return None


def print_signal_summary(result: SignalResult) -> None:
    """Print one signal in a clean format."""

    print()
    print(format_signal(result))
    print(
        f"SUGGESTED ANALYSIS EXPIRY: "
        f"{DEFAULT_EXPIRY_MINUTES} minute(s)"
    )


def run_scanner() -> list[SignalResult]:
    """
    Analyze every configured symbol.

    Returns only successfully calculated signals.
    """

    results: list[SignalResult] = []

    print("=" * 60)
    print(BOT_NAME)
    print(f"Version: {VERSION}")
    print("=" * 60)
    print(f"Time: {get_display_time()}")
    print(f"Timezone setting: {TIMEZONE}")
    print(f"Timeframe: {INTERVAL}")
    print(f"Minimum confidence: {MIN_CONFIDENCE}%")
    print(f"Pairs configured: {len(SYMBOLS)}")
    print()

    for symbol in SYMBOLS:
        result = analyze_symbol(symbol)

        if result is None:
            continue

        results.append(result)

        print_signal_summary(result)

    return results


def print_final_summary(
    results: list[SignalResult],
) -> None:
    """Print a compact summary after scanning all pairs."""

    print()
    print("=" * 60)
    print("FINAL SCAN SUMMARY")
    print("=" * 60)

    if not results:
        print("No valid signals were calculated.")
        return

    call_count = 0
    put_count = 0
    wait_count = 0

    for result in results:

        if result.signal == "CALL":
            call_count += 1

        elif result.signal == "PUT":
            put_count += 1

        elif result.signal == "WAIT":
            wait_count += 1

        print(
            f"{result.symbol:<12} "
            f"{result.signal:<5} "
            f"{result.confidence:>3}% "
            f"@ {result.price}"
        )

    print()
    print(f"CALL signals: {call_count}")
    print(f"PUT signals:  {put_count}")
    print(f"WAIT signals: {wait_count}")
    print(f"Successful analyses: {len(results)}")
    print("=" * 60)


def main() -> int:
    """
    Main application entry point.
    """

    try:
        validate_configuration()

    except Exception as error:
        print("CONFIGURATION ERROR")
        print(error)
        return 1

    print()
    print(f"Starting {BOT_NAME}...")
    print()

    results = run_scanner()

    print_final_summary(results)

    print()
    print(
        "IMPORTANT: Signals are analytical outputs, "
        "not guaranteed winning trades."
    )
    print(
        "This version does not place trades automatically."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
