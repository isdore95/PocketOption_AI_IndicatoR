import os
from dotenv import load_dotenv

# Load environment variables from .env when running locally.
load_dotenv()

# =========================
# TELEGRAM CONFIGURATION
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


# =========================
# MARKET CONFIGURATION
# =========================

# Start with major forex pairs.
# We can add more later after the core system is tested.
SYMBOLS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
]

# 5-minute candles are used for the initial strategy.
INTERVAL = "5m"

# Yahoo Finance limits very recent intraday history,
# so one trading day is sufficient for the live signal engine.
PERIOD = "1d"

# Number of candles required before calculating a signal.
MINIMUM_CANDLES = 100


# =========================
# INDICATOR SETTINGS
# =========================

EMA_FAST = 9
EMA_SLOW = 21

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

ATR_PERIOD = 14


# =========================
# SIGNAL SETTINGS
# =========================

# The strategy will only produce a trade signal when
# the calculated confidence reaches this level.
MIN_CONFIDENCE = 70

# Possible signal directions.
CALL_SIGNAL = "CALL"
PUT_SIGNAL = "PUT"
WAIT_SIGNAL = "WAIT"


# =========================
# EXPIRY SETTINGS
# =========================

# This is a suggested analysis/expiry duration.
# It is NOT a guarantee that a Pocket Option trade will win.
DEFAULT_EXPIRY_MINUTES = 5


# =========================
# SAFETY SETTINGS
# =========================

# Prevent repeated signals for the same pair/direction
# within this many minutes.
SIGNAL_COOLDOWN_MINUTES = 5

# The bot will not issue a signal when market data is
# missing or insufficient.
REQUIRE_COMPLETE_DATA = True


# =========================
# APPLICATION SETTINGS
# =========================

BOT_NAME = "PocketOption AI Indicator"

VERSION = "1.0.0"

TIMEZONE = "Africa/Lagos"


def validate_configuration():
    """
    Check the configuration before the application starts.

    Telegram credentials are intentionally allowed to be empty
    during development because the strategy can be tested without
    sending Telegram messages.
    """

    if not SYMBOLS:
        raise ValueError("SYMBOLS cannot be empty.")

    if MINIMUM_CANDLES < 50:
        raise ValueError("MINIMUM_CANDLES must be at least 50.")

    if EMA_FAST <= 0 or EMA_SLOW <= 0:
        raise ValueError("EMA periods must be greater than zero.")

    if EMA_FAST >= EMA_SLOW:
        raise ValueError("EMA_FAST must be smaller than EMA_SLOW.")

    if RSI_PERIOD <= 0:
        raise ValueError("RSI_PERIOD must be greater than zero.")

    if MIN_CONFIDENCE < 0 or MIN_CONFIDENCE > 100:
        raise ValueError("MIN_CONFIDENCE must be between 0 and 100.")

    if DEFAULT_EXPIRY_MINUTES <= 0:
        raise ValueError("DEFAULT_EXPIRY_MINUTES must be greater than zero.")

    return True
