"""
Configuration for Indian Options Trading Strategy
All tunable parameters in one place. Adjust these based on backtesting results.
"""
from dataclasses import dataclass, field
from typing import Dict

# ──────────────────────────────────────────────
# Angel SmartAPI Credentials (fill before running)
# ──────────────────────────────────────────────
ANGEL_API_KEY = ""
ANGEL_CLIENT_ID = ""
ANGEL_PASSWORD = ""
ANGEL_TOTP_SECRET = ""  # For auto TOTP generation

# ──────────────────────────────────────────────
# Capital & Risk
# ──────────────────────────────────────────────
INITIAL_CAPITAL = 50_000
RISK_PER_TRADE_PCT = 0.03          # 3% per trade = ₹1500
MAX_DAILY_LOSS_PCT = 0.05          # 5% daily stop = ₹2500
MAX_OPEN_POSITIONS = 3
MAX_TRADES_PER_DAY = 10            # V3: more setups across instruments + scalps
SLIPPAGE_POINTS = 2                # Expected slippage in index points

# ──────────────────────────────────────────────
# Instrument Configuration
# ──────────────────────────────────────────────
@dataclass
class InstrumentConfig:
    symbol: str
    token: str           # Angel token for data
    exchange: str        # Spot data exchange: "NSE" or "BSE"
    option_exchange: str # Options exchange: "NFO" (NSE) or "BFO" (BSE)
    lot_size: int
    tick_size: float
    expiry_day: str      # "tuesday" / "thursday" etc.
    expiry_type: str     # "weekly" or "monthly"
    strike_gap: int      # Gap between strikes


# ──────────────────────────────────────────────
# INSTRUMENT CONFIGS
# ──────────────────────────────────────────────
# Post-SEBI Nov 2024: Only 1 weekly expiry per exchange.
#   NSE weekly: NIFTY (Tuesday)
#   BSE weekly: SENSEX (Thursday)
#   BankNifty/FinNifty/MidcapNifty: monthly only (weekly discontinued)
# Lot sizes updated per SEBI minimum ₹15L contract value.
# ──────────────────────────────────────────────
INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "NIFTY": InstrumentConfig(
        symbol="NIFTY",
        token="99926000",
        exchange="NSE",
        option_exchange="NFO",
        lot_size=75,
        tick_size=0.05,
        expiry_day="tuesday",
        expiry_type="weekly",
        strike_gap=50
    ),
    "BANKNIFTY": InstrumentConfig(
        symbol="BANKNIFTY",
        token="99926009",
        exchange="NSE",
        option_exchange="NFO",
        lot_size=30,
        tick_size=0.05,
        expiry_day="thursday",     # Monthly expiry: last Thursday
        expiry_type="monthly",     # NO weekly after Nov 2024
        strike_gap=100
    ),
    "SENSEX": InstrumentConfig(
        symbol="SENSEX",
        token="99919000",
        exchange="BSE",
        option_exchange="BFO",     # BSE F&O segment (not NFO!)
        lot_size=20,
        tick_size=0.05,
        expiry_day="thursday",
        expiry_type="weekly",
        strike_gap=100
    ),
}

# Default instrument to trade
DEFAULT_INSTRUMENT = "NIFTY"  # Weekly expiry = most liquid & best gamma edge


# ──────────────────────────────────────────────
# Holiday Calendar & Expiry Shift Logic
# ──────────────────────────────────────────────
# When a scheduled expiry day is a market holiday, expiry shifts to the
# PREVIOUS trading day. Update this list at the start of each year.
# Source: NSE/BSE holiday circulars.
# Format: "YYYY-MM-DD"
# ──────────────────────────────────────────────
NSE_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-10",  # Maha Shivaratri
    "2026-03-17",  # Holi
    "2026-03-31",  # Id-ul-Fitr (Eid)
    "2026-04-02",  # Ram Navami
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-25",  # Buddha Purnima
    "2026-06-07",  # Bakri Id (Eid ul-Adha)
    "2026-07-07",  # Muharram
    "2026-08-15",  # Independence Day
    "2026-08-16",  # Janmashtami (adjust per NSE if different)
    "2026-09-05",  # Milad-un-Nabi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-20",  # Dussehra
    "2026-11-09",  # Diwali (Laxmi Pujan)
    "2026-11-10",  # Diwali Balipratipada
    "2026-11-19",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]

# BSE holidays are usually the same as NSE; add any BSE-specific ones here
BSE_HOLIDAYS_2026 = NSE_HOLIDAYS_2026.copy()

# ──────────────────────────────────────────────
# Regime Detection Parameters
# ──────────────────────────────────────────────
@dataclass
class RegimeParams:
    # EMA
    ema_fast: int = 9
    ema_slow: int = 21
    ema_weight: float = 20.0        # Max ±20 points

    # ADX
    adx_period: int = 14
    adx_strong_threshold: float = 25.0
    adx_weight: float = 20.0        # Max ±20 points

    # RSI
    rsi_period: int = 14
    rsi_ob: float = 70.0
    rsi_os: float = 30.0
    rsi_weight: float = 15.0        # Max ±15 points

    # VWAP
    vwap_weight: float = 15.0       # Max ±15 points

    # Supertrend
    st_period: int = 10
    st_multiplier: float = 3.0
    st_weight: float = 15.0         # Max ±15 points

    # Previous day levels
    pdhl_weight: float = 10.0       # Max ±10 points (reduced to fit OI)

    # OI Analysis (NEW — biggest edge improvement)
    oi_weight: float = 20.0          # Max ±20 points from OI/PCR/MaxPain
    orderbook_weight: float = 10.0   # Max ±10 points from order flow

    # Regime thresholds
    strong_trend_threshold: float = 60.0
    mild_trend_threshold: float = 25.0

REGIME = RegimeParams()

# ──────────────────────────────────────────────
# Volatility Thresholds (India VIX based)
# ──────────────────────────────────────────────
VIX_LOW = 12.0
VIX_NORMAL = 18.0
VIX_HIGH = 25.0

# ──────────────────────────────────────────────
# Strategy Parameters
# ──────────────────────────────────────────────
@dataclass
class MomentumBuyParams:
    """Strategy A - Momentum Option Buy for strong trends"""
    premium_sl_pct: float = 0.30          # 30% of premium as SL
    atr_sl_multiplier: float = 1.5        # OR 1.5x ATR whichever is larger
    trail_with_supertrend: bool = True
    prefer_atm: bool = True               # True=ATM, False=1 OTM
    min_premium: float = 50.0             # Min premium to buy (avoid illiquid)
    max_premium: float = 300.0            # Max premium (capital constraint)
    time_stop_candles: int = 20           # Exit if no movement in 20 5-min candles

@dataclass
class DebitSpreadParams:
    """Strategy B - Debit Spread for mild trends"""
    spread_width_strikes: int = 1         # 1 strike gap (e.g., 200pt for BankNifty)
    sl_pct_of_max_loss: float = 0.50      # Exit at 50% of max loss
    target_pct_of_max_profit: float = 0.65  # Book at 65% of max profit
    rsi_pullback_threshold: float = 5.0    # RSI must turn X points from extreme

@dataclass
class ORBParams:
    """Strategy C - Opening Range Breakout"""
    orb_start: str = "09:15"
    orb_end: str = "09:30"
    max_range_pct: float = 0.50            # Max 0.5% range for valid ORB
    breakout_buffer_pct: float = 0.02      # 2% above/below range for entry
    target_multiplier: float = 1.5         # 1.5x range as target
    sl_at_opposite_end: bool = True
    volume_confirmation: bool = True       # Need above-avg volume on breakout

@dataclass
class MeanReversionParams:
    """Strategy D - Mean Reversion Scalp in sideways"""
    rsi_extreme_low: float = 25.0
    rsi_extreme_high: float = 75.0
    premium_sl_pct: float = 0.25           # Tight 25% SL (quick scalp)
    target_pct: float = 0.50               # 50% profit target
    max_hold_candles: int = 12             # Max 1 hour hold (12 x 5min)

MOMENTUM = MomentumBuyParams()
SPREAD = DebitSpreadParams()
ORB = ORBParams()
MEAN_REV = MeanReversionParams()

# ──────────────────────────────────────────────
# Scoring System (replaces hard filters)
# ──────────────────────────────────────────────
# Instead of: IF ema_cross AND rsi_ok AND adx_ok AND vwap_ok → trade
# We use:     score = w1*ema + w2*adx + w3*rsi + w4*oi + ... → trade if score > threshold
ENTRY_SCORE_THRESHOLD = 55.0       # Out of 100. Lower = more trades but lower quality
MIN_SCORE_FOR_MOMENTUM = 60.0      # Need higher conviction for directional buys
MIN_SCORE_FOR_SPREAD = 40.0        # Spreads need less conviction (defined risk)
NO_TRADE_RELAXATION_HOUR = 11      # If no trades by 11 AM, reduce threshold by 10

# ──────────────────────────────────────────────
# OI Analysis Configuration
# ──────────────────────────────────────────────
OI_CHAIN_STRIKES_EACH_SIDE = 10    # Fetch 10 strikes above + below ATM
OI_REFRESH_INTERVAL_SEC = 60       # Refresh OI chain every 60 sec (API limit)
OI_BOOST_ON_EXPIRY_DAY = 1.5       # OI signals 1.5x weight on expiry days
OI_BOOST_ON_MONTHLY_EXPIRY = 2.0   # Monthly expiry has even stronger OI gravity
PCR_EXTREME_HIGH = 1.5             # Above this = extreme bullish (contrarian)
PCR_EXTREME_LOW = 0.6              # Below this = extreme bearish (contrarian)
MAX_PAIN_DISTANCE_THRESHOLD = 0.5  # Max pain signal strongest within 0.5% of spot

# ──────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────
MARKET_OPEN = "09:15"
NO_TRADE_BEFORE = "09:30"          # Skip first 15 min
PRIMARY_WINDOW_END = "11:30"       # Best trading window
SECONDARY_WINDOW_START = "13:00"   # Afternoon session
CLOSE_ALL_BY = "15:15"             # Mandate closure
MARKET_CLOSE = "15:30"

# ──────────────────────────────────────────────
# Order Management
# ──────────────────────────────────────────────
LIMIT_ORDER_BUFFER = 2.0           # Place limit at ask + 2 for buys (covers bid-ask spread)
FILL_TIMEOUT_SEC = 10              # Wait 10 sec then modify (5s too short in volatile markets)
ORDER_RETRY_COUNT = 2              # Max retries on order failure
MIN_LOT_MULTIPLIER = 1             # Trade minimum 1 lot

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
CANDLE_INTERVAL_PRIMARY = "FIVE_MINUTE"
CANDLE_INTERVAL_SECONDARY = "FIFTEEN_MINUTE"
DATA_LOOKBACK_DAYS = 5             # Days of historical data to fetch
REFRESH_INTERVAL_SEC = 5           # How often to refresh data (5 sec)

# ──────────────────────────────────────────────
# Logging & Persistence
# ──────────────────────────────────────────────
LOG_DIR = "logs"
TRADE_LOG_FILE = "logs/trades.csv"
DAILY_PNL_FILE = "logs/daily_pnl.csv"
LOG_LEVEL = "INFO"
