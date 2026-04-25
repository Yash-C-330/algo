"""
Time-Session Strategy Router

Markets behave DIFFERENTLY at different times. This is one of the biggest
edges that retail traders miss. They apply the SAME strategy all day long.

Session Map (IST, based on 10+ years of Indian market behavior):

┌──────────────────────────────────────────────────────────────────┐
│ 09:15-09:30  │ DO NOTHING — Wait for noise to settle            │
│ 09:30-10:30  │ OPENING RANGE — ORB + First Pullback (best edge) │
│ 10:30-11:00  │ CONFIRMATION — Trend establishes, add to winners │
│ 11:00-12:30  │ TREND FOLLOWING — If trending, ride it. Scalp.   │
│ 12:30-13:30  │ LUNCH LULL — Low volume, mean reversion only     │
│ 13:30-14:30  │ AFTERNOON SURGE — Global markets open influence  │
│ 14:30-15:00  │ CLOSING PLAY — Final push, expiry adjustments    │
│ 15:00-15:30  │ EXIT ONLY — Close positions, no new entries       │
└──────────────────────────────────────────────────────────────────┘

Why this matters:
- ORB in first hour catches 70% of daily range
- Lunch lull has lowest win rate for trend strategies
- Afternoon session brings fresh volume from global cues
- Last 30 min = theta acceleration on expiry day

Each session has different:
1. Strategy weights (which strategies are active)
2. Score thresholds (how confident to be)
3. Target multipliers (how far to aim)
4. Risk tolerance (tighter/wider SL)
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, time
from enum import Enum

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    PRE_MARKET = "pre_market"           # 09:15-09:30
    OPENING_RANGE = "opening_range"     # 09:30-10:30
    CONFIRMATION = "confirmation"       # 10:30-11:00
    MIDDAY_TREND = "midday_trend"       # 11:00-12:30
    LUNCH_LULL = "lunch_lull"           # 12:30-13:30
    AFTERNOON_SURGE = "afternoon_surge" # 13:30-14:30
    CLOSING_PLAY = "closing_play"       # 14:30-15:00
    EXIT_ONLY = "exit_only"             # 15:00-15:30


@dataclass
class SessionConfig:
    """Configuration for each trading session."""
    session: TradingSession
    start: time
    end: time

    # Strategy activation (which strategies are allowed)
    allow_orb: bool = False
    allow_momentum: bool = False
    allow_debit_spread: bool = False
    allow_mean_reversion: bool = False
    allow_scalping: bool = False
    allow_micro_patterns: bool = False
    allow_new_entries: bool = True

    # Score threshold adjustment
    score_threshold_modifier: int = 0     # Add to base threshold (+ = stricter)

    # Target/Risk modifications
    target_multiplier: float = 1.0         # 1.0 = normal, 0.8 = reduced targets
    sl_multiplier: float = 1.0             # 1.0 = normal, 0.8 = tighter SL
    max_trades_in_session: int = 3

    # Behavior flags
    trail_aggressively: bool = False       # Tighter trailing SL
    move_to_breakeven_fast: bool = False   # Move SL to entry faster
    prefer_quick_exits: bool = False       # Exit partial at first target


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SESSION_MAP: Dict[TradingSession, SessionConfig] = {
    TradingSession.PRE_MARKET: SessionConfig(
        session=TradingSession.PRE_MARKET,
        start=time(9, 15), end=time(9, 30),
        allow_new_entries=False,  # Just watch
        max_trades_in_session=0,
    ),
    TradingSession.OPENING_RANGE: SessionConfig(
        session=TradingSession.OPENING_RANGE,
        start=time(9, 30), end=time(10, 30),
        allow_orb=True,
        allow_momentum=True,
        allow_micro_patterns=True,
        allow_scalping=True,
        score_threshold_modifier=-5,         # Slightly easier entry (best edge time)
        target_multiplier=1.2,               # Aim higher (moves are bigger)
        sl_multiplier=1.1,                   # Slightly wider SL for volatility
        max_trades_in_session=3,
    ),
    TradingSession.CONFIRMATION: SessionConfig(
        session=TradingSession.CONFIRMATION,
        start=time(10, 30), end=time(11, 0),
        allow_momentum=True,
        allow_debit_spread=True,
        allow_micro_patterns=True,
        allow_scalping=True,
        target_multiplier=1.0,
        max_trades_in_session=2,
    ),
    TradingSession.MIDDAY_TREND: SessionConfig(
        session=TradingSession.MIDDAY_TREND,
        start=time(11, 0), end=time(12, 30),
        allow_momentum=True,
        allow_debit_spread=True,
        allow_micro_patterns=True,
        allow_scalping=True,
        allow_mean_reversion=True,
        target_multiplier=1.0,
        max_trades_in_session=3,
    ),
    TradingSession.LUNCH_LULL: SessionConfig(
        session=TradingSession.LUNCH_LULL,
        start=time(12, 30), end=time(13, 30),
        allow_mean_reversion=True,           # Only reversion works in chop
        allow_scalping=True,                 # Quick scalps still OK
        score_threshold_modifier=10,         # HIGHER threshold (worse conditions)
        target_multiplier=0.7,               # Reduced targets (smaller moves)
        sl_multiplier=0.8,                   # Tighter SL (chop kills)
        max_trades_in_session=2,
        prefer_quick_exits=True,
    ),
    TradingSession.AFTERNOON_SURGE: SessionConfig(
        session=TradingSession.AFTERNOON_SURGE,
        start=time(13, 30), end=time(14, 30),
        allow_momentum=True,
        allow_debit_spread=True,
        allow_micro_patterns=True,
        allow_scalping=True,
        target_multiplier=1.0,
        max_trades_in_session=3,
    ),
    TradingSession.CLOSING_PLAY: SessionConfig(
        session=TradingSession.CLOSING_PLAY,
        start=time(14, 30), end=time(15, 0),
        allow_momentum=False,                # Too late for directional bets
        allow_scalping=True,                 # Quick scalps still OK
        allow_micro_patterns=False,          # Patterns need time to play out
        allow_new_entries=False,             # Block new entries — manage existing only
        score_threshold_modifier=10,         # Extra careful near close
        target_multiplier=0.6,               # Small targets (less time)
        sl_multiplier=0.7,                   # Tighter SL
        max_trades_in_session=1,
        trail_aggressively=True,
        move_to_breakeven_fast=True,
    ),
    TradingSession.EXIT_ONLY: SessionConfig(
        session=TradingSession.EXIT_ONLY,
        start=time(15, 0), end=time(15, 30),
        allow_new_entries=False,
        max_trades_in_session=0,
        trail_aggressively=True,
    ),
}


# Expiry day overrides
EXPIRY_SESSION_OVERRIDES: Dict[TradingSession, dict] = {
    TradingSession.CLOSING_PLAY: {
        "allow_scalping": True,
        "score_threshold_modifier": -5,    # More aggressive on expiry close
        "target_multiplier": 1.5,          # Gamma makes targets achievable
        "max_trades_in_session": 4,
    },
    TradingSession.LUNCH_LULL: {
        "allow_momentum": True,            # Expiry days trend through lunch
        "score_threshold_modifier": 5,
        "target_multiplier": 0.9,
    },
}


class SessionRouter:
    """
    Determines current session and returns active configuration.

    Usage:
        router = SessionRouter()
        config = router.get_current_config(now, is_expiry_day=True)
        if config.allow_momentum and config.allow_new_entries:
            # evaluate momentum strategy
            adjusted_threshold = base_threshold + config.score_threshold_modifier
    """

    def __init__(self):
        self.session_trades: Dict[TradingSession, int] = {}

    def reset_daily(self):
        """Call at start of each trading day."""
        self.session_trades = {}

    def get_current_session(self, current_time: datetime) -> TradingSession:
        """Determine which session we're in."""
        t = current_time.time()
        for session, config in SESSION_MAP.items():
            if config.start <= t < config.end:
                return session

        # Before market or after hours
        if t < time(9, 15):
            return TradingSession.PRE_MARKET
        return TradingSession.EXIT_ONLY

    def get_current_config(self, current_time: datetime,
                            is_expiry_day: bool = False) -> SessionConfig:
        """
        Get session configuration, with expiry day overrides applied.
        """
        session = self.get_current_session(current_time)
        config = SESSION_MAP[session]

        # Apply expiry overrides
        if is_expiry_day and session in EXPIRY_SESSION_OVERRIDES:
            overrides = EXPIRY_SESSION_OVERRIDES[session]
            # Create modified config (don't mutate the original)
            config = SessionConfig(
                session=config.session,
                start=config.start,
                end=config.end,
                allow_orb=overrides.get("allow_orb", config.allow_orb),
                allow_momentum=overrides.get("allow_momentum", config.allow_momentum),
                allow_debit_spread=overrides.get("allow_debit_spread", config.allow_debit_spread),
                allow_mean_reversion=overrides.get("allow_mean_reversion", config.allow_mean_reversion),
                allow_scalping=overrides.get("allow_scalping", config.allow_scalping),
                allow_micro_patterns=overrides.get("allow_micro_patterns", config.allow_micro_patterns),
                allow_new_entries=overrides.get("allow_new_entries", config.allow_new_entries),
                score_threshold_modifier=overrides.get("score_threshold_modifier", config.score_threshold_modifier),
                target_multiplier=overrides.get("target_multiplier", config.target_multiplier),
                sl_multiplier=overrides.get("sl_multiplier", config.sl_multiplier),
                max_trades_in_session=overrides.get("max_trades_in_session", config.max_trades_in_session),
                trail_aggressively=overrides.get("trail_aggressively", config.trail_aggressively),
                move_to_breakeven_fast=overrides.get("move_to_breakeven_fast", config.move_to_breakeven_fast),
                prefer_quick_exits=overrides.get("prefer_quick_exits", config.prefer_quick_exits),
            )

        return config

    def can_trade_in_session(self, current_time: datetime,
                              is_expiry_day: bool = False) -> Tuple[bool, str]:
        """
        Check if we can take new trades in current session.
        Note: This checks general entry permission. Scalps and patterns
        have their own allow_scalping / allow_micro_patterns flags that
        are checked separately in the main loop.
        """
        config = self.get_current_config(current_time, is_expiry_day)

        session = config.session
        trades_taken = self.session_trades.get(session, 0)
        if trades_taken >= config.max_trades_in_session:
            return False, f"Session {config.session.value}: max trades ({config.max_trades_in_session}) reached"

        return True, f"Session {config.session.value}: OK"

    def record_trade(self, current_time: datetime):
        """Record a trade taken in current session."""
        session = self.get_current_session(current_time)
        self.session_trades[session] = self.session_trades.get(session, 0) + 1

    def get_allowed_strategies(self, current_time: datetime,
                                is_expiry_day: bool = False) -> List[str]:
        """Get list of strategy names allowed in current session."""
        config = self.get_current_config(current_time, is_expiry_day)
        allowed = []
        if config.allow_orb:
            allowed.append("orb")
        if config.allow_momentum:
            allowed.append("momentum_buy")
        if config.allow_debit_spread:
            allowed.append("debit_spread")
        if config.allow_mean_reversion:
            allowed.append("mean_reversion")
        if config.allow_scalping:
            allowed.append("scalping")
        if config.allow_micro_patterns:
            allowed.append("micro_patterns")
        return allowed

    def adjust_threshold(self, base_threshold: int,
                          current_time: datetime,
                          is_expiry_day: bool = False) -> int:
        """Adjust strategy entry threshold for current session."""
        config = self.get_current_config(current_time, is_expiry_day)
        return base_threshold + config.score_threshold_modifier

    def adjust_target(self, base_target: float,
                       current_time: datetime,
                       is_expiry_day: bool = False) -> float:
        """Adjust target price for current session."""
        config = self.get_current_config(current_time, is_expiry_day)
        return base_target * config.target_multiplier

    def adjust_sl(self, base_sl: float,
                   current_time: datetime,
                   is_expiry_day: bool = False) -> float:
        """Adjust stop loss for current session."""
        config = self.get_current_config(current_time, is_expiry_day)
        return base_sl * config.sl_multiplier
