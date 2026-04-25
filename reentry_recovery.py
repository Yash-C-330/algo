"""
Re-Entry & Recovery Logic

The #1 mistake retail traders make: Getting stopped out and NEVER re-entering.

Reality: 60% of the time, the original trade thesis was RIGHT. You just
got stopped out by noise (a wick hunt, a brief spike). A professional
trader re-enters when the thesis is CONFIRMED again.

Re-Entry Rules (NOT revenge trading — these are strict):

1. REGIME MUST STILL HOLD — If you were bullish and got stopped out,
   the regime must still be bullish or stronger
2. COOLING PERIOD — Wait at least 2 candles (10 min) after stop out
3. PULLBACK REQUIRED — Don't re-enter at any price. Wait for a pullback
   to a key level (VWAP, EMA21, support)
4. REDUCED SIZE — First re-entry at 75% size, second at 50%
5. MAX 2 RE-ENTRIES — After 2 failed re-entries on same thesis, STOP
6. WIDER SL on re-entry — Use 1.3x the original SL (the "wick" area is
   now known, set SL below it)

Recovery Logic (after a losing streak):

When you've had 2+ consecutive losses:
- Reduce position size by 50%
- Increase score threshold by 15 (be pickier)
- Only trade A+ setups (scored >= 80)
- After 2 consecutive wins, restore normal size
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class StoppedTrade:
    """Record of a trade that was stopped out."""
    instrument: str
    direction: int               # +1 bullish, -1 bearish
    strategy_name: str
    regime_score: float          # Regime score when stopped
    stop_time: datetime
    stop_price: float            # Index price at stop
    sl_price: float              # SL that was hit
    entry_price: float           # Original entry
    loss_amount: float
    re_entries_made: int = 0
    re_entry_allowed: bool = True


@dataclass
class ReEntrySignal:
    """Signal to re-enter a previously stopped-out trade."""
    original_trade: StoppedTrade
    re_entry_price: float
    new_sl_price: float          # Wider SL
    size_multiplier: float       # 0.75 first re-entry, 0.50 second
    reason: str
    confidence: float


class ReEntryManager:
    """
    Manages re-entry logic after stop-outs.

    Flow:
    1. When a trade is stopped out, register it via record_stop_out()
    2. Every candle, call check_reentry() to see if conditions are met
    3. If re-entry signal returned, main engine executes at reduced size
    """

    MAX_REENTRIES = 2
    COOLING_CANDLES = 2           # Wait 2 candles (10 min on 5min chart)
    FIRST_REENTRY_SIZE = 0.75
    SECOND_REENTRY_SIZE = 0.50
    SL_WIDEN_FACTOR = 1.3

    def __init__(self):
        self.stopped_trades: List[StoppedTrade] = []

    def reset_daily(self):
        """Clear at start of each day."""
        self.stopped_trades = []

    def record_stop_out(self, instrument: str, direction: int,
                        strategy_name: str, regime_score: float,
                        stop_time: datetime, stop_price: float,
                        sl_price: float, entry_price: float,
                        loss_amount: float):
        """Register a stopped-out trade for potential re-entry."""
        trade = StoppedTrade(
            instrument=instrument,
            direction=direction,
            strategy_name=strategy_name,
            regime_score=regime_score,
            stop_time=stop_time,
            stop_price=stop_price,
            sl_price=sl_price,
            entry_price=entry_price,
            loss_amount=loss_amount,
        )
        self.stopped_trades.append(trade)
        logger.info(f"Recorded stop-out: {instrument} {strategy_name} "
                     f"direction={direction} loss={loss_amount:.0f}")

    def check_reentry(self, instrument: str, current_time: datetime,
                       current_price: float, current_regime_score: float,
                       vwap_price: float = None,
                       ema21_price: float = None,
                       support_level: float = None,
                       resistance_level: float = None) -> Optional[ReEntrySignal]:
        """
        Check if any stopped trade qualifies for re-entry.

        Conditions:
        1. Same instrument
        2. Cooling period elapsed
        3. Regime still confirms original direction
        4. Price at a key support/resistance (pullback confirmation)
        5. Max re-entries not exceeded
        """
        for trade in self.stopped_trades:
            if trade.instrument != instrument:
                continue

            if not trade.re_entry_allowed:
                continue

            if trade.re_entries_made >= self.MAX_REENTRIES:
                trade.re_entry_allowed = False
                continue

            # Check cooling period (2 candles = 10 min)
            elapsed = (current_time - trade.stop_time).total_seconds() / 60
            if elapsed < self.COOLING_CANDLES * 5:
                continue

            # Regime must confirm original direction
            if trade.direction > 0 and current_regime_score < 20:
                continue  # Was bullish, regime no longer bullish
            if trade.direction < 0 and current_regime_score > -20:
                continue  # Was bearish, regime no longer bearish

            # Regime should be at least as strong as when trade was taken
            if abs(current_regime_score) < abs(trade.regime_score) * 0.7:
                continue  # Regime weakened too much

            # Check pullback to key level
            pullback_confirmed = False
            pullback_reason = ""

            if trade.direction > 0:
                # Bullish: need price to pull back to support
                if vwap_price and current_price <= vwap_price * 1.002:
                    pullback_confirmed = True
                    pullback_reason = "price at VWAP support"
                elif ema21_price and current_price <= ema21_price * 1.003:
                    pullback_confirmed = True
                    pullback_reason = "price at EMA21 support"
                elif support_level and current_price <= support_level * 1.005:
                    pullback_confirmed = True
                    pullback_reason = "price at key support"
            else:
                # Bearish: need price to rally to resistance
                if vwap_price and current_price >= vwap_price * 0.998:
                    pullback_confirmed = True
                    pullback_reason = "price at VWAP resistance"
                elif ema21_price and current_price >= ema21_price * 0.997:
                    pullback_confirmed = True
                    pullback_reason = "price at EMA21 resistance"
                elif resistance_level and current_price >= resistance_level * 0.995:
                    pullback_confirmed = True
                    pullback_reason = "price at key resistance"

            if not pullback_confirmed:
                continue

            # Calculate re-entry parameters
            size_mult = (self.FIRST_REENTRY_SIZE if trade.re_entries_made == 0
                         else self.SECOND_REENTRY_SIZE)

            # Wider SL — set beyond the original SL that got hit
            original_sl_distance = abs(trade.entry_price - trade.sl_price)
            wider_sl_distance = original_sl_distance * self.SL_WIDEN_FACTOR

            if trade.direction > 0:
                new_sl = current_price - wider_sl_distance
            else:
                new_sl = current_price + wider_sl_distance

            signal = ReEntrySignal(
                original_trade=trade,
                re_entry_price=current_price,
                new_sl_price=new_sl,
                size_multiplier=size_mult,
                reason=f"Re-entry #{trade.re_entries_made + 1}: {pullback_reason}, "
                       f"regime={current_regime_score:.0f}",
                confidence=65 - (trade.re_entries_made * 10),  # Decreasing confidence
            )

            # Update trade record
            trade.re_entries_made += 1
            if trade.re_entries_made >= self.MAX_REENTRIES:
                trade.re_entry_allowed = False

            logger.info(f"Re-entry signal: {signal.reason}")
            return signal

        return None


class RecoveryManager:
    """
    Adjusts trading behavior after losing streaks.

    This prevents emotional over-trading and protects capital
    during unfavorable conditions.
    """

    def __init__(self):
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.in_recovery_mode = False
        self.total_trades_today = 0
        self.total_pnl_today = 0.0

    def reset_daily(self):
        """Start fresh each day (losing streaks don't carry over)."""
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.in_recovery_mode = False
        self.total_trades_today = 0
        self.total_pnl_today = 0.0

    def record_trade_result(self, pnl: float):
        """Record P&L of completed trade."""
        self.total_trades_today += 1
        self.total_pnl_today += pnl

        if pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0

        # Enter recovery mode after 2 consecutive losses
        if self.consecutive_losses >= 2:
            if not self.in_recovery_mode:
                logger.warning(f"ENTERING RECOVERY MODE after {self.consecutive_losses} consecutive losses")
            self.in_recovery_mode = True

        # Exit recovery mode after 2 consecutive wins
        if self.in_recovery_mode and self.consecutive_wins >= 2:
            logger.info("EXITING RECOVERY MODE — 2 consecutive wins")
            self.in_recovery_mode = False

    def get_size_multiplier(self) -> float:
        """
        Position size multiplier.
        1.0 = normal, 0.5 = recovery mode (half size).
        """
        if self.in_recovery_mode:
            return 0.5
        return 1.0

    def get_threshold_adjustment(self) -> int:
        """
        Score threshold increase during recovery.
        Higher threshold = only take best setups.
        """
        if self.in_recovery_mode:
            return 15  # Add 15 to threshold
        return 0

    def get_min_score_override(self) -> Optional[int]:
        """
        In recovery mode, minimum score for any trade.
        None = use normal threshold.
        """
        if self.in_recovery_mode:
            return 80  # Only A+ setups
        return None

    def should_stop_trading(self, capital: float) -> Tuple[bool, str]:
        """
        Emergency stop conditions.
        """
        # 4+ consecutive losses = stop for the day
        if self.consecutive_losses >= 4:
            return True, "4 consecutive losses — stopping for the day"

        # Daily loss limit (5% of capital)
        if capital > 0 and self.total_pnl_today < 0:
            loss_pct = abs(self.total_pnl_today) / capital * 100
            if loss_pct >= 5.0:
                return True, f"Daily loss limit hit ({loss_pct:.1f}%)"

        return False, "OK"

