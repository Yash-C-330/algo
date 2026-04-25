"""
Strategy implementations — each strategy is a self-contained class that:
1. Checks if conditions are met (returns a score, not boolean)
2. Generates an entry signal with strike, SL, target
3. Manages the position (trailing SL, time stop, etc.)

Scoring > Filtering: A strategy returns a readiness score 0-100.
If score > threshold → execute. No hard gates that block trades.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from indicators import ema, rsi, atr, supertrend, vwap, opening_range
from config import (MOMENTUM, SPREAD, ORB, MEAN_REV, REGIME,
                    SLIPPAGE_POINTS, MIN_SCORE_FOR_MOMENTUM,
                    MIN_SCORE_FOR_SPREAD, ENTRY_SCORE_THRESHOLD)


class OptionType(Enum):
    CE = "CE"
    PE = "PE"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class OptionLeg:
    """Single leg of an options trade."""
    strike: float
    option_type: OptionType
    side: OrderSide
    lots: int
    expected_premium: float


@dataclass
class TradeSignal:
    """Complete trade signal with entry, SL, target."""
    strategy_name: str
    legs: list               # List of OptionLeg
    stop_loss_premium: float  # SL in premium terms
    target_premium: float     # Target in premium terms
    trail_type: str           # "supertrend" / "fixed" / "percentage"
    time_stop_candles: int    # Exit after N candles of no movement
    score: float              # Entry quality score 0-100
    reason: str               # Human-readable entry reason


def get_atm_strike(spot_price: float, strike_gap: int) -> float:
    """Round to nearest strike."""
    return round(spot_price / strike_gap) * strike_gap


def get_otm_strike(spot_price: float, strike_gap: int, option_type: OptionType,
                   steps: int = 1) -> float:
    """Get OTM strike N steps away."""
    atm = get_atm_strike(spot_price, strike_gap)
    if option_type == OptionType.CE:
        return atm + steps * strike_gap
    else:
        return atm - steps * strike_gap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY A: MOMENTUM OPTION BUY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MomentumBuyStrategy:
    """
    Buy ATM/slightly OTM option in strong trending market.
    Entry: Pullback to VWAP/EMA21, or breakout continuation.
    SL: Max(30% premium, 1.5x ATR)
    Trail: Supertrend on 5-min
    """

    def __init__(self, params=None):
        self.params = params or MOMENTUM
        self.name = "momentum_buy"

    def evaluate(self, df_5min: pd.DataFrame, regime_score: float,
                 spot_price: float, strike_gap: int) -> Optional[TradeSignal]:
        """
        Score the setup and return TradeSignal if good enough.
        Returns None if score below threshold.
        """
        score = 0.0
        reasons = []

        # ── Directional alignment (40 points max) ──
        direction = 1 if regime_score > 0 else -1
        abs_score = abs(regime_score)
        dir_score = min(abs_score / 100 * 40, 40)
        score += dir_score
        reasons.append(f"regime={regime_score:.0f}")

        # ── Pullback detection (25 points) ──
        close = df_5min["close"]
        ema21 = ema(close, 21).iloc[-1]
        vwap_series = vwap(df_5min) if "volume" in df_5min.columns else None
        vwap_val = vwap_series.iloc[-1] if vwap_series is not None and not vwap_series.empty and not pd.isna(vwap_series.iloc[-1]) else ema21
        atr_val = atr(df_5min).iloc[-1]

        current = close.iloc[-1]
        prev = close.iloc[-2] if len(close) >= 2 else current

        # Price near VWAP/EMA21 = pullback opportunity
        dist_to_ema = abs(current - ema21) / atr_val if atr_val > 0 else 99
        if dist_to_ema < 0.5:
            score += 25
            reasons.append("pullback_to_ema21")
        elif dist_to_ema < 1.0:
            score += 15
            reasons.append("near_ema21")
        elif dist_to_ema < 1.5:
            score += 8
            reasons.append("moderate_from_ema21")

        # ── Momentum confirmation (20 points) ──
        rsi_val = rsi(close, 14).iloc[-1]
        if direction == 1 and 45 < rsi_val < 75:
            score += 20
            reasons.append(f"rsi_bullish={rsi_val:.0f}")
        elif direction == -1 and 25 < rsi_val < 55:
            score += 20
            reasons.append(f"rsi_bearish={rsi_val:.0f}")
        elif direction == 1 and rsi_val > 75:
            score += 5  # Overbought but trending — less ideal
            reasons.append(f"rsi_ob={rsi_val:.0f}")
        elif direction == -1 and rsi_val < 25:
            score += 5
            reasons.append(f"rsi_os={rsi_val:.0f}")

        # ── Recent price action (15 points) ──
        st = supertrend(df_5min)
        if st["st_direction"].iloc[-1] == direction:
            score += 15
            reasons.append("supertrend_aligned")

        # ── Check threshold ──
        if score < MIN_SCORE_FOR_MOMENTUM:
            return None

        # ── Build trade signal ──
        option_type = OptionType.CE if direction == 1 else OptionType.PE
        if self.params.prefer_atm:
            strike = get_atm_strike(spot_price, strike_gap)
        else:
            strike = get_otm_strike(spot_price, strike_gap, option_type, 1)

        # SL calculation (in index points, we'll estimate premium SL)
        sl_atr = self.params.atr_sl_multiplier * atr_val

        leg = OptionLeg(
            strike=strike,
            option_type=option_type,
            side=OrderSide.BUY,
            lots=1,  # Risk manager will adjust
            expected_premium=0  # Will be filled by order manager
        )

        return TradeSignal(
            strategy_name=self.name,
            legs=[leg],
            stop_loss_premium=0,  # Will compute after getting actual premium
            target_premium=0,     # Trailing — no fixed target
            trail_type="supertrend",
            time_stop_candles=self.params.time_stop_candles,
            score=round(score, 1),
            reason=" | ".join(reasons)
        )

    def compute_sl_target(self, entry_premium: float, atr_val: float) -> tuple:
        """After fill, compute actual SL and initial target in premium terms."""
        # SL = max of percentage-based and ATR-based
        pct_sl = entry_premium * self.params.premium_sl_pct
        atr_sl = self.params.atr_sl_multiplier * atr_val * 0.5  # Rough delta adjustment
        sl = max(pct_sl, atr_sl) + SLIPPAGE_POINTS  # Add slippage buffer

        return round(sl, 1), None  # No fixed target for trailing strategy


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY B: DEBIT SPREAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DebitSpreadStrategy:
    """
    Bull Call Spread or Bear Put Spread in mild trends.
    Lower cost than naked buy, defined risk.
    Entry: RSI pullback turning in trend direction.
    """

    def __init__(self, params=None):
        self.params = params or SPREAD
        self.name = "debit_spread"

    def evaluate(self, df_5min: pd.DataFrame, regime_score: float,
                 spot_price: float, strike_gap: int) -> Optional[TradeSignal]:
        score = 0.0
        reasons = []

        direction = 1 if regime_score > 0 else -1
        abs_regime = abs(regime_score)

        # ── Directional alignment (35 points) ──
        dir_score = min(abs_regime / 100 * 35, 35)
        score += dir_score
        reasons.append(f"regime={regime_score:.0f}")

        # ── RSI pullback turning (30 points) ──
        close = df_5min["close"]
        rsi_series = rsi(close, 14)
        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2] if len(rsi_series) >= 2 else current_rsi

        rsi_turning = current_rsi - prev_rsi

        if direction == 1 and rsi_turning > self.params.rsi_pullback_threshold:
            score += 30
            reasons.append("rsi_turning_up")
        elif direction == -1 and rsi_turning < -self.params.rsi_pullback_threshold:
            score += 30
            reasons.append("rsi_turning_down")
        elif direction == 1 and rsi_turning > 0:
            score += 15
            reasons.append("rsi_slightly_up")
        elif direction == -1 and rsi_turning < 0:
            score += 15
            reasons.append("rsi_slightly_down")

        # ── EMA alignment (20 points) ──
        ema9 = ema(close, 9).iloc[-1]
        ema21 = ema(close, 21).iloc[-1]
        if direction == 1 and ema9 > ema21:
            score += 20
            reasons.append("ema_bullish")
        elif direction == -1 and ema9 < ema21:
            score += 20
            reasons.append("ema_bearish")

        # ── Volatility not too high (15 points) ──
        # Spreads work better in normal vol — less premium decay risk
        atr_val = atr(df_5min).iloc[-1]
        atr_pct = atr_val / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        if atr_pct < 0.3:
            score += 15
            reasons.append("low_vol_good_for_spread")
        elif atr_pct < 0.5:
            score += 10
            reasons.append("normal_vol")

        if score < MIN_SCORE_FOR_SPREAD:
            return None

        # ── Build spread legs ──
        atm = get_atm_strike(spot_price, strike_gap)

        if direction == 1:
            # Bull Call Spread: Buy ATM CE, Sell OTM CE
            buy_leg = OptionLeg(atm, OptionType.CE, OrderSide.BUY, 1, 0)
            sell_leg = OptionLeg(
                atm + self.params.spread_width_strikes * strike_gap,
                OptionType.CE, OrderSide.SELL, 1, 0
            )
        else:
            # Bear Put Spread: Buy ATM PE, Sell OTM PE
            buy_leg = OptionLeg(atm, OptionType.PE, OrderSide.BUY, 1, 0)
            sell_leg = OptionLeg(
                atm - self.params.spread_width_strikes * strike_gap,
                OptionType.PE, OrderSide.SELL, 1, 0
            )

        return TradeSignal(
            strategy_name=self.name,
            legs=[buy_leg, sell_leg],
            stop_loss_premium=0,  # Computed after fill
            target_premium=0,
            trail_type="fixed",
            time_stop_candles=30,  # ~2.5 hours
            score=round(score, 1),
            reason=" | ".join(reasons)
        )

    def compute_sl_target(self, net_debit: float, max_profit: float) -> tuple:
        """After fill, compute SL and target for the spread."""
        sl = net_debit * self.params.sl_pct_of_max_loss  # Loss already limited to debit
        target = max_profit * self.params.target_pct_of_max_profit
        return round(sl, 1), round(target, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY C: OPENING RANGE BREAKOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ORBStrategy:
    """
    Opening Range Breakout — first 15 minutes.
    Works in any regime. Best when opening range is compressed.
    Entry: Break of 9:15-9:30 high/low.
    SL: Opposite end of range.
    Target: 1.5x range.
    """

    def __init__(self, params=None):
        self.params = params or ORB
        self.name = "orb"

    def evaluate(self, df_5min: pd.DataFrame, regime_score: float,
                 spot_price: float, strike_gap: int) -> Optional[TradeSignal]:
        """Returns signal if ORB breakout detected."""
        or_data = opening_range(df_5min, self.params.orb_start, self.params.orb_end)

        or_high = or_data["or_high"]
        or_low = or_data["or_low"]
        or_range = or_data["or_range"]

        if pd.isna(or_high) or pd.isna(or_low) or or_range <= 0:
            return None

        score = 0.0
        reasons = []

        # ── Range compression check (30 points) ──
        range_pct = or_range / spot_price * 100
        if range_pct < self.params.max_range_pct * 0.5:
            score += 30
            reasons.append(f"tight_or={range_pct:.2f}%")
        elif range_pct < self.params.max_range_pct:
            score += 20
            reasons.append(f"normal_or={range_pct:.2f}%")
        else:
            # Range too wide — ORB less reliable
            score += 5
            reasons.append(f"wide_or={range_pct:.2f}%")

        # ── Breakout detection (40 points) ──
        current = df_5min["close"].iloc[-1]
        buffer = or_range * self.params.breakout_buffer_pct

        if current > or_high + buffer:
            direction = 1
            score += 40
            reasons.append("breakout_high")
        elif current < or_low - buffer:
            direction = -1
            score += 40
            reasons.append("breakout_low")
        else:
            # No breakout yet
            return None

        # ── Regime alignment bonus (15 points) ──
        if (direction == 1 and regime_score > 0) or (direction == -1 and regime_score < 0):
            score += 15
            reasons.append("regime_aligned")
        elif abs(regime_score) < 20:
            score += 8  # Sideways regime — ORB still works
            reasons.append("regime_neutral")

        # ── Volume confirmation (15 points / -15 penalty) ──
        if "volume" in df_5min.columns and len(df_5min) >= 5:
            recent_vol = df_5min["volume"].iloc[-1]
            avg_vol = df_5min["volume"].iloc[-6:-1].mean()
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio >= 1.5:
                    score += 15
                    reasons.append(f"strong_volume_surge_{vol_ratio:.1f}x")
                elif vol_ratio >= 1.3:
                    score += 12
                    reasons.append("volume_surge")
                elif vol_ratio >= 1.0:
                    score += 5
                    reasons.append("above_avg_volume")
                elif vol_ratio >= 0.7:
                    score -= 8  # Breakout on weak volume — likely false
                    reasons.append("LOW_volume_breakout")
                else:
                    score -= 15  # Very thin volume — almost certainly false
                    reasons.append("VERY_LOW_volume_AVOID")
        else:
            score -= 5  # No volume data is suspicious for breakout
            reasons.append("no_volume_data")

        if score < ENTRY_SCORE_THRESHOLD:
            return None

        option_type = OptionType.CE if direction == 1 else OptionType.PE
        strike = get_atm_strike(spot_price, strike_gap)

        sl_points = or_range + SLIPPAGE_POINTS  # SL at opposite end + slippage

        leg = OptionLeg(strike, option_type, OrderSide.BUY, 1, 0)

        return TradeSignal(
            strategy_name=self.name,
            legs=[leg],
            stop_loss_premium=0,  # Compute after fill
            target_premium=0,
            trail_type="fixed",
            time_stop_candles=24,  # ~2 hours
            score=round(score, 1),
            reason=" | ".join(reasons)
        )

    def compute_sl_target(self, entry_premium: float, or_range: float,
                          spot_price: float) -> tuple:
        """
        SL = premium equivalent of opposite end of OR.
        Target = 1.5x range equivalent.
        Rough delta-based estimation.
        """
        # Approximate delta for ATM option ≈ 0.5
        delta = 0.5
        sl_premium = or_range * delta + SLIPPAGE_POINTS
        target_premium = or_range * self.params.target_multiplier * delta

        # Clamp SL to percentage of premium
        max_sl = entry_premium * 0.35
        sl_premium = min(sl_premium, max_sl)

        return round(sl_premium, 1), round(target_premium, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY D: MEAN REVERSION SCALP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MeanReversionStrategy:
    """
    In sideways markets, fade extremes at support/resistance.
    Entry: RSI extreme + price at key level (PDH/PDL/VWAP).
    Quick scalp with tight SL.
    """

    def __init__(self, params=None):
        self.params = params or MEAN_REV
        self.name = "mean_reversion"

    def evaluate(self, df_5min: pd.DataFrame, regime_score: float,
                 spot_price: float, strike_gap: int,
                 pdh: float = None, pdl: float = None) -> Optional[TradeSignal]:
        score = 0.0
        reasons = []

        # ── Must be in sideways regime (25 points) ──
        if abs(regime_score) < 25:
            score += 25
            reasons.append("sideways_regime")
        elif abs(regime_score) < 40:
            score += 10
            reasons.append("weak_trend")
        else:
            return None  # Don't mean-revert in trending markets

        # ── RSI extreme (35 points) ──
        close = df_5min["close"]
        rsi_val = rsi(close, 14).iloc[-1]

        if rsi_val <= self.params.rsi_extreme_low:
            direction = 1  # Oversold → buy CE
            score += 35
            reasons.append(f"rsi_oversold={rsi_val:.0f}")
        elif rsi_val >= self.params.rsi_extreme_high:
            direction = -1  # Overbought → buy PE
            score += 35
            reasons.append(f"rsi_overbought={rsi_val:.0f}")
        else:
            return None  # Need extreme RSI for mean reversion

        # ── Price at key level (25 points) ──
        atr_val = atr(df_5min).iloc[-1]
        at_level = False

        if pdh is not None and pdl is not None:
            if direction == 1 and abs(spot_price - pdl) < atr_val:
                score += 25
                reasons.append("at_PDL_support")
                at_level = True
            elif direction == -1 and abs(spot_price - pdh) < atr_val:
                score += 25
                reasons.append("at_PDH_resistance")
                at_level = True

        # VWAP as level
        if not at_level and "volume" in df_5min.columns:
            vwap_series = vwap(df_5min)
            if not vwap_series.empty:
                vwap_val = vwap_series.iloc[-1]
                if not pd.isna(vwap_val) and abs(spot_price - vwap_val) < atr_val * 0.5:
                    score += 20
                    reasons.append("at_VWAP")

        # ── RSI showing reversal (15 points) ──
        rsi_series = rsi(close, 14)
        if len(rsi_series) >= 2:
            rsi_delta = rsi_series.iloc[-1] - rsi_series.iloc[-2]
            if direction == 1 and rsi_delta > 2:
                score += 15
                reasons.append("rsi_turning_up")
            elif direction == -1 and rsi_delta < -2:
                score += 15
                reasons.append("rsi_turning_down")

        if score < ENTRY_SCORE_THRESHOLD:
            return None

        option_type = OptionType.CE if direction == 1 else OptionType.PE
        strike = get_atm_strike(spot_price, strike_gap)

        leg = OptionLeg(strike, option_type, OrderSide.BUY, 1, 0)

        return TradeSignal(
            strategy_name=self.name,
            legs=[leg],
            stop_loss_premium=0,
            target_premium=0,
            trail_type="fixed",
            time_stop_candles=self.params.max_hold_candles,
            score=round(score, 1),
            reason=" | ".join(reasons)
        )

    def compute_sl_target(self, entry_premium: float) -> tuple:
        sl = entry_premium * self.params.premium_sl_pct + SLIPPAGE_POINTS
        target = entry_premium * self.params.target_pct
        return round(sl, 1), round(target, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRATEGY SELECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StrategySelector:
    """
    Given a regime state, evaluates all recommended strategies
    and returns the best signal (highest score).
    """

    def __init__(self):
        self.strategies = {
            "momentum_buy": MomentumBuyStrategy(),
            "debit_spread": DebitSpreadStrategy(),
            "orb": ORBStrategy(),
            "mean_reversion": MeanReversionStrategy(),
        }

    def select_best(self, df_5min: pd.DataFrame, regime_score: float,
                    recommended: list, spot_price: float, strike_gap: int,
                    pdh: float = None, pdl: float = None) -> Optional[TradeSignal]:
        """
        Evaluate each recommended strategy and return the one with highest score.
        """
        best_signal = None
        best_score = 0

        for strategy_name in recommended:
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue

            if strategy_name == "mean_reversion":
                signal = strategy.evaluate(
                    df_5min, regime_score, spot_price, strike_gap, pdh, pdl
                )
            else:
                signal = strategy.evaluate(
                    df_5min, regime_score, spot_price, strike_gap
                )

            if signal and signal.score > best_score:
                best_signal = signal
                best_score = signal.score

        return best_signal
