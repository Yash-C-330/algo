"""
Market Regime Detection Engine

Uses a SCORING system instead of hard filters.
Each indicator contributes a weighted score (-max to +max).
Total score determines regime: Strong Trend / Mild Trend / Sideways.

This solves the "no trades" problem — we never need ALL indicators to agree.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from indicators import ema, rsi, adx, supertrend, vwap, atr, bollinger_bands
from config import REGIME, VIX_LOW, VIX_NORMAL, VIX_HIGH
from oi_analyzer import OISignal


class MarketRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    MILD_UPTREND = "mild_uptrend"
    SIDEWAYS = "sideways"
    MILD_DOWNTREND = "mild_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class RegimeState:
    """Complete snapshot of current market regime."""
    regime: MarketRegime
    volatility: VolatilityRegime
    total_score: float                # -100 to +100
    confidence: float                 # 0 to 1 (how aligned are indicators)
    component_scores: dict            # Individual indicator contributions
    recommended_strategies: list      # Ordered list of suitable strategies
    timestamp: Optional[pd.Timestamp] = None

    @property
    def is_bullish(self) -> bool:
        return self.total_score > 0

    @property
    def is_strong_trend(self) -> bool:
        return abs(self.total_score) >= REGIME.strong_trend_threshold

    @property
    def is_sideways(self) -> bool:
        return abs(self.total_score) < REGIME.mild_trend_threshold


class RegimeDetector:
    """
    Multi-indicator regime detection engine.

    Key design: SCORING over FILTERING
    - Each indicator scores independently
    - Total score determines regime
    - No single indicator can block a trade
    - Confidence = how many indicators agree on direction
    """

    def __init__(self, params=None):
        self.params = params or REGIME

    def detect(self, df_5min: pd.DataFrame, df_15min: pd.DataFrame,
               prev_day_high: float, prev_day_low: float, prev_day_close: float,
               india_vix: float = 15.0,
               oi_signal: Optional[OISignal] = None,
               orderbook_signal: Optional[dict] = None) -> RegimeState:
        """
        Detect current market regime from multi-timeframe data + OI + order flow.

        Args:
            df_5min: 5-minute OHLCV DataFrame (at least 50 candles)
            df_15min: 15-minute OHLCV DataFrame (at least 30 candles)
            prev_day_high/low/close: Previous day's levels
            india_vix: Current India VIX value
            oi_signal: OI analysis signal (from OIAnalyzer) — NEW
            orderbook_signal: Order book depth signal — NEW

        Returns:
            RegimeState with all scoring details
        """
        scores = {}

        # 1. EMA Crossover Score (5-min and 15-min)
        scores["ema_5min"] = self._score_ema(df_5min)
        scores["ema_15min"] = self._score_ema(df_15min)
        # Average of both timeframes, weighted
        ema_score = (scores["ema_5min"] * 0.4 + scores["ema_15min"] * 0.6)
        ema_score = np.clip(ema_score, -self.params.ema_weight, self.params.ema_weight)
        scores["ema_combined"] = ema_score

        # 2. ADX Score (15-min for trend strength)
        scores["adx"] = self._score_adx(df_15min)

        # 3. RSI Score (5-min)
        scores["rsi"] = self._score_rsi(df_5min)

        # 4. VWAP Score (5-min)
        scores["vwap"] = self._score_vwap(df_5min)

        # 5. Supertrend Score (5-min)
        scores["supertrend"] = self._score_supertrend(df_5min)

        # 6. Previous Day Levels Score
        scores["prev_day"] = self._score_prev_day_levels(
            df_5min, prev_day_high, prev_day_low, prev_day_close
        )

        # 7. OI Analysis Score (NEW — institutional positioning)
        scores["oi"] = self._score_oi(oi_signal)

        # 8. Order Book / Depth Score (NEW — real-time flow)
        scores["orderbook"] = self._score_orderbook(orderbook_signal)

        # Total score (now includes OI and order flow)
        total = (scores["ema_combined"] + scores["adx"] + scores["rsi"] +
                 scores["vwap"] + scores["supertrend"] + scores["prev_day"] +
                 scores["oi"] + scores["orderbook"])
        total = np.clip(total, -100, 100)

        # Confidence: what fraction of indicators agree on direction
        directions = [
            np.sign(scores["ema_combined"]),
            np.sign(scores["adx"]),
            np.sign(scores["rsi"]),
            np.sign(scores["vwap"]),
            np.sign(scores["supertrend"]),
            np.sign(scores["prev_day"]),
            np.sign(scores["oi"]),
            np.sign(scores["orderbook"])
        ]
        non_zero = [d for d in directions if d != 0]
        if non_zero:
            majority = np.sign(sum(non_zero))
            confidence = sum(1 for d in non_zero if d == majority) / len(non_zero)
        else:
            confidence = 0.0

        # Determine regime
        regime = self._classify_regime(total)

        # Determine volatility regime
        vol_regime = self._classify_volatility(india_vix)

        # Recommend strategies
        strategies = self._recommend_strategies(regime, vol_regime, confidence)

        return RegimeState(
            regime=regime,
            volatility=vol_regime,
            total_score=round(total, 1),
            confidence=round(confidence, 2),
            component_scores={k: round(v, 1) for k, v in scores.items()},
            recommended_strategies=strategies,
            timestamp=df_5min.index[-1] if not df_5min.empty and isinstance(df_5min.index, pd.DatetimeIndex) else None
        )

    # ──────────────────────────────────────────────
    # Individual Indicator Scoring
    # ──────────────────────────────────────────────

    def _score_ema(self, df: pd.DataFrame) -> float:
        """Score based on EMA crossover and distance. Range: -20 to +20."""
        close = df["close"]
        ema_fast = ema(close, self.params.ema_fast)
        ema_slow = ema(close, self.params.ema_slow)

        if ema_fast.empty or ema_slow.empty:
            return 0.0

        latest_fast = ema_fast.iloc[-1]
        latest_slow = ema_slow.iloc[-1]
        latest_close = close.iloc[-1]

        # Base: which side of EMA are we?
        if latest_fast > latest_slow:
            base = 1
        elif latest_fast < latest_slow:
            base = -1
        else:
            base = 0

        # Magnitude: how far apart are EMAs (normalized by ATR)
        atr_val = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        if atr_val > 0:
            separation = abs(latest_fast - latest_slow) / atr_val
            magnitude = min(separation / 2.0, 1.0)  # Cap at 1.0
        else:
            magnitude = 0.5

        # Bonus: price above/below both EMAs
        price_bonus = 0
        if latest_close > latest_fast > latest_slow:
            price_bonus = 0.2
        elif latest_close < latest_fast < latest_slow:
            price_bonus = -0.2

        score = base * magnitude * self.params.ema_weight + price_bonus * self.params.ema_weight
        return np.clip(score, -self.params.ema_weight, self.params.ema_weight)

    def _score_adx(self, df: pd.DataFrame) -> float:
        """Score based on ADX strength and DI direction. Range: -20 to +20."""
        adx_df = adx(df, self.params.adx_period)

        if adx_df.empty:
            return 0.0

        adx_val = adx_df["adx"].iloc[-1]
        plus_di = adx_df["plus_di"].iloc[-1]
        minus_di = adx_df["minus_di"].iloc[-1]

        if pd.isna(adx_val):
            return 0.0

        # Direction from DI crossover
        if plus_di > minus_di:
            direction = 1
        elif minus_di > plus_di:
            direction = -1
        else:
            direction = 0

        # Strength from ADX value
        if adx_val >= self.params.adx_strong_threshold:
            strength = 1.0
        elif adx_val >= 20:
            strength = 0.7
        elif adx_val >= 15:
            strength = 0.4
        else:
            strength = 0.1  # Weak trend, low score

        return direction * strength * self.params.adx_weight

    def _score_rsi(self, df: pd.DataFrame) -> float:
        """Score based on RSI momentum. Range: -15 to +15."""
        rsi_val = rsi(df["close"], self.params.rsi_period)

        if rsi_val.empty or pd.isna(rsi_val.iloc[-1]):
            return 0.0

        current_rsi = rsi_val.iloc[-1]

        # Linear mapping: RSI 50 → 0, RSI 70+ → +15, RSI 30- → -15
        if current_rsi >= 50:
            # Bullish zone
            score = ((current_rsi - 50) / 30) * self.params.rsi_weight
        else:
            # Bearish zone
            score = ((current_rsi - 50) / 30) * self.params.rsi_weight

        # Penalize extremes slightly (mean reversion risk)
        if current_rsi > self.params.rsi_ob:
            score *= 0.8  # Slightly reduce in overbought
        elif current_rsi < self.params.rsi_os:
            score *= 0.8  # Slightly reduce in oversold

        return np.clip(score, -self.params.rsi_weight, self.params.rsi_weight)

    def _score_vwap(self, df: pd.DataFrame) -> float:
        """Score based on price position relative to VWAP. Range: -15 to +15."""
        from indicators import vwap as calc_vwap

        if "volume" not in df.columns or df["volume"].sum() == 0:
            return 0.0

        vwap_series = calc_vwap(df)
        if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
            return 0.0

        latest_close = df["close"].iloc[-1]
        latest_vwap = vwap_series.iloc[-1]

        # Distance from VWAP, normalized by ATR
        atr_val = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        if atr_val <= 0:
            return 0.0

        distance = (latest_close - latest_vwap) / atr_val

        # Positive = above VWAP (bullish), negative = below (bearish)
        score = np.clip(distance / 2.0, -1.0, 1.0) * self.params.vwap_weight
        return score

    def _score_supertrend(self, df: pd.DataFrame) -> float:
        """Score based on Supertrend direction. Range: -15 to +15."""
        st = supertrend(df, self.params.st_period, self.params.st_multiplier)

        if st.empty or pd.isna(st["st_direction"].iloc[-1]):
            return 0.0

        direction = st["st_direction"].iloc[-1]

        # How long has Supertrend been in this direction?
        count = 0
        for i in range(len(st) - 1, -1, -1):
            if st["st_direction"].iloc[i] == direction:
                count += 1
            else:
                break

        # Longer in one direction = higher confidence
        persistence = min(count / 10.0, 1.0)  # Cap at 10 candles = 1.0

        return direction * persistence * self.params.st_weight

    def _score_prev_day_levels(self, df: pd.DataFrame, pdh: float, pdl: float,
                                pdc: float) -> float:
        """Score based on position relative to previous day's H/L/C. Range: -15 to +15."""
        if pd.isna(pdh) or pd.isna(pdl) or pd.isna(pdc):
            return 0.0

        latest_close = df["close"].iloc[-1]
        pd_range = pdh - pdl

        if pd_range <= 0:
            return 0.0

        # Above PDH = strong bullish, below PDL = strong bearish
        if latest_close > pdh:
            score = 1.0
        elif latest_close < pdl:
            score = -1.0
        elif latest_close > pdc:
            # Between PDC and PDH
            score = (latest_close - pdc) / (pdh - pdc) * 0.7
        else:
            # Between PDL and PDC
            score = (latest_close - pdc) / (pdc - pdl) * 0.7

        return score * self.params.pdhl_weight

    def _score_oi(self, oi_signal: Optional[OISignal]) -> float:
        """
        Score from OI analysis. Range: -20 to +20.
        This is the HIGHEST EDGE indicator — institutional positioning.
        """
        if oi_signal is None:
            return 0.0

        # Direction from OI analysis × strength normalized to our weight
        direction = oi_signal.direction
        strength_normalized = oi_signal.strength / 100  # 0 to 1

        score = direction * strength_normalized * self.params.oi_weight

        # Boost if OI buildup confirms direction (strong vs weak signals)
        if oi_signal.oi_buildup in ("long_buildup", "short_buildup"):
            score *= 1.2  # Fresh positions = stronger signal
        elif oi_signal.oi_buildup in ("short_covering", "long_unwinding"):
            score *= 0.7  # Unwinding = weaker, not fresh conviction

        return np.clip(score, -self.params.oi_weight, self.params.oi_weight)

    def _score_orderbook(self, orderbook_signal: Optional[dict]) -> float:
        """
        Score from order book depth analysis. Range: -10 to +10.
        Real-time flow data — faster than any indicator.
        """
        if orderbook_signal is None:
            return 0.0

        signal = orderbook_signal.get("signal", "neutral")
        buy_pressure = orderbook_signal.get("buy_pressure", 50)

        # Map buy_pressure (0-100) to score (-10 to +10)
        score = (buy_pressure - 50) / 50 * self.params.orderbook_weight

        # Boost for absorption detection
        if orderbook_signal.get("absorption_detected", False):
            abs_side = orderbook_signal.get("absorption_side", "")
            if abs_side == "bid":
                score = max(score, self.params.orderbook_weight * 0.7)
            elif abs_side == "ask":
                score = min(score, -self.params.orderbook_weight * 0.7)

        return np.clip(score, -self.params.orderbook_weight, self.params.orderbook_weight)

    # ──────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────

    def _classify_regime(self, total_score: float) -> MarketRegime:
        if total_score >= self.params.strong_trend_threshold:
            return MarketRegime.STRONG_UPTREND
        elif total_score >= self.params.mild_trend_threshold:
            return MarketRegime.MILD_UPTREND
        elif total_score <= -self.params.strong_trend_threshold:
            return MarketRegime.STRONG_DOWNTREND
        elif total_score <= -self.params.mild_trend_threshold:
            return MarketRegime.MILD_DOWNTREND
        else:
            return MarketRegime.SIDEWAYS

    def _classify_volatility(self, vix: float) -> VolatilityRegime:
        if vix < VIX_LOW:
            return VolatilityRegime.LOW
        elif vix < VIX_NORMAL:
            return VolatilityRegime.NORMAL
        elif vix < VIX_HIGH:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.VERY_HIGH

    def _recommend_strategies(self, regime: MarketRegime, vol: VolatilityRegime,
                               confidence: float) -> list:
        """Return ordered list of recommended strategies based on regime + volatility."""
        strategies = []

        if regime in (MarketRegime.STRONG_UPTREND, MarketRegime.STRONG_DOWNTREND):
            if vol in (VolatilityRegime.LOW, VolatilityRegime.NORMAL):
                strategies = ["momentum_buy", "debit_spread", "orb"]
            else:
                # High vol = expensive premiums, prefer spreads
                strategies = ["debit_spread", "momentum_buy"]

        elif regime in (MarketRegime.MILD_UPTREND, MarketRegime.MILD_DOWNTREND):
            strategies = ["debit_spread", "orb", "momentum_buy"]

        elif regime == MarketRegime.SIDEWAYS:
            if confidence < 0.5:
                strategies = ["mean_reversion", "orb"]
            else:
                # Somewhat directional but score near 0 — cautious
                strategies = ["orb", "mean_reversion"]

        # If high confidence, can be more aggressive
        if confidence >= 0.8 and regime != MarketRegime.SIDEWAYS:
            # Push momentum_buy to top
            if "momentum_buy" in strategies:
                strategies.remove("momentum_buy")
                strategies.insert(0, "momentum_buy")

        return strategies
