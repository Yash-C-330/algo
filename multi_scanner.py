"""
Multi-Instrument Scanner — Trade Multiplier #1

PROBLEM: V1/V2 watches one instrument. If Nifty is sideways, you sit idle.
SOLUTION: Scan Nifty, BankNifty, Sensex — trade whichever has the BEST setup.

POST-SEBI (Nov 2024) EXPIRY REALITY:
  Only 1 weekly expiry per exchange is allowed.
  - NSE weekly: NIFTY (Tuesday)
  - BSE weekly: SENSEX (Thursday)
  - BankNifty: MONTHLY only (last Thursday of the month)
  - FinNifty & MidcapNifty: DISCONTINUED (no more F&O weeklies)

  So on most weeks:
    Tuesday   = Nifty weekly expiry (highest gamma edge)
    Thursday  = Sensex weekly expiry
    Last Thursday of month = BankNifty monthly expiry (overlaps with Sensex!)

HOLIDAY SHIFT:
  If Tuesday is a market holiday, Nifty expiry moves to MONDAY.
  If Thursday is a holiday, Sensex expiry moves to WEDNESDAY.
  We check the holiday calendar in config.py.

WHY EXPIRY DAY MATTERS:
- Gamma is highest on expiry → small moves = big premium swings
- Max Pain pull is strongest
- OI data is most actionable
- TRADE THE INSTRUMENT WHOSE EXPIRY IS TODAY
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime, date, timedelta

from config import INSTRUMENTS, InstrumentConfig, NSE_HOLIDAYS_2026, BSE_HOLIDAYS_2026
from regime_detector import RegimeDetector, RegimeState

logger = logging.getLogger(__name__)


@dataclass
class InstrumentScore:
    """Score for one instrument — which one should we trade?"""
    instrument_key: str
    config: InstrumentConfig
    regime_state: Optional[RegimeState]
    trading_score: float        # Combined score for "how tradeable is this right now"
    is_expiry_day: bool         # True if this instrument expires today
    reasons: list


class MultiInstrumentScanner:
    """
    Scans all configured instruments and ranks them by:
    1. Setup quality (regime strength + confidence)
    2. Is it expiry day? (massive edge — trade this one)
    3. Liquidity (volume, spread)
    4. ATR relative to premium (bang for buck)
    5. Avoid instruments in dead zones (sideways + low ADX)

    Returns the BEST instrument to trade right now.
    """

    # Day name → weekday number (for holiday shift calculation)
    DAY_TO_NUM = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4,
    }

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.last_scan_results: List[InstrumentScore] = []

    # ── Holiday-aware expiry detection ──────────────

    @staticmethod
    def _get_holidays_for_exchange(exchange: str) -> List[str]:
        """Return holiday date strings for the exchange."""
        if exchange == "BSE":
            return BSE_HOLIDAYS_2026
        return NSE_HOLIDAYS_2026

    @staticmethod
    def _is_holiday(d: date, holidays: List[str]) -> bool:
        """Check if a date is a market holiday."""
        return d.strftime("%Y-%m-%d") in holidays

    def _get_previous_trading_day(self, d: date, holidays: List[str]) -> date:
        """Walk backward from d to find the previous trading day (skip weekends + holidays)."""
        candidate = d - timedelta(days=1)
        while candidate.weekday() >= 5 or self._is_holiday(candidate, holidays):
            candidate -= timedelta(days=1)
        return candidate

    def is_weekly_expiry_today(self, config: InstrumentConfig) -> bool:
        """
        Check if today is the weekly expiry day for this instrument,
        accounting for holidays. If the normal expiry weekday is a holiday,
        expiry shifts to the previous trading day.
        """
        if config.expiry_type != "weekly":
            return False

        today = date.today()
        normal_weekday = self.DAY_TO_NUM.get(config.expiry_day, -1)
        if normal_weekday < 0:
            return False

        holidays = self._get_holidays_for_exchange(config.exchange)

        # Find this week's scheduled expiry date
        days_until = (normal_weekday - today.weekday()) % 7
        this_weeks_expiry = today + timedelta(days=days_until)
        # If we've passed it, look at this week (not next)
        if days_until > 0:
            this_weeks_expiry = today + timedelta(days=days_until - 7)
            # Unless it's actually later this week
            if (normal_weekday - today.weekday()) >= 0:
                this_weeks_expiry = today + timedelta(days=days_until)

        # Simpler: get the expiry date for the week containing today
        # Start of week (Monday)
        start_of_week = today - timedelta(days=today.weekday())
        this_weeks_expiry = start_of_week + timedelta(days=normal_weekday)

        # If that day is a holiday, shift to previous trading day
        if self._is_holiday(this_weeks_expiry, holidays):
            this_weeks_expiry = self._get_previous_trading_day(this_weeks_expiry, holidays)

        return today == this_weeks_expiry

    def is_monthly_expiry_today(self, config: InstrumentConfig) -> bool:
        """
        Check if today is the monthly expiry for this instrument.
        Monthly expiry = LAST occurrence of expiry_day in the current month.
        E.g., BankNifty monthly = last Thursday.
        If that day is a holiday, shifts to previous trading day.
        """
        if config.expiry_type != "monthly":
            return False

        today = date.today()
        normal_weekday = self.DAY_TO_NUM.get(config.expiry_day, -1)
        if normal_weekday < 0:
            return False

        holidays = self._get_holidays_for_exchange(config.exchange)

        # Find last occurrence of the weekday in this month
        # Start from last day of month, walk backward
        if today.month == 12:
            next_month_first = date(today.year + 1, 1, 1)
        else:
            next_month_first = date(today.year, today.month + 1, 1)
        last_day = next_month_first - timedelta(days=1)

        candidate = last_day
        while candidate.weekday() != normal_weekday:
            candidate -= timedelta(days=1)

        # candidate is now the last <weekday> of the month
        if self._is_holiday(candidate, holidays):
            candidate = self._get_previous_trading_day(candidate, holidays)

        return today == candidate

    def is_expiry_today(self, key: str, config: InstrumentConfig) -> bool:
        """Check if any type of expiry is today for this instrument."""
        return self.is_weekly_expiry_today(config) or self.is_monthly_expiry_today(config)

    def scan_all(self, data_dict: Dict[str, dict],
                 prev_days: Dict[str, dict],
                 india_vix: float = 15.0,
                 oi_signals: Dict[str, object] = None,
                 orderbook_signals: Dict[str, dict] = None) -> List[InstrumentScore]:
        """
        Score all instruments and return ranked list.

        Args:
            data_dict: {instrument_key: {"5min": df, "15min": df, "spot": price}}
            prev_days: {instrument_key: {"prev_high": x, "prev_low": y, "prev_close": z}}
            india_vix: Current VIX
            oi_signals: {instrument_key: OISignal} — from OI analyzer
            orderbook_signals: {instrument_key: dict} — from order book analyzer

        Returns:
            Sorted list of InstrumentScore (best first)
        """
        results = []

        for key, config in INSTRUMENTS.items():
            if key not in data_dict:
                continue

            inst_data = data_dict[key]
            df_5min = inst_data.get("5min")
            df_15min = inst_data.get("15min")
            spot = inst_data.get("spot")

            if df_5min is None or df_5min.empty or spot is None:
                continue

            prev = prev_days.get(key, {})
            oi_sig = oi_signals.get(key) if oi_signals else None
            ob_sig = orderbook_signals.get(key) if orderbook_signals else None

            # Detect regime
            try:
                regime = self.regime_detector.detect(
                    df_5min, df_15min,
                    prev.get("prev_high", 0),
                    prev.get("prev_low", 0),
                    prev.get("prev_close", 0),
                    india_vix,
                    oi_sig, ob_sig
                )
            except Exception as e:
                logger.warning(f"Regime detection failed for {key}: {e}")
                continue

            is_expiry = self.is_expiry_today(key, config)
            score, reasons = self._score_instrument(
                key, config, regime, is_expiry, df_5min, spot, oi_sig
            )

            results.append(InstrumentScore(
                instrument_key=key,
                config=config,
                regime_state=regime,
                trading_score=round(score, 1),
                is_expiry_day=is_expiry,
                reasons=reasons
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.trading_score, reverse=True)
        self.last_scan_results = results

        if results:
            best = results[0]
            logger.info(f"SCANNER: Best={best.instrument_key} "
                         f"score={best.trading_score} "
                         f"expiry={best.is_expiry_day} "
                         f"regime={best.regime_state.regime.value}")

        return results

    def get_best_instrument(self, results: List[InstrumentScore] = None,
                            data_dict=None, prev_days=None, india_vix=15.0,
                            oi_signals=None, orderbook_signals=None) -> Optional[InstrumentScore]:
        """Return just the best instrument. Can pass pre-computed results or raw data."""
        if results is None and data_dict is not None:
            results = self.scan_all(data_dict, prev_days or {}, india_vix,
                                    oi_signals, orderbook_signals)
        elif results is None:
            results = self.last_scan_results
        return results[0] if results else None

    def _score_instrument(self, key: str, config: InstrumentConfig,
                           regime: RegimeState, is_expiry: bool,
                           df_5min, spot: float,
                           oi_signal=None) -> Tuple[float, list]:
        """Score a single instrument for tradeability."""
        score = 0.0
        reasons = []

        # ─── 1. Regime Strength (40 points) ───
        abs_regime = abs(regime.total_score)
        regime_score = min(abs_regime / 100 * 40, 40)
        score += regime_score
        reasons.append(f"regime_str={abs_regime:.0f}")

        # Sideways = penalize (fewer opportunities)
        if regime.is_sideways:
            score -= 15
            reasons.append("sideways_penalty")

        # ─── 2. Expiry Day Bonus (25 points) — HUGE ───
        if is_expiry:
            score += 25
            reasons.append("EXPIRY_DAY_BONUS")

            # Extra bonus in afternoon (max pain effect stronger)
            hour = datetime.now().hour
            if hour >= 13:
                score += 10
                reasons.append("afternoon_expiry_prime_time")

        # ─── 3. Confidence (15 points) ───
        conf_score = regime.confidence * 15
        score += conf_score
        reasons.append(f"confidence={regime.confidence:.2f}")

        # ─── 4. Volatility / Movement (10 points) ───
        if len(df_5min) > 14:
            from indicators import atr as calc_atr
            atr_val = calc_atr(df_5min).iloc[-1]
            atr_pct = atr_val / spot * 100 if spot > 0 else 0

            if atr_pct > 0.15:
                score += 10
                reasons.append(f"good_movement={atr_pct:.2f}%")
            elif atr_pct > 0.08:
                score += 5
                reasons.append(f"moderate_movement={atr_pct:.2f}%")
            else:
                score -= 5
                reasons.append(f"dead_movement={atr_pct:.2f}%")

        # ─── 5. OI Signal Alignment (10 points) ───
        if oi_signal:
            if oi_signal.strength > 50:
                score += 10
                reasons.append("strong_oi_signal")
            elif oi_signal.strength > 25:
                score += 5
                reasons.append("moderate_oi_signal")

        return score, reasons

    def get_todays_expiry_instruments(self) -> list:
        """Get instruments that have any expiry (weekly or monthly) today."""
        return [k for k, cfg in INSTRUMENTS.items() if self.is_expiry_today(k, cfg)]
