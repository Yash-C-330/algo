"""
Open Interest (OI) Analysis Engine — THE KEY EDGE

Why this matters more than any technical indicator:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Price and indicators tell you WHAT happened.
Open Interest tells you WHERE BIG MONEY is positioned.

In Indian markets, option sellers (institutions) control 80%+ of OI.
They sell at strikes they believe WON'T be breached.
This creates support/resistance walls you can trade around.

Key Concepts:
- Max Pain: The strike where option sellers profit most = where price GRAVITATES
- PCR (Put/Call Ratio): Measures sentiment; extreme readings = reversal signals
- OI Change: Rising OI + price move = strong; falling OI = weak move
- OI Walls: Strikes with huge OI act as magnets / barriers
- Unwinding: Sudden OI drop = big players exiting = trend exhaustion
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrikeOI:
    """OI data for a single strike."""
    strike: float
    ce_oi: int = 0
    pe_oi: int = 0
    ce_oi_change: int = 0     # Change from previous session / interval
    pe_oi_change: int = 0
    ce_volume: int = 0
    pe_volume: int = 0
    ce_ltp: float = 0.0
    pe_ltp: float = 0.0
    ce_iv: float = 0.0        # Implied volatility
    pe_iv: float = 0.0


@dataclass
class OISnapshot:
    """Complete option chain OI snapshot."""
    timestamp: str
    spot_price: float
    strikes: Dict[float, StrikeOI] = field(default_factory=dict)
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr: float = 1.0          # Put/Call Ratio
    max_pain: float = 0.0
    max_ce_oi_strike: float = 0.0   # Resistance wall
    max_pe_oi_strike: float = 0.0   # Support wall


@dataclass
class OISignal:
    """Actionable signal from OI analysis."""
    direction: int           # +1 bullish, -1 bearish, 0 neutral
    strength: float          # 0-100
    support_level: float     # Near-term support from OI
    resistance_level: float  # Near-term resistance from OI
    max_pain: float          # Price gravitational center
    pcr_signal: str          # "bullish" / "bearish" / "neutral" / "extreme_bullish" etc
    oi_buildup: str          # "long_buildup" / "short_buildup" / "long_unwinding" / "short_covering"
    reasoning: list          # Human-readable reasons


class OIAnalyzer:
    """
    Analyzes option chain OI data to extract:
    1. Max Pain — where price tends to expire
    2. PCR — sentiment gauge
    3. OI Walls — support/resistance from big positions
    4. OI Change interpretation — what smart money is doing
    5. IV Skew — fear/greed in specific directions

    This is the single biggest edge improvement over pure technical analysis.
    """

    def __init__(self):
        self.oi_history: List[OISnapshot] = []  # Track changes over time
        self.pcr_history: List[float] = []

    def reset_daily(self):
        """Clear accumulated history for a new trading day."""
        self.oi_history.clear()
        self.pcr_history.clear()

    def analyze(self, chain_data: Dict[float, StrikeOI],
                spot_price: float,
                prev_snapshot: Optional[OISnapshot] = None) -> OISignal:
        """
        Full OI analysis. Returns actionable signal.

        Args:
            chain_data: Dict of strike -> StrikeOI from option chain
            spot_price: Current spot/index price
            prev_snapshot: Previous OI snapshot for change analysis
        """
        if not chain_data:
            return self._neutral_signal(spot_price)

        snapshot = self._build_snapshot(chain_data, spot_price)
        self.oi_history.append(snapshot)

        direction = 0
        strength = 0
        reasons = []

        # ─── 1. Max Pain Analysis (25 points) ───
        mp_dir, mp_str, mp_reason = self._analyze_max_pain(snapshot, spot_price)
        direction += mp_dir * 25
        strength += mp_str
        reasons.append(mp_reason)

        # ─── 2. PCR Analysis (25 points) ───
        pcr_dir, pcr_str, pcr_reason, pcr_label = self._analyze_pcr(snapshot)
        direction += pcr_dir * 25
        strength += pcr_str
        reasons.append(pcr_reason)
        self.pcr_history.append(snapshot.pcr)

        # ─── 3. OI Wall Analysis (25 points) ───
        wall_dir, wall_str, wall_reason = self._analyze_oi_walls(snapshot, spot_price)
        direction += wall_dir * 25
        strength += wall_str
        reasons.append(wall_reason)

        # ─── 4. OI Change / Buildup Analysis (25 points) ───
        if prev_snapshot:
            buildup_dir, buildup_str, buildup_reason, buildup_type = \
                self._analyze_oi_change(snapshot, prev_snapshot, spot_price)
            direction += buildup_dir * 25
            strength += buildup_str
            reasons.append(buildup_reason)
        else:
            buildup_type = "unknown"

        # Normalize
        max_possible = 100
        final_direction = 1 if direction > 0 else (-1 if direction < 0 else 0)
        final_strength = min(abs(strength), max_possible)

        return OISignal(
            direction=final_direction,
            strength=final_strength,
            support_level=snapshot.max_pe_oi_strike,
            resistance_level=snapshot.max_ce_oi_strike,
            max_pain=snapshot.max_pain,
            pcr_signal=pcr_label,
            oi_buildup=buildup_type,
            reasoning=reasons
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. MAX PAIN CALCULATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_max_pain(self, chain_data: Dict[float, StrikeOI]) -> float:
        """
        Max Pain = strike at which total intrinsic value loss for option BUYERS is maximum.
        Price tends to gravitate here, especially near expiry.

        Method: For each possible expiry price, calculate total payout.
        Max pain = price with minimum total payout.
        """
        strikes = sorted(chain_data.keys())
        if not strikes:
            return 0

        min_pain = float('inf')
        max_pain_strike = strikes[len(strikes) // 2]

        for assumed_price in strikes:
            total_pain = 0
            for strike, oi in chain_data.items():
                # CE buyers lose if price < strike
                if assumed_price > strike:
                    total_pain += (assumed_price - strike) * oi.ce_oi
                # PE buyers lose if price > strike
                if assumed_price < strike:
                    total_pain += (strike - assumed_price) * oi.pe_oi

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = assumed_price

        return max_pain_strike

    def _analyze_max_pain(self, snapshot: OISnapshot,
                           spot_price: float) -> Tuple[int, float, str]:
        """
        Max Pain signal:
        - Spot above max pain → bearish pull (sellers will push down)
        - Spot below max pain → bullish pull (sellers will push up)
        - Near max pain (within 0.3%) → neutral, expect range

        This is most powerful 1-2 days before expiry.
        """
        mp = snapshot.max_pain
        if mp == 0:
            return 0, 0, "max_pain: no data"

        distance_pct = (spot_price - mp) / spot_price * 100

        if abs(distance_pct) < 0.3:
            return 0, 10, f"max_pain: spot near MP({mp:.0f}), expect range"
        elif distance_pct > 0.8:
            return -1, 22, f"max_pain: spot {distance_pct:.1f}% ABOVE MP({mp:.0f}), bearish drag"
        elif distance_pct > 0.3:
            return -1, 15, f"max_pain: spot slightly above MP({mp:.0f})"
        elif distance_pct < -0.8:
            return 1, 22, f"max_pain: spot {abs(distance_pct):.1f}% BELOW MP({mp:.0f}), bullish pull"
        else:
            return 1, 15, f"max_pain: spot slightly below MP({mp:.0f})"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. PUT/CALL RATIO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_pcr(self, snapshot: OISnapshot) -> Tuple[int, float, str, str]:
        """
        PCR = Total PE OI / Total CE OI

        Interpretation (contrarian for extremes):
        - PCR > 1.3  → Extreme bearish sentiment → BULLISH (too many puts = support)
        - PCR 1.0-1.3 → Healthy bullish
        - PCR 0.7-1.0 → Neutral to mild bearish
        - PCR < 0.7   → Extreme bullish → BEARISH (complacency, no protection)

        KEY INSIGHT: PCR > 1.2 means institutions are SELLING puts = they're bullish.
        PCR < 0.7 means institutions are SELLING calls = they're bearish.
        """
        pcr = snapshot.pcr

        if pcr > 1.5:
            return 1, 25, f"PCR={pcr:.2f}: extreme put writing = very bullish", "extreme_bullish"
        elif pcr > 1.2:
            return 1, 20, f"PCR={pcr:.2f}: heavy put support = bullish", "bullish"
        elif pcr > 1.0:
            return 1, 12, f"PCR={pcr:.2f}: mild bullish", "mild_bullish"
        elif pcr > 0.8:
            return 0, 5, f"PCR={pcr:.2f}: neutral zone", "neutral"
        elif pcr > 0.6:
            return -1, 15, f"PCR={pcr:.2f}: call heavy = bearish", "bearish"
        else:
            return -1, 22, f"PCR={pcr:.2f}: extreme call writing = very bearish", "extreme_bearish"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. OI WALL DETECTION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_oi_walls(self, snapshot: OISnapshot,
                           spot_price: float) -> Tuple[int, float, str]:
        """
        OI Walls = Strikes with maximum CE and PE OI.

        - Max CE OI strike = RESISTANCE (call sellers believe price won't cross)
        - Max PE OI strike = SUPPORT (put sellers believe price won't fall below)

        Trading logic:
        - If spot is near max CE OI → expect resistance, bearish bias
        - If spot is near max PE OI → expect support, bullish bias
        - Wider gap between walls → bigger range to trade in
        """
        resistance = snapshot.max_ce_oi_strike
        support = snapshot.max_pe_oi_strike

        if resistance == 0 or support == 0:
            return 0, 0, "oi_walls: insufficient data"

        range_size = resistance - support
        spot_position = (spot_price - support) / range_size if range_size > 0 else 0.5

        # Where is spot within the OI range?
        if spot_position < 0.25:
            # Near support wall → bullish
            return 1, 22, (f"oi_walls: spot near PE wall({support:.0f}), "
                           f"resistance at {resistance:.0f}")
        elif spot_position > 0.75:
            # Near resistance wall → bearish
            return -1, 22, (f"oi_walls: spot near CE wall({resistance:.0f}), "
                            f"support at {support:.0f}")
        elif 0.4 < spot_position < 0.6:
            # Mid-range → neutral
            return 0, 8, (f"oi_walls: spot in middle, S={support:.0f} R={resistance:.0f}")
        elif spot_position <= 0.4:
            return 1, 15, f"oi_walls: spot in lower half, support {support:.0f}"
        else:
            return -1, 15, f"oi_walls: spot in upper half, resistance {resistance:.0f}"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. OI CHANGE INTERPRETATION (BUILDUP ANALYSIS)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_oi_change(self, current: OISnapshot, previous: OISnapshot,
                            spot_price: float) -> Tuple[int, float, str, str]:
        """
        OI Change + Price Change = Smart Money Action:

        | Price | OI     | Interpretation      | Action  |
        |-------|--------|---------------------|---------|
        | ↑     | ↑      | LONG BUILDUP        | Bullish |
        | ↑     | ↓      | SHORT COVERING       | Weak bullish (exit, not entry) |
        | ↓     | ↑      | SHORT BUILDUP        | Bearish |
        | ↓     | ↓      | LONG UNWINDING       | Weak bearish (exit, not entry) |

        We also look at WHERE the OI is changing:
        - CE OI increasing at higher strikes = new resistance being written = bullish range expansion unlikely
        - PE OI increasing at lower strikes = new support being created = bullish
        """
        price_change = spot_price - previous.spot_price
        total_oi_change = (current.total_ce_oi + current.total_pe_oi) - \
                          (previous.total_ce_oi + previous.total_pe_oi)

        price_up = price_change > 0
        oi_up = total_oi_change > 0

        if price_up and oi_up:
            buildup_type = "long_buildup"
            direction = 1
            strength = 25
            reason = f"oi_change: LONG BUILDUP (price↑ + OI↑), strong bullish"
        elif price_up and not oi_up:
            buildup_type = "short_covering"
            direction = 1
            strength = 10  # Weak — just shorts exiting, not new longs
            reason = f"oi_change: SHORT COVERING (price↑ + OI↓), weak bullish"
        elif not price_up and oi_up:
            buildup_type = "short_buildup"
            direction = -1
            strength = 25
            reason = f"oi_change: SHORT BUILDUP (price↓ + OI↑), strong bearish"
        else:
            buildup_type = "long_unwinding"
            direction = -1
            strength = 10
            reason = f"oi_change: LONG UNWINDING (price↓ + OI↓), weak bearish"

        # Additional: Check if PE OI is building (support) or CE OI building (resistance)
        pe_oi_delta = current.total_pe_oi - previous.total_pe_oi
        ce_oi_delta = current.total_ce_oi - previous.total_ce_oi

        if pe_oi_delta > ce_oi_delta * 1.5:
            # Much more put writing = institutions confident → bullish boost
            strength = min(strength + 5, 25)
            reason += " | heavy put writing = bullish support"
        elif ce_oi_delta > pe_oi_delta * 1.5:
            strength = min(strength + 5, 25)
            reason += " | heavy call writing = bearish cap"

        return direction, strength, reason, buildup_type

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HELPER METHODS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_snapshot(self, chain_data: Dict[float, StrikeOI],
                         spot_price: float) -> OISnapshot:
        """Build a complete OI snapshot from raw chain data."""
        total_ce = sum(s.ce_oi for s in chain_data.values())
        total_pe = sum(s.pe_oi for s in chain_data.values())
        pcr = total_pe / total_ce if total_ce > 0 else 1.0

        # Find max OI strikes
        max_ce_strike = max(chain_data.keys(),
                            key=lambda k: chain_data[k].ce_oi,
                            default=spot_price)
        max_pe_strike = max(chain_data.keys(),
                            key=lambda k: chain_data[k].pe_oi,
                            default=spot_price)

        max_pain = self._calculate_max_pain(chain_data)

        return OISnapshot(
            timestamp=pd.Timestamp.now().strftime("%H:%M:%S"),
            spot_price=spot_price,
            strikes=chain_data,
            total_ce_oi=total_ce,
            total_pe_oi=total_pe,
            pcr=round(pcr, 3),
            max_pain=max_pain,
            max_ce_oi_strike=max_ce_strike,
            max_pe_oi_strike=max_pe_strike
        )

    def _neutral_signal(self, spot_price: float) -> OISignal:
        return OISignal(
            direction=0, strength=0,
            support_level=spot_price - 200,
            resistance_level=spot_price + 200,
            max_pain=spot_price,
            pcr_signal="neutral",
            oi_buildup="unknown",
            reasoning=["No OI data available"]
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # IV SKEW ANALYSIS (Bonus Edge)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_iv_skew(self, chain_data: Dict[float, StrikeOI],
                         spot_price: float, strike_gap: int) -> dict:
        """
        IV Skew = difference in implied volatility between PE and CE at same distance.

        - PE IV > CE IV → market fears downside → short-term bearish sentiment
        - CE IV > PE IV → unusual → potential upside squeeze expected

        Also: ATM IV vs OTM IV tells you about tail risk pricing.
        """
        atm = round(spot_price / strike_gap) * strike_gap

        atm_data = chain_data.get(atm)
        if atm_data is None:
            return {"skew": 0, "atm_iv": 0, "interpretation": "no data"}

        ce_iv = atm_data.ce_iv
        pe_iv = atm_data.pe_iv
        skew = pe_iv - ce_iv

        # Check 1-OTM strikes
        otm_ce_strike = atm + strike_gap
        otm_pe_strike = atm - strike_gap
        otm_ce_data = chain_data.get(otm_ce_strike)
        otm_pe_data = chain_data.get(otm_pe_strike)

        otm_skew = 0
        if otm_ce_data and otm_pe_data:
            otm_skew = otm_pe_data.pe_iv - otm_ce_data.ce_iv

        if skew > 5:
            interpretation = "put_fear_high"  # Bearish sentiment but contrarian = bullish
        elif skew < -3:
            interpretation = "call_demand_high"  # Unusual upside demand
        else:
            interpretation = "balanced"

        return {
            "skew": round(skew, 2),
            "otm_skew": round(otm_skew, 2),
            "atm_ce_iv": round(ce_iv, 2),
            "atm_pe_iv": round(pe_iv, 2),
            "interpretation": interpretation
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REAL-TIME OI CHANGE TRACKING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def detect_sudden_oi_shift(self, current: OISnapshot,
                                threshold_pct: float = 5.0) -> Optional[dict]:
        """
        Detect sudden large OI changes at specific strikes.
        This indicates institutional activity — a high-value signal.

        Example: If suddenly 50 lakh PE OI appears at 47000 strike,
        institutions are SELLING puts there → they're confident price stays above 47000.
        """
        if len(self.oi_history) < 2:
            return None

        prev = self.oi_history[-2]
        alerts = []

        for strike, curr_oi in current.strikes.items():
            prev_oi = prev.strikes.get(strike)
            if prev_oi is None:
                continue

            # Check CE OI sudden change
            if prev_oi.ce_oi > 0:
                ce_change_pct = (curr_oi.ce_oi - prev_oi.ce_oi) / prev_oi.ce_oi * 100
                if abs(ce_change_pct) > threshold_pct:
                    alerts.append({
                        "strike": strike,
                        "type": "CE",
                        "change_pct": round(ce_change_pct, 1),
                        "new_oi": curr_oi.ce_oi,
                        "signal": "resistance_building" if ce_change_pct > 0 else "resistance_unwinding"
                    })

            # Check PE OI sudden change
            if prev_oi.pe_oi > 0:
                pe_change_pct = (curr_oi.pe_oi - prev_oi.pe_oi) / prev_oi.pe_oi * 100
                if abs(pe_change_pct) > threshold_pct:
                    alerts.append({
                        "strike": strike,
                        "type": "PE",
                        "change_pct": round(pe_change_pct, 1),
                        "new_oi": curr_oi.pe_oi,
                        "signal": "support_building" if pe_change_pct > 0 else "support_weakening"
                    })

        if alerts:
            return {"timestamp": current.timestamp, "alerts": alerts}
        return None
