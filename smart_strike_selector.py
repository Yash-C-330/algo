"""
Smart Strike Selector v2

Answers your critical question: "Should I buy lower strikes for more lots?"

SHORT ANSWER: NO for momentum plays, YES for scalps — here's why:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE DELTA TRAP — Why cheap OTM options are a losing game
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example with BankNifty at 48000:
┌──────────┬─────────┬───────┬────────┬─────────────────────────────────┐
│ Strike   │ Premium │ Delta │ Lots   │ What happens if BN moves +200   │
│          │         │       │ @50K   │ points                          │
├──────────┼─────────┼───────┼────────┼─────────────────────────────────┤
│ 48000 CE │ ₹200    │ 0.50  │ 8 lots │ Premium → ₹300 = +₹100 × 8    │
│ (ATM)    │         │       │        │ = +₹24,000 profit              │
├──────────┼─────────┼───────┼────────┼─────────────────────────────────┤
│ 48500 CE │ ₹70     │ 0.25  │ 23 lots│ Premium → ₹120 = +₹50 × 23    │
│ (OTM)    │         │       │        │ = +₹34,500 profit BUT...       │
├──────────┼─────────┼───────┼────────┼─────────────────────────────────┤
│ 49000 CE │ ₹20     │ 0.08  │ 83 lots│ Premium → ₹35 = +₹15 × 83     │
│ (Deep OTM)│        │       │        │ = +₹37,350 BUT...              │
└──────────┴─────────┴───────┴────────┴─────────────────────────────────┘

LOOKS like deep OTM wins? Here's the catch:

1. THETA DECAY: Deep OTM loses 30-50% per day. If move doesn't happen
   in 2-3 hours, you're dead. ATM only loses 5-10%.

2. BID-ASK SPREAD: ₹20 option has ₹2-3 spread = 10-15% ENTRY TAX.
   ₹200 option has ₹1-2 spread = 0.5-1% entry tax.

3. GAMMA RISK: Deep OTM delta STAYS LOW until strike is near.
   200-point move barely moves ₹20 premium.

4. LIQUIDITY: Deep OTM has LOW volume. Hard to exit, massive slippage.

5. WIN RATE: ATM options profit on ANY move in your direction.
   Deep OTM needs a BIG move. This KILLS win rate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VERDICT:
- For 60-80% win rate → ATM or 1-strike OTM (ALWAYS)
- Deep OTM only for lottery ticket (5% of capital max)
"""
import logging
import math
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from oi_analyzer import StrikeOI

logger = logging.getLogger(__name__)


@dataclass
class StrikeSelection:
    """Recommended strike with full reasoning."""
    strike: float
    option_type: str           # "CE" / "PE"
    estimated_premium: float
    estimated_delta: float
    estimated_theta_per_day: float
    liquidity_score: float     # 0-100
    oi_support_score: float    # 0-100 (OI analysis)
    bid_ask_spread: float
    recommended_lots: int
    risk_per_lot: float
    reason: str


class SmartStrikeSelector:
    """
    Advanced strike selection using:
    1. Delta-adjusted risk/reward
    2. Liquidity (volume + OI)
    3. Bid-ask spread cost
    4. OI analysis (avoid strikes where big sellers are positioned against you)
    5. IV comparison across strikes
    6. Days to expiry consideration

    The goal: Maximize P(profit) × E(profit) - P(loss) × E(loss)
    NOT just maximize number of lots.
    """

    def __init__(self, capital: float = 50000):
        self.capital = capital

    # ─── Black-Scholes helpers ───────────────────

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF (Abramowitz & Stegun approximation)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _bs_delta(spot: float, strike: float, T: float, r: float,
                  sigma: float, option_type: str) -> float:
        """
        Black-Scholes delta.
          T     = time to expiry in years (e.g. 1/365 for one day)
          r     = risk-free rate (annualised, e.g. 0.07)
          sigma = annualised volatility (e.g. VIX/100)
        """
        if T <= 0 or sigma <= 0:
            # At / past expiry — delta is binary
            if option_type == "CE":
                return 1.0 if spot > strike else 0.0
            else:
                return -1.0 if spot < strike else 0.0

        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        cdf_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

        if option_type == "CE":
            return cdf_d1
        else:
            return cdf_d1 - 1.0  # put delta is negative

    @staticmethod
    def _bs_price(spot: float, strike: float, T: float, r: float,
                  sigma: float, option_type: str) -> float:
        """Black-Scholes option price."""
        if T <= 0:
            if option_type == "CE":
                return max(spot - strike, 0.0)
            else:
                return max(strike - spot, 0.0)
        if sigma <= 0:
            sigma = 0.001

        sqrt_T = math.sqrt(T)
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

        if option_type == "CE":
            return spot * Nd1 - strike * math.exp(-r * T) * Nd2
        else:
            return strike * math.exp(-r * T) * (1 - Nd2) - spot * (1 - Nd1)

    # ─── Public API ──────────────────────────────

    def select_optimal_strike(
        self,
        spot_price: float,
        option_type: str,          # "CE" or "PE"
        strike_gap: int,
        lot_size: int,
        chain_data: Dict[float, StrikeOI],
        days_to_expiry: int,
        regime_strength: float,    # 0-100, from regime detector
        strategy_name: str,        # "momentum_buy", "orb", "mean_reversion", "debit_spread"
        risk_per_trade: float = 1500,
        iv: float = 15.0,          # India VIX (annualised vol in % terms)
    ) -> StrikeSelection:
        """
        Select the best strike considering ALL factors.
        iv is India VIX as a percentage (e.g. 15 means 15%).
        """
        sigma = iv / 100.0  # Convert VIX % → decimal for BS
        atm = round(spot_price / strike_gap) * strike_gap

        # Generate candidate strikes: ATM, 1 OTM, 2 OTM, 1 ITM
        if option_type == "CE":
            candidates = [
                atm - strike_gap,      # 1 ITM
                atm,                    # ATM
                atm + strike_gap,      # 1 OTM
                atm + 2 * strike_gap,  # 2 OTM
            ]
        else:
            candidates = [
                atm + strike_gap,      # 1 ITM
                atm,                    # ATM
                atm - strike_gap,      # 1 OTM
                atm - 2 * strike_gap,  # 2 OTM
            ]

        best_score = -999
        best_selection = None

        for strike in candidates:
            score, selection = self._score_strike(
                strike, spot_price, option_type, lot_size,
                chain_data, days_to_expiry, regime_strength,
                strategy_name, risk_per_trade, strike_gap, sigma
            )

            if score > best_score:
                best_score = score
                best_selection = selection

        if best_selection is None:
            # Fallback to ATM
            return self._default_atm(atm, option_type, lot_size, risk_per_trade)

        return best_selection

    def _score_strike(
        self, strike: float, spot_price: float, option_type: str,
        lot_size: int, chain_data: Dict[float, StrikeOI],
        dte: int, regime_strength: float, strategy: str,
        risk_per_trade: float, strike_gap: int,
        sigma: float = 0.15
    ) -> tuple:
        """Score a candidate strike. Returns (score, StrikeSelection)."""
        score = 0
        reasons = []

        # ─── 1. Delta Score (30 points) ───
        # Black-Scholes delta (non-linear, accounts for IV and DTE)
        T = max(dte, 0.25) / 365.0  # Time to expiry in years (floor at ~6 hrs)
        r = 0.07  # India risk-free rate (~repo rate)
        delta = self._bs_delta(spot_price, strike, T, r, sigma, option_type)
        delta = abs(delta)  # Work with absolute delta for scoring

        # For HIGH WIN RATE: prefer higher delta (ATM or slight ITM)
        if strategy in ("momentum_buy", "orb"):
            # High delta = higher win rate but higher cost
            if 0.45 <= delta <= 0.55:
                score += 30  # ATM sweet spot
                reasons.append("ATM_delta_optimal")
            elif 0.55 < delta <= 0.70:
                score += 25  # Slight ITM — even higher win rate
                reasons.append("slight_ITM_high_winrate")
            elif 0.30 <= delta < 0.45:
                score += 20  # 1 OTM
                reasons.append("1OTM_decent_delta")
            elif delta < 0.30:
                score += 5   # Deep OTM — BAD for win rate
                reasons.append("deep_OTM_low_delta")

        elif strategy == "mean_reversion":
            # Scalping = ATM for quick moves
            if 0.45 <= delta <= 0.55:
                score += 30
                reasons.append("ATM_best_for_scalp")
            elif delta > 0.55:
                score += 20
            else:
                score += 10

        elif strategy == "debit_spread":
            # Buy ATM, sell OTM — ATM leg needs good delta
            if 0.45 <= delta <= 0.55:
                score += 28
                reasons.append("ATM_for_spread_buy_leg")
            else:
                score += 15

        # ─── 2. Liquidity Score (25 points) ───
        strike_data = chain_data.get(strike)
        if strike_data:
            volume = strike_data.ce_volume if option_type == "CE" else strike_data.pe_volume
            oi = strike_data.ce_oi if option_type == "CE" else strike_data.pe_oi

            if volume > 100000:
                score += 25
                reasons.append("high_liquidity")
            elif volume > 50000:
                score += 18
                reasons.append("good_liquidity")
            elif volume > 10000:
                score += 10
                reasons.append("adequate_liquidity")
            else:
                score += 2  # Low liquidity penalty
                reasons.append("LOW_LIQUIDITY_WARNING")
        else:
            score += 5  # Unknown liquidity

        # ─── 3. OI Analysis Score (20 points) ───
        if strike_data:
            ce_oi = strike_data.ce_oi
            pe_oi = strike_data.pe_oi

            if option_type == "CE":
                # Buying CE: we don't want massive CE OI at this strike
                # (that means sellers believe price won't reach here)
                if ce_oi > pe_oi * 2:
                    score -= 5  # Heavy call writing = resistance here
                    reasons.append("heavy_CE_OI=resistance")
                elif pe_oi > ce_oi:
                    score += 15  # More put writing = support forming
                    reasons.append("PE_OI>CE_OI=supportive")
                else:
                    score += 8
            else:  # PE
                if pe_oi > ce_oi * 2:
                    score -= 5  # Heavy put writing = support (bad for PE buy)
                    reasons.append("heavy_PE_OI=support")
                elif ce_oi > pe_oi:
                    score += 15
                    reasons.append("CE_OI>PE_OI=resistance_forming")
                else:
                    score += 8

        # ─── 4. Cost Efficiency (15 points) ───
        estimated_premium = self._estimate_premium(
            spot_price, strike, option_type, dte, sigma
        )
        cost_per_lot = estimated_premium * lot_size

        if cost_per_lot <= 0:
            return -999, None

        lots = max(1, int(risk_per_trade / (estimated_premium * 0.30 * lot_size)))

        total_cost = cost_per_lot * lots

        if total_cost > self.capital * 0.4:
            # Too expensive — reduce lots
            lots = max(1, int(self.capital * 0.4 / cost_per_lot))

        # Don't reward MORE lots from cheap strikes — reward DELTA-ADJUSTED profit
        expected_move_profit = delta * 100 * lot_size * lots  # If index moves 100 pts
        cost = estimated_premium * lot_size * lots

        if cost > 0:
            efficiency = expected_move_profit / cost
            eff_score = min(efficiency * 5, 15)
            score += eff_score
            reasons.append(f"efficiency={efficiency:.2f}")

        # ─── 5. DTE Consideration (10 points) ───
        theta_per_day = estimated_premium * (0.05 if delta > 0.4 else 0.15)

        if dte <= 1:
            # Expiry day: prefer slight ITM (no extrinsic, pure delta)
            if delta >= 0.55:
                score += 10
                reasons.append("expiry_day_ITM_preferred")
            elif delta >= 0.45:
                score += 7
            else:
                score -= 5  # OTM on expiry = rapid decay
                reasons.append("expiry_day_OTM_AVOID")
        elif dte <= 3:
            # Near expiry: ATM is optimal (gamma explosion helps)
            if 0.4 <= delta <= 0.6:
                score += 10
                reasons.append("near_expiry_gamma_boost")
        else:
            score += 5  # Normal DTE

        # Build spread estimate
        spread = estimated_premium * (0.01 if delta > 0.4 else 0.05)

        selection = StrikeSelection(
            strike=strike,
            option_type=option_type,
            estimated_premium=round(estimated_premium, 1),
            estimated_delta=round(delta, 2),
            estimated_theta_per_day=round(theta_per_day, 1),
            liquidity_score=min(score, 100),
            oi_support_score=0,
            bid_ask_spread=round(spread, 2),
            recommended_lots=lots,
            risk_per_lot=round(estimated_premium * 0.30 * lot_size, 0),
            reason=" | ".join(reasons)
        )

        return score, selection

    def _estimate_premium(self, spot_price: float, strike: float,
                           option_type: str, dte: int,
                           sigma: float = 0.15) -> float:
        """
        Black-Scholes–based premium estimation.
        In production the real LTP is used; this is a fallback for strike scoring.
        """
        T = max(dte, 0.25) / 365.0
        r = 0.07
        price = self._bs_price(spot_price, strike, T, r, sigma, option_type)
        return max(price, 0.05)  # Floor to 1 tick

    def _default_atm(self, atm: float, option_type: str,
                      lot_size: int, risk_per_trade: float) -> StrikeSelection:
        """Fallback to ATM when no chain data available."""
        return StrikeSelection(
            strike=atm,
            option_type=option_type,
            estimated_premium=150,
            estimated_delta=0.50,
            estimated_theta_per_day=8,
            liquidity_score=70,
            oi_support_score=50,
            bid_ask_spread=1.5,
            recommended_lots=max(1, int(risk_per_trade / (150 * 0.30 * lot_size))),
            risk_per_lot=150 * 0.30 * lot_size,
            reason="fallback_ATM_no_chain_data"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXPIRY DAY SPECIAL STRATEGY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExpiryDayStrategy:
    """
    Special handling for weekly expiry days (highest opportunity AND risk).

    On expiry day, gamma is extreme near ATM:
    - Small 50-point move → 200-300% premium change
    - But if wrong direction → 0 in minutes

    Expiry Edge:
    1. Max Pain gravity is STRONGEST on expiry day
    2. Options below ₹10 can become ₹100+ on sharp moves
    3. Theta burns fast → sell strategy works if time is right

    Strategy: Use Max Pain as primary target, trade toward it.
    """

    def should_trade_toward_max_pain(self, spot_price: float,
                                      max_pain: float,
                                      current_time_hour: int) -> dict:
        """
        On expiry day, price gravitates to max pain.

        Before 1 PM: Mild pull (other forces dominate)
        After 1 PM: Strong pull (option sellers start hedging)
        After 2 PM: Very strong (delta hedging cascade)
        """
        distance = spot_price - max_pain
        distance_pct = abs(distance) / spot_price * 100

        if current_time_hour < 11:
            gravity = 0.3  # Weak
        elif current_time_hour < 13:
            gravity = 0.5  # Moderate
        elif current_time_hour < 14:
            gravity = 0.7  # Strong
        else:
            gravity = 0.9  # Very strong

        if distance_pct > 1.0:
            # Too far from max pain — max pain pull weaker
            gravity *= 0.5

        if distance > 0:
            # Spot above max pain → bearish pull
            direction = "bearish"
            option_type = "PE"
        elif distance < 0:
            direction = "bullish"
            option_type = "CE"
        else:
            direction = "neutral"
            option_type = None

        return {
            "direction": direction,
            "option_type": option_type,
            "gravity_strength": round(gravity, 2),
            "max_pain": max_pain,
            "distance": round(distance, 0),
            "distance_pct": round(distance_pct, 2),
            "recommendation": (
                f"{'Strong' if gravity > 0.6 else 'Mild'} pull toward "
                f"MP={max_pain:.0f} ({direction})"
                if option_type else "At max pain — expect range"
            )
        }
