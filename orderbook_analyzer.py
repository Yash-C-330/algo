"""
Order Book / Market Depth Analyzer

Reads Level 2 data (bid/ask depth) to understand:
1. Are buyers or sellers more aggressive?
2. Is there hidden institutional buying/selling?
3. What's the real expected slippage at our order size?
4. Are big orders stacking at specific levels (absorption)?

This is a REAL-TIME edge — price and OI are lagging, order book is NOW.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DepthLevel:
    """Single level in the order book."""
    price: float
    quantity: int
    orders: int     # Number of distinct orders at this level


@dataclass
class OrderBookSnapshot:
    """Complete 5-level order book."""
    bids: List[DepthLevel] = field(default_factory=list)  # Sorted high → low
    asks: List[DepthLevel] = field(default_factory=list)  # Sorted low → high
    ltp: float = 0.0
    total_bid_qty: int = 0
    total_ask_qty: int = 0
    timestamp: str = ""


@dataclass
class DepthSignal:
    """Actionable signal from order book analysis."""
    buy_pressure: float       # 0-100 (100 = extreme buy pressure)
    imbalance_ratio: float    # > 1 = buyers dominate, < 1 = sellers dominate
    expected_slippage: float  # Expected slippage in points for our order size
    absorption_detected: bool # Large orders sitting and absorbing selling/buying
    absorption_side: str      # "bid" or "ask" — which side is absorbing
    signal: str               # "strong_buy" / "mild_buy" / "neutral" / "mild_sell" / "strong_sell"
    reasoning: str


class OrderBookAnalyzer:
    """
    Analyzes market depth / Level 2 data for:

    1. BID/ASK IMBALANCE — Who's more aggressive?
       - Total bid qty > total ask qty → buyers stacking = bullish
       - Ratio > 2.0 → strong imbalance

    2. ABSORPTION DETECTION — Large resting orders absorbing flow
       - Price stays flat but volume is heavy → someone is absorbing
       - Bid absorption = institutional buying
       - Ask absorption = institutional selling

    3. SLIPPAGE ESTIMATION — Can we fill at our target price?
       - Sum quantities at each level vs our order size
       - Determines real cost of entry/exit

    4. SPOOFING/PULLING DETECTION — Fake orders disappearing
       - Track order book changes — if large orders vanish repeatedly, fake

    SmartAPI provides 5 levels of market depth via getMarketData(mode="FULL").
    """

    def __init__(self, history_len: int = 20):
        self.snapshot_history: deque = deque(maxlen=history_len)
        self.imbalance_history: deque = deque(maxlen=50)

    def analyze(self, book: OrderBookSnapshot,
                our_order_qty: int = 0) -> DepthSignal:
        """
        Analyze current order book state.

        Args:
            book: Current 5-level OrderBookSnapshot
            our_order_qty: Our intended order quantity (for slippage calc)
        """
        self.snapshot_history.append(book)

        # 1. Basic imbalance
        imbalance = self._calculate_imbalance(book)

        # 2. Weighted imbalance (closer levels matter more)
        weighted_imb = self._weighted_imbalance(book)

        # 3. Absorption detection
        absorption, abs_side = self._detect_absorption()

        # 4. Slippage estimation
        slippage = self._estimate_slippage(book, our_order_qty) if our_order_qty > 0 else 0

        # 5. Aggregate signal
        # Weighted imbalance is the primary signal
        buy_pressure = min(max(weighted_imb * 50 + 50, 0), 100)  # Map to 0-100

        if weighted_imb > 1.5:
            signal = "strong_buy"
        elif weighted_imb > 0.3:
            signal = "mild_buy"
        elif weighted_imb < -1.5:
            signal = "strong_sell"
        elif weighted_imb < -0.3:
            signal = "mild_sell"
        else:
            signal = "neutral"

        # Absorption overrides
        reason_parts = [f"imbalance={weighted_imb:.2f}"]
        if absorption:
            if abs_side == "bid":
                signal = "strong_buy" if signal != "strong_sell" else signal
                reason_parts.append("bid_absorption_detected!")
            elif abs_side == "ask":
                signal = "strong_sell" if signal != "strong_buy" else signal
                reason_parts.append("ask_absorption_detected!")

        return DepthSignal(
            buy_pressure=round(buy_pressure, 1),
            imbalance_ratio=round(imbalance, 2),
            expected_slippage=round(slippage, 2),
            absorption_detected=absorption,
            absorption_side=abs_side,
            signal=signal,
            reasoning=" | ".join(reason_parts)
        )

    def _calculate_imbalance(self, book: OrderBookSnapshot) -> float:
        """
        Simple bid/ask quantity ratio.
        > 1 = more bids than asks = buyers dominating
        """
        if book.total_ask_qty == 0:
            return 5.0 if book.total_bid_qty > 0 else 1.0
        return book.total_bid_qty / book.total_ask_qty

    def _weighted_imbalance(self, book: OrderBookSnapshot) -> float:
        """
        Weighted imbalance: closer-to-market levels get more weight.

        Level 1 (best bid/ask): weight 5
        Level 2: weight 3
        Level 3: weight 2
        Level 4: weight 1
        Level 5: weight 0.5

        Returns: positive = buy pressure, negative = sell pressure
        """
        weights = [5, 3, 2, 1, 0.5]

        weighted_bid = 0
        weighted_ask = 0

        for i, level in enumerate(book.bids[:5]):
            w = weights[i] if i < len(weights) else 0.5
            weighted_bid += level.quantity * w

        for i, level in enumerate(book.asks[:5]):
            w = weights[i] if i < len(weights) else 0.5
            weighted_ask += level.quantity * w

        total = weighted_bid + weighted_ask
        if total == 0:
            return 0

        # Normalize to -2 to +2
        return (weighted_bid - weighted_ask) / total * 4

    def _detect_absorption(self) -> Tuple[bool, str]:
        """
        Absorption: Large orders that DON'T move despite heavy volume.

        Pattern: Price stays in a tight range, but bid/ask quantities
        keep getting refreshed at the same level → someone is absorbing flow.

        We detect this by looking at historical snapshots:
        - If best bid stays same price across multiple snapshots AND bid qty is stable/growing
          while price didn't drop → bid absorption (bullish)
        - Vice versa for asks
        """
        if len(self.snapshot_history) < 5:
            return False, ""

        recent = list(self.snapshot_history)[-5:]

        # Check bid absorption
        bid_prices = [s.bids[0].price for s in recent if s.bids]
        ask_prices = [s.asks[0].price for s in recent if s.asks]

        if len(bid_prices) >= 5:
            # Same best bid price across snapshots = absorption
            if len(set(bid_prices)) <= 2:  # Price barely moved
                bid_qtys = [s.bids[0].quantity for s in recent if s.bids]
                if bid_qtys[-1] >= bid_qtys[0] * 0.8:  # Quantity not depleting
                    return True, "bid"

        if len(ask_prices) >= 5:
            if len(set(ask_prices)) <= 2:
                ask_qtys = [s.asks[0].quantity for s in recent if s.asks]
                if ask_qtys[-1] >= ask_qtys[0] * 0.8:
                    return True, "ask"

        return False, ""

    def _estimate_slippage(self, book: OrderBookSnapshot,
                            order_qty: int) -> float:
        """
        Estimate actual slippage for our order size.

        Walk through ask levels (for buy order) and calculate
        weighted average fill price vs best ask.
        """
        if not book.asks or order_qty <= 0:
            return 0

        best_ask = book.asks[0].price
        remaining = order_qty
        total_cost = 0

        for level in book.asks:
            fill_at_level = min(remaining, level.quantity)
            total_cost += fill_at_level * level.price
            remaining -= fill_at_level
            if remaining <= 0:
                break

        if remaining > 0:
            # Not enough liquidity in visible book
            if book.asks:
                last_ask = book.asks[-1].price
                total_cost += remaining * (last_ask * 1.01)  # Assume 1% worse

        avg_fill = total_cost / order_qty
        slippage = avg_fill - best_ask

        return slippage

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # VOLUME DELTA (Aggressor Analysis)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def analyze_trade_flow(self, trades: list) -> dict:
        """
        Analyze recent trades to determine if buyers or sellers are aggressing.

        A trade at ask price = buyer aggressor (bullish)
        A trade at bid price = seller aggressor (bearish)

        This is the PUREST form of supply/demand data.
        """
        if not trades:
            return {"delta": 0, "buy_volume": 0, "sell_volume": 0, "signal": "neutral"}

        buy_vol = 0
        sell_vol = 0

        for trade in trades:
            price = trade.get("price", 0)
            qty = trade.get("qty", 0)
            # If trade price >= ask → buyer aggressor
            # If trade price <= bid → seller aggressor
            buyer_initiated = trade.get("buyer_initiated", None)

            if buyer_initiated is True or (buyer_initiated is None and
                                           price >= trade.get("ask", price)):
                buy_vol += qty
            else:
                sell_vol += qty

        total = buy_vol + sell_vol
        delta = buy_vol - sell_vol

        if total == 0:
            return {"delta": 0, "buy_volume": 0, "sell_volume": 0, "signal": "neutral"}

        ratio = buy_vol / total

        if ratio > 0.65:
            signal = "strong_buy_flow"
        elif ratio > 0.55:
            signal = "mild_buy_flow"
        elif ratio < 0.35:
            signal = "strong_sell_flow"
        elif ratio < 0.45:
            signal = "mild_sell_flow"
        else:
            signal = "balanced"

        return {
            "delta": delta,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "buy_pct": round(ratio * 100, 1),
            "signal": signal
        }
