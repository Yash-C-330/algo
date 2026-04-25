"""
Risk Management Module

Handles:
1. Position sizing (how many lots based on risk)
2. Stop loss management (initial + trailing)
3. Daily P&L tracking and circuit breakers
4. Max position limits
5. Position persistence to disk (crash recovery)
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, time

from config import (
    INITIAL_CAPITAL, RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT,
    MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY, SLIPPAGE_POINTS,
    MIN_LOT_MULTIPLIER, LOG_DIR
)
from strategies import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position being tracked."""
    id: str
    strategy_name: str
    entry_time: datetime
    entry_premium: float
    current_premium: float
    strike: float
    option_type: str       # "CE" or "PE"
    lots: int
    lot_size: int
    sl_premium: float       # Absolute SL level (premium drops to this)
    target_premium: Optional[float]  # Absolute target level (None if trailing)
    trail_type: str
    highest_premium: float  # For trailing SL
    time_stop_candles: int
    candles_held: int = 0
    is_trailing: bool = False  # SL moved to cost
    pnl: float = 0.0


@dataclass
class DailyState:
    """Track daily performance."""
    date: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    daily_limit_hit: bool = False


class RiskManager:
    """
    Central risk management with capital protection.
    """

    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital = capital
        self.current_capital = capital
        self.positions: dict = {}  # id -> Position
        self.daily_state = DailyState(date=datetime.now().strftime("%Y-%m-%d"))
        self.trade_counter = 0
        self._recent_results = []  # List of recent P&L floats for consecutive loss tracking

    # ──────────────────────────────────────────────
    # Pre-Trade Checks
    # ──────────────────────────────────────────────

    def can_trade(self) -> tuple:
        """Check if we're allowed to take a new trade. Returns (allowed, reason)."""
        # Daily loss limit
        if self.daily_state.realized_pnl <= -(self.current_capital * MAX_DAILY_LOSS_PCT):
            self.daily_state.daily_limit_hit = True
            return False, f"Daily loss limit hit: ₹{self.daily_state.realized_pnl:.0f}"

        # Max positions
        if len(self.positions) >= MAX_OPEN_POSITIONS:
            return False, f"Max positions ({MAX_OPEN_POSITIONS}) reached"

        # Max trades per day
        if self.daily_state.trades_taken >= MAX_TRADES_PER_DAY:
            return False, f"Max trades ({MAX_TRADES_PER_DAY}) reached for today"

        return True, "OK"

    def calculate_lots(self, signal: TradeSignal, lot_size: int,
                       estimated_premium: float, sl_premium_distance: float) -> int:
        """
        Calculate number of lots based on risk per trade.

        risk_per_trade = capital * RISK_PER_TRADE_PCT
        lots = floor(risk_per_trade / (sl_distance * lot_size))
        """
        if estimated_premium <= 0:
            logger.warning("estimated_premium is 0 or negative — cannot size position")
            return 0

        risk_amount = self.current_capital * RISK_PER_TRADE_PCT

        # Account for slippage in SL distance
        effective_sl = max(sl_premium_distance + SLIPPAGE_POINTS, 3.0)  # Minimum ₹3 SL

        if effective_sl <= 0 or lot_size <= 0:
            logger.warning("Invalid SL or lot size for position sizing")
            return 0

        max_loss_per_lot = effective_sl * lot_size
        lots = int(risk_amount / max_loss_per_lot)

        # Ensure minimum lots
        lots = max(lots, MIN_LOT_MULTIPLIER)

        # Check if we can afford the premium
        total_premium_cost = estimated_premium * lot_size * lots
        if total_premium_cost > self.current_capital * 0.5:
            # Don't use more than 50% of capital on one trade
            affordable_lots = int(self.current_capital * 0.5 / (estimated_premium * lot_size))
            if affordable_lots <= 0:
                logger.warning("Premium too expensive: even 1 lot breaches 50% capital cap")
                return 0
            lots = affordable_lots

        # For spreads, check net debit
        if signal is not None and len(signal.legs) > 1:
            # Spread — cost is net debit, typically lower
            lots = max(lots, MIN_LOT_MULTIPLIER)

        logger.info(f"Position sizing: risk=₹{risk_amount:.0f}, SL={effective_sl:.1f}, "
                     f"lots={lots}")
        return lots

    # ──────────────────────────────────────────────
    # Position Tracking
    # ──────────────────────────────────────────────

    def open_position(self, strategy_name: str, entry_premium: float,
                      strike: float, option_type: str, lots: int,
                      lot_size: int, sl_premium: float,
                      target_premium: Optional[float], trail_type: str,
                      time_stop_candles: int) -> str:
        """Register a new position."""
        self.trade_counter += 1
        pos_id = f"{strategy_name}_{self.trade_counter}_{datetime.now().strftime('%H%M%S')}"

        position = Position(
            id=pos_id,
            strategy_name=strategy_name,
            entry_time=datetime.now(),
            entry_premium=entry_premium,
            current_premium=entry_premium,
            strike=strike,
            option_type=option_type,
            lots=lots,
            lot_size=lot_size,
            sl_premium=max(entry_premium - sl_premium, 0.05),  # SL as absolute level, floor at 1 tick
            target_premium=(entry_premium + target_premium) if target_premium is not None else None,
            trail_type=trail_type,
            highest_premium=entry_premium,
            time_stop_candles=time_stop_candles
        )

        self.positions[pos_id] = position
        self.daily_state.trades_taken += 1

        logger.info(f"OPENED: {pos_id} | {option_type} {strike} @ ₹{entry_premium} | "
                     f"SL=₹{position.sl_premium:.1f} | Lots={lots}")
        return pos_id

    def close_position(self, pos_id: str, exit_premium: float, reason: str) -> float:
        """Close a position and record P&L.
        
        Note: In LIVE mode, exit_premium already reflects actual fill
        (including slippage). Only deduct estimated slippage in PAPER mode
        or manual_exit where exit_premium is an estimate.
        """
        if pos_id not in self.positions:
            logger.warning(f"Position {pos_id} not found")
            return 0.0

        pos = self.positions[pos_id]
        pnl_per_unit = exit_premium - pos.entry_premium
        total_pnl = pnl_per_unit * pos.lots * pos.lot_size

        # Only deduct estimated slippage when exit price is not a real fill
        # (paper mode, manual exit, or when we use last-known premium)
        if reason in ("manual_exit",):
            total_pnl -= SLIPPAGE_POINTS * pos.lots * pos.lot_size

        self.daily_state.realized_pnl += total_pnl
        self.current_capital += total_pnl

        if total_pnl > 0:
            self.daily_state.wins += 1
        else:
            self.daily_state.losses += 1

        self._recent_results.append(total_pnl)

        logger.info(f"CLOSED: {pos_id} | Exit=₹{exit_premium} | PnL=₹{total_pnl:.0f} | "
                     f"Reason={reason}")

        del self.positions[pos_id]
        return total_pnl

    # ──────────────────────────────────────────────
    # Position Management (called every candle)
    # ──────────────────────────────────────────────

    def update_position(self, pos_id: str, current_premium: float,
                        supertrend_val: float = None,
                        supertrend_dir: int = None) -> Optional[str]:
        """
        Update position and check exit conditions.
        Returns exit reason if should exit, None otherwise.
        """
        if pos_id not in self.positions:
            return None

        pos = self.positions[pos_id]
        pos.current_premium = current_premium
        pos.candles_held += 1

        # Track highest premium for trailing
        if current_premium > pos.highest_premium:
            pos.highest_premium = current_premium

        # ── Check Stop Loss ──
        if current_premium <= pos.sl_premium:
            return "stop_loss_hit"

        # ── Check Target (for fixed target strategies) ──
        if pos.target_premium is not None and current_premium >= pos.target_premium:
            return "target_hit"

        # ── Check Time Stop ──
        if pos.candles_held >= pos.time_stop_candles:
            if current_premium <= pos.entry_premium:
                return "time_stop_no_profit"

        # ── Trailing Stop Logic ──
        self._update_trailing_sl(pos, current_premium, supertrend_val, supertrend_dir)

        # ── Move SL to cost after 1:1 R:R ──
        if not pos.is_trailing:
            risk = pos.entry_premium - pos.sl_premium
            if risk > 0 and current_premium >= pos.entry_premium + risk:
                # Achieved 1:1 — move SL to breakeven + slippage
                pos.sl_premium = pos.entry_premium + SLIPPAGE_POINTS
                pos.is_trailing = True
                logger.info(f"TRAIL: {pos_id} SL moved to breakeven ₹{pos.sl_premium:.1f}")

        return None

    def _update_trailing_sl(self, pos: Position, current_premium: float,
                            supertrend_val: float = None,
                            supertrend_dir: int = None):
        """Update trailing stop loss based on strategy type.

        Trail types:
          supertrend — 25% peak retracement trail, tightens on ST flip
          fixed      — 30% peak retracement trail (simple)
          breakeven  — Moves SL to cost+slippage once 1:1 achieved, then
                       20% peak trail from there (tighter than fixed)
          percentage — Continuous percentage trail from high-water mark
                       (15% retracement from peak once profitable)
        """

        if pos.trail_type == "supertrend" and supertrend_val is not None:
            # For CE: SL at spot's supertrend (rough delta conversion)
            # Simplified: trail SL using premium high watermark
            if pos.is_trailing:
                trail_sl = pos.highest_premium * 0.75  # 25% from peak
                if trail_sl > pos.sl_premium:
                    pos.sl_premium = trail_sl

            # If Supertrend flips against us, set tight trail (not instant exit)
            if pos.option_type == "CE" and supertrend_dir == -1:
                pos.sl_premium = max(pos.sl_premium, current_premium * 0.92)

            elif pos.option_type == "PE" and supertrend_dir == 1:
                pos.sl_premium = max(pos.sl_premium, current_premium * 0.92)

        elif pos.trail_type == "breakeven":
            # Phase 1: Once 1:1 R:R is achieved, SL moves to breakeven
            # (handled by the generic 1:1 block in update_position)
            # Phase 2: Once trailing, use tighter 20% retracement from peak
            if pos.is_trailing:
                trail_sl = pos.highest_premium * 0.80  # 20% from peak
                if trail_sl > pos.sl_premium:
                    pos.sl_premium = trail_sl
                    logger.debug(f"BREAKEVEN TRAIL: {pos.id} SL → ₹{trail_sl:.1f}")

        elif pos.trail_type == "percentage":
            # Once in profit, trail at 15% from high-water mark
            if current_premium > pos.entry_premium:
                trail_sl = pos.highest_premium * 0.85  # 15% retracement
                # Only ratchet UP, never move SL down
                if trail_sl > pos.sl_premium:
                    pos.sl_premium = trail_sl
                    logger.debug(f"PCT TRAIL: {pos.id} SL → ₹{trail_sl:.1f}")

        elif pos.trail_type == "fixed" and pos.is_trailing:
            # Simple percentage trail from high watermark
            trail_sl = pos.highest_premium * 0.70  # 30% from peak
            if trail_sl > pos.sl_premium:
                pos.sl_premium = trail_sl

    # ──────────────────────────────────────────────
    # EOD and Forced Exits
    # ──────────────────────────────────────────────

    def get_positions_to_close(self) -> list:
        """Get list of position IDs that need forced closure (EOD, etc.)."""
        now = datetime.now().time()
        close_time = time(15, 15)

        if now >= close_time:
            return list(self.positions.keys())
        return []

    def get_daily_summary(self) -> dict:
        """Return daily performance summary."""
        return {
            "date": self.daily_state.date,
            "realized_pnl": round(self.daily_state.realized_pnl, 0),
            "trades": self.daily_state.trades_taken,
            "wins": self.daily_state.wins,
            "losses": self.daily_state.losses,
            "win_rate": (self.daily_state.wins / max(self.daily_state.trades_taken, 1) * 100),
            "capital": round(self.current_capital, 0),
            "open_positions": len(self.positions),
            "daily_limit_hit": self.daily_state.daily_limit_hit
        }

    def reset_daily(self):
        """Reset daily state for new trading day."""
        self.daily_state = DailyState(date=datetime.now().strftime("%Y-%m-%d"))
        self._recent_results.clear()

    # ──────────────────────────────────────────────
    # Anti-Consecutive-Loss Protection
    # ──────────────────────────────────────────────

    def should_reduce_size(self) -> bool:
        """After 2+ consecutive losses, reduce position size by half."""
        if len(self._recent_results) < 2:
            return False
        # Check last N results for consecutive losses
        consecutive = 0
        for pnl in reversed(self._recent_results):
            if pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive >= 2

    # ──────────────────────────────────────────────
    # Position Persistence (Crash Recovery)
    # ──────────────────────────────────────────────

    _POSITIONS_FILE = os.path.join(LOG_DIR, "open_positions.json")

    def save_positions(self):
        """Persist open positions to disk for crash recovery."""
        os.makedirs(LOG_DIR, exist_ok=True)
        data = {}
        for pos_id, pos in self.positions.items():
            data[pos_id] = {
                "strategy_name": pos.strategy_name,
                "entry_time": pos.entry_time.isoformat(),
                "entry_premium": pos.entry_premium,
                "current_premium": pos.current_premium,
                "strike": pos.strike,
                "option_type": pos.option_type,
                "lots": pos.lots,
                "lot_size": pos.lot_size,
                "sl_premium": pos.sl_premium,
                "target_premium": pos.target_premium,
                "trail_type": pos.trail_type,
                "highest_premium": pos.highest_premium,
                "time_stop_candles": pos.time_stop_candles,
                "candles_held": pos.candles_held,
                "is_trailing": pos.is_trailing,
                "pnl": pos.pnl,
            }
        daily = {
            "date": self.daily_state.date,
            "realized_pnl": self.daily_state.realized_pnl,
            "trades_taken": self.daily_state.trades_taken,
            "wins": self.daily_state.wins,
            "losses": self.daily_state.losses,
            "daily_limit_hit": self.daily_state.daily_limit_hit,
            "current_capital": self.current_capital,
            "trade_counter": self.trade_counter,
        }
        payload = {"positions": data, "daily_state": daily}
        try:
            tmp = self._POSITIONS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # Atomic rename so a crash mid-write doesn't corrupt the file
            os.replace(tmp, self._POSITIONS_FILE)
            logger.debug(f"Saved {len(data)} positions to disk")
        except Exception as e:
            logger.warning(f"Failed to save positions: {e}")

    def load_positions(self) -> int:
        """Restore positions from disk after a crash.
        Only loads if the saved date matches today (stale data is ignored).
        Returns number of positions restored."""
        if not os.path.exists(self._POSITIONS_FILE):
            return 0

        try:
            with open(self._POSITIONS_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read positions file: {e}")
            return 0

        daily = payload.get("daily_state", {})
        today = datetime.now().strftime("%Y-%m-%d")
        if daily.get("date") != today:
            logger.info("Positions file is from a previous day — ignoring")
            self._remove_positions_file()
            return 0

        # Restore daily state
        self.daily_state.realized_pnl = daily.get("realized_pnl", 0.0)
        self.daily_state.trades_taken = daily.get("trades_taken", 0)
        self.daily_state.wins = daily.get("wins", 0)
        self.daily_state.losses = daily.get("losses", 0)
        self.daily_state.daily_limit_hit = daily.get("daily_limit_hit", False)
        self.current_capital = daily.get("current_capital", self.capital)
        self.trade_counter = daily.get("trade_counter", 0)

        # Restore positions
        positions_data = payload.get("positions", {})
        for pos_id, p in positions_data.items():
            self.positions[pos_id] = Position(
                id=pos_id,
                strategy_name=p["strategy_name"],
                entry_time=datetime.fromisoformat(p["entry_time"]),
                entry_premium=p["entry_premium"],
                current_premium=p["current_premium"],
                strike=p["strike"],
                option_type=p["option_type"],
                lots=p["lots"],
                lot_size=p["lot_size"],
                sl_premium=p["sl_premium"],
                target_premium=p.get("target_premium"),
                trail_type=p["trail_type"],
                highest_premium=p["highest_premium"],
                time_stop_candles=p["time_stop_candles"],
                candles_held=p.get("candles_held", 0),
                is_trailing=p.get("is_trailing", False),
                pnl=p.get("pnl", 0.0),
            )

        logger.info(f"Restored {len(self.positions)} positions from disk "
                     f"(daily P&L: ₹{self.daily_state.realized_pnl:.0f}, "
                     f"trades: {self.daily_state.trades_taken})")
        return len(self.positions)

    def _remove_positions_file(self):
        """Delete the positions file (called after EOD or when stale)."""
        try:
            if os.path.exists(self._POSITIONS_FILE):
                os.remove(self._POSITIONS_FILE)
        except OSError:
            pass
