"""
Main Trading Engine V3 — Multi-Instrument, Multi-Strategy, Session-Aware.

V3 Upgrades:
- Multi-instrument scanning (trade Nifty/BankNifty/Sensex)
- Micro-pattern detection (7 high-probability setups)
- Scalping module (quick 10-20% premium captures)
- Time-session routing (right strategy at right time)
- Re-entry after stop-outs (when thesis still holds)
- Recovery mode (protect capital during losing streaks)

Daily Flow:
1. Connect to SmartAPI
2. Load symbol master for ALL instruments
3. Fetch pre-market data (previous day levels, VIX, OI)
4. Wait for market open
5. After opening range period (9:30 AM):
   - Multi-instrument scan → pick best instrument
   - Detect regime on best instrument
   - Session router filters allowed strategies
   - Micro-pattern detector finds setups
   - Scalp engine finds quick opportunities
   - Re-entry manager checks for second chances
   - Recovery manager adjusts sizing if losing
   - Execute best signal
   - Monitor and manage open positions
6. Close all positions by 3:15 PM
7. Log daily performance

Run modes:
- LIVE: Real trading with real orders
- PAPER: Simulated trading (logs signals but no real orders)
"""
import logging
import logging.handlers
import os
import sys
import time as time_module
import csv
from datetime import datetime, time, timedelta
from enum import Enum

import pandas as pd

from config import (
    ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET,
    INSTRUMENTS, DEFAULT_INSTRUMENT, INITIAL_CAPITAL,
    REFRESH_INTERVAL_SEC, ENTRY_SCORE_THRESHOLD,
    NO_TRADE_RELAXATION_HOUR, LOG_DIR, TRADE_LOG_FILE, DAILY_PNL_FILE,
    LOG_LEVEL, OI_REFRESH_INTERVAL_SEC
)
from data_fetcher import DataFetcher
from regime_detector import RegimeDetector
from strategies import StrategySelector
from risk_manager import RiskManager
from order_manager import OrderManager
from indicators import supertrend, atr, ema, vwap

# V3 modules
from multi_scanner import MultiInstrumentScanner
from micro_patterns import MicroPatternDetector
from scalping import ScalpEngine, ScalpConfig
from session_router import SessionRouter
from reentry_recovery import ReEntryManager, RecoveryManager
from oi_analyzer import OIAnalyzer
from orderbook_analyzer import OrderBookAnalyzer
from smart_strike_selector import SmartStrikeSelector, ExpiryDayStrategy


class RunMode(Enum):
    LIVE = "live"
    PAPER = "paper"


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # ── File handler: detailed, rotating (10 MB per file, keep 30 backups) ──
    main_log = os.path.join(LOG_DIR, f"trading_{today}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        main_log, maxBytes=10 * 1024 * 1024, backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(file_handler)

    # ── Console handler: concise ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    root.addHandler(console_handler)

    # ── Trade-specific log: easy post-market review ──
    trade_log = os.path.join(LOG_DIR, f"trades_{today}.log")
    trade_handler = logging.handlers.RotatingFileHandler(
        trade_log, maxBytes=5 * 1024 * 1024, backupCount=30, encoding="utf-8"
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    trade_logger = logging.getLogger("trades")
    trade_logger.addHandler(trade_handler)
    trade_logger.propagate = False  # Don't duplicate into main log


logger = logging.getLogger(__name__)


class TradingEngine:
    """
    V3 Main Trading Engine.
    Multi-instrument, multi-strategy, session-aware.
    """

    def __init__(self, mode: RunMode = RunMode.PAPER,
                 instrument_key: str = DEFAULT_INSTRUMENT):
        self.mode = mode
        self.instrument = INSTRUMENTS[instrument_key]
        self.instrument_key = instrument_key

        # Core components
        self.fetcher = DataFetcher(
            ANGEL_API_KEY, ANGEL_CLIENT_ID,
            ANGEL_PASSWORD, ANGEL_TOTP_SECRET
        )
        self.regime_detector = RegimeDetector()
        self.strategy_selector = StrategySelector()
        self.risk_manager = RiskManager(INITIAL_CAPITAL)
        self.order_manager = OrderManager(self.fetcher)

        # V3 components
        self.scanner = MultiInstrumentScanner()
        self.micro_detector = MicroPatternDetector()
        self.scalp_engine = ScalpEngine()
        self.session_router = SessionRouter()
        self.reentry_manager = ReEntryManager()
        self.recovery_manager = RecoveryManager()
        self.oi_analyzer = OIAnalyzer()
        self.orderbook_analyzer = OrderBookAnalyzer()
        self.strike_selector = SmartStrikeSelector(INITIAL_CAPITAL)
        self.expiry_strategy = ExpiryDayStrategy()

        # State
        self.prev_day = {"prev_high": None, "prev_low": None, "prev_close": None}
        self.india_vix = 15.0
        self.current_regime = None
        self.no_trade_relaxed = False
        self.active_instrument_key = instrument_key  # Can change via scanner
        self._last_scan_time = None  # Throttle instrument scanning
        self._scan_interval_sec = 60  # Scan every 60s, not every tick
        self._last_oi_signal = None   # Cache OI signal (refreshes every 60s)
        self._last_oi_time = None
        self._last_oi_snapshot = None
        self._last_chain_data = {}    # Cached option chain for strike selector
        self._recent_entry_keys = set()  # Duplicate signal guard per loop iteration

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    def initialize(self) -> bool:
        """Pre-market setup. Call before market opens."""
        logger.info("=" * 60)
        logger.info(f"TRADING ENGINE STARTING | Mode={self.mode.value} | "
                     f"Instrument={self.instrument_key}")
        logger.info(f"Capital=₹{self.risk_manager.current_capital:,.0f} | "
                     f"Lot Size={self.instrument.lot_size}")
        logger.info("=" * 60)

        # Connect to SmartAPI
        if self.mode == RunMode.LIVE:
            if not ANGEL_API_KEY:
                logger.error("API credentials not set in config.py!")
                return False
            if not self.fetcher.connect():
                return False

            # Load symbol master for token resolution
            logger.info("Loading symbol master...")
            self.fetcher.load_symbol_master()
        else:
            # Paper mode can still use live market data feed while skipping real orders.
            if ANGEL_API_KEY:
                if self.fetcher.connect():
                    logger.info("Paper mode connected to SmartAPI for market data")
                    logger.info("Loading symbol master...")
                    self.fetcher.load_symbol_master()
                else:
                    logger.warning("Paper mode data feed unavailable; loop will wait for market data")
            else:
                logger.warning("Paper mode running without SmartAPI credentials; use backtest.py for offline simulation")

        # Fetch pre-market data
        self._fetch_premarket_data()

        logger.info(f"Previous Day: H={self.prev_day['prev_high']} "
                     f"L={self.prev_day['prev_low']} C={self.prev_day['prev_close']}")
        logger.info(f"India VIX: {self.india_vix}")

        # Crash recovery: restore positions from disk
        restored = self.risk_manager.load_positions()
        if restored > 0:
            logger.info(f"CRASH RECOVERY: {restored} positions restored from disk")

        logger.info("Initialization complete. Waiting for market...")

        return True

    def run(self):
        """
        Main trading loop. Runs until market close.
        """
        if not self.initialize():
            logger.error("Initialization failed. Exiting.")
            return

        try:
            self._wait_for_market_open()
            self._wait_for_opening_range()
            self._trading_loop()
        except KeyboardInterrupt:
            logger.info("Manual interruption — closing positions...")
            self._close_all_positions("manual_stop")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self._close_all_positions("error")
        finally:
            self._end_of_day()

    # ──────────────────────────────────────────────
    # Pre-Market
    # ──────────────────────────────────────────────

    def _fetch_premarket_data(self):
        """Fetch previous day levels and VIX."""
        if self.mode == RunMode.LIVE:
            self.prev_day = self.fetcher.get_previous_day_ohlc(
                self.instrument.token, self.instrument.exchange
            )
            self.india_vix = self.fetcher.get_india_vix()
        else:
            # Paper mode — use placeholder data or load from file
            logger.info("Paper mode: Using placeholder pre-market data")
            # In paper mode, these will be populated from historical data
            self.prev_day = {"prev_high": 0, "prev_low": 0, "prev_close": 0}
            self.india_vix = 15.0

    def _wait_for_market_open(self):
        """Wait until 9:15 AM IST."""
        market_open = time(9, 15)
        now = datetime.now().time()

        if now < market_open:
            wait_secs = (datetime.combine(datetime.today(), market_open) -
                         datetime.now()).total_seconds()
            logger.info(f"Waiting {wait_secs / 60:.1f} min for market open...")

            # Don't actually block in paper mode
            if self.mode == RunMode.LIVE:
                time_module.sleep(max(0, wait_secs))

    def _wait_for_opening_range(self):
        """Wait until 9:30 AM for opening range to form."""
        orb_end = time(9, 30)
        now = datetime.now().time()

        if now < orb_end:
            wait_secs = (datetime.combine(datetime.today(), orb_end) -
                         datetime.now()).total_seconds()
            logger.info(f"Waiting {wait_secs / 60:.1f} min for opening range...")

            if self.mode == RunMode.LIVE:
                time_module.sleep(max(0, wait_secs))

    # ──────────────────────────────────────────────
    # Main Trading Loop
    # ──────────────────────────────────────────────

    def _trading_loop(self):
        """
        V3 Core loop:
        scan instruments → detect regime → session filter → micro-patterns →
        scalps → re-entries → evaluate strategies → execute → manage
        """
        close_time = time(15, 15)

        while True:
            now = datetime.now()

            # End of day
            if now.time() >= close_time:
                logger.info("Market close time reached")
                self._close_all_positions("eod")
                break

            # Recovery check — should we stop trading entirely?
            should_stop, stop_reason = self.recovery_manager.should_stop_trading(
                self.risk_manager.current_capital
            )
            if should_stop:
                logger.warning(f"STOPPING TRADING: {stop_reason}")
                self._close_all_positions("recovery_stop")
                break

            try:
                # Reset duplicate signal guard each iteration
                self._recent_entry_keys.clear()

                # ── Session health check — reconnect if needed ──
                if self.mode == RunMode.LIVE and not self.fetcher.is_session_healthy():
                    logger.warning("API session unhealthy — attempting reconnect...")
                    if self.fetcher.reconnect(max_attempts=3, backoff_base=5.0):
                        logger.info("Session restored — resuming trading")
                    else:
                        logger.error("Reconnect failed — waiting 60s before retry")
                        time_module.sleep(60)
                        continue

                # 0. Session router — what strategies are allowed now?
                session_config = self.session_router.get_current_config(
                    now, is_expiry_day=self._is_expiry_day()
                )

                can_trade_session, session_reason = self.session_router.can_trade_in_session(
                    now, is_expiry_day=self._is_expiry_day()
                )

                # 1. Multi-instrument scan (throttled — every 60s, not every tick)
                if (self._last_scan_time is None or
                        (now - self._last_scan_time).total_seconds() >= self._scan_interval_sec):
                    best_instrument = self._scan_instruments()
                    self._last_scan_time = now
                    if best_instrument and best_instrument != self.active_instrument_key:
                        logger.info(f"INSTRUMENT SWITCH: {self.active_instrument_key} → {best_instrument}")
                        self.active_instrument_key = best_instrument
                        self.instrument = INSTRUMENTS[best_instrument]
                        self._fetch_premarket_data()  # Refresh levels for new instrument

                # 2. Fetch latest data
                df_5min, df_15min, spot_price = self._fetch_market_data()
                if df_5min is None or df_5min.empty or spot_price is None:
                    logger.warning("No market data available, retrying...")
                    time_module.sleep(REFRESH_INTERVAL_SEC)
                    continue

                # Stale data guard: skip if last candle is too old (>10 min)
                if self.mode == RunMode.LIVE and not df_5min.empty:
                    last_candle_time = df_5min.index[-1]
                    if hasattr(last_candle_time, 'to_pydatetime'):
                        last_candle_time = last_candle_time.to_pydatetime()
                    # Make timezone-naive for comparison
                    if last_candle_time.tzinfo is not None:
                        last_candle_time = last_candle_time.replace(tzinfo=None)
                    candle_age = (now - last_candle_time).total_seconds()
                    if candle_age > 600:  # >10 minutes old
                        logger.warning(f"Stale data detected: last candle {candle_age:.0f}s old. Skipping.")
                        time_module.sleep(REFRESH_INTERVAL_SEC)
                        continue

                # 3. Fetch OI and orderbook signals (throttled to reduce API load)
                oi_signal = None
                orderbook_signal = None
                if self.mode == RunMode.LIVE:
                    oi_signal, orderbook_signal = self._fetch_oi_and_orderbook(
                        spot_price
                    )

                # 4. Detect regime (with OI + orderbook data)
                regime_state = self.regime_detector.detect(
                    df_5min, df_15min,
                    self.prev_day.get("prev_high", 0),
                    self.prev_day.get("prev_low", 0),
                    self.prev_day.get("prev_close", 0),
                    self.india_vix,
                    oi_signal=oi_signal,
                    orderbook_signal=orderbook_signal
                )
                self.current_regime = regime_state

                logger.info(f"[{self.active_instrument_key}] "
                            f"REGIME: {regime_state.regime.value} | "
                            f"Score={regime_state.total_score} | "
                            f"Session={session_config.session.value}")

                # 5. Manage existing positions (FIRST — exit fast)
                self._manage_positions(df_5min, spot_price)

                # 6. Check re-entry opportunities (stopped-out trades)
                if can_trade_session and session_config.allow_new_entries:
                    self._check_reentries(df_5min, spot_price, regime_state)

                # 7. Scalping opportunities (own flag — can run even when
                #    allow_new_entries=False, e.g. CLOSING_PLAY session)
                if can_trade_session and session_config.allow_scalping:
                    self._check_scalps(df_5min, spot_price, now)

                # 8. Micro-pattern signals (own flag)
                if can_trade_session and session_config.allow_micro_patterns:
                    self._check_micro_patterns(df_5min, spot_price, regime_state, session_config)

                # 9. Standard strategy entries (momentum, ORB, debit spread, mean reversion)
                if can_trade_session and session_config.allow_new_entries:
                    self._evaluate_new_entries(df_5min, regime_state, spot_price, session_config)

            except Exception as e:
                logger.error(f"Loop iteration error: {e}", exc_info=True)

            # Sleep between iterations
            time_module.sleep(REFRESH_INTERVAL_SEC)

    # ──────────────────────────────────────────────
    # Data Fetching
    # ──────────────────────────────────────────────

    def _fetch_market_data(self):
        """Fetch 5-min, 15-min candles and spot price."""
        if self.mode == RunMode.LIVE or self.fetcher.smart_api is not None:
            df_5min = self.fetcher.get_5min_data(
                self.instrument.token, self.instrument.exchange, 3
            )
            df_15min = self.fetcher.get_15min_data(
                self.instrument.token, self.instrument.exchange, 5
            )
            spot_price = self.fetcher.get_spot_price(
                self.instrument.token, self.instrument.exchange
            )
            return df_5min, df_15min, spot_price
        else:
            # Offline paper mode (no data feed) — handled by backtester script.
            return None, None, None

    def _fetch_oi_and_orderbook(self, spot_price: float):
        """Fetch OI and orderbook signals. OI is throttled to 60s refresh.
        Returns (oi_signal, orderbook_dict_or_None)."""
        oi_signal = self._last_oi_signal
        orderbook_signal = None
        now = datetime.now()

        # OI: refresh every 60s (API rate limit friendly)
        if (self._last_oi_time is None or
                (now - self._last_oi_time).total_seconds() >= OI_REFRESH_INTERVAL_SEC):
            try:
                chain_data = self.fetcher.get_option_chain(
                    self.active_instrument_key, spot_price,
                    self.instrument.strike_gap
                )
                if chain_data:
                    self._last_chain_data = chain_data
                    oi_signal = self.oi_analyzer.analyze(
                        chain_data, spot_price, self._last_oi_snapshot
                    )
                    self._last_oi_signal = oi_signal
                    # Store snapshot for change detection on next fetch
                    from oi_analyzer import OISnapshot
                    self._last_oi_snapshot = self.oi_analyzer.oi_history[-1] \
                        if self.oi_analyzer.oi_history else None
                    self._last_oi_time = now
            except Exception as e:
                logger.debug(f"OI fetch error: {e}")

        # Orderbook: fetch index depth every tick (lightweight call)
        try:
            book = self.fetcher.get_market_depth(
                self.instrument.exchange, "",
                self.instrument.token
            )
            if book and book.bids:
                depth_signal = self.orderbook_analyzer.analyze(book)
                orderbook_signal = {
                    "signal": depth_signal.signal,
                    "buy_pressure": depth_signal.buy_pressure,
                    "absorption_detected": depth_signal.absorption_detected,
                    "absorption_side": depth_signal.absorption_side,
                }
        except Exception as e:
            logger.debug(f"Orderbook fetch error: {e}")

        return oi_signal, orderbook_signal

    # ──────────────────────────────────────────────
    # Position Management
    # ──────────────────────────────────────────────

    def _manage_positions(self, df_5min: pd.DataFrame, spot_price: float):
        """Update all open positions and check exit conditions.
        Also detects positions that were manually exited by the user."""
        if not self.risk_manager.positions:
            return

        # Get supertrend for trailing
        st = supertrend(df_5min)
        st_val = st["supertrend"].iloc[-1] if not st.empty else None
        st_dir = st["st_direction"].iloc[-1] if not st.empty else None

        positions_to_close = []
        manually_exited = []

        # In LIVE mode, fetch broker positions once to compare
        broker_positions = {}
        if self.mode == RunMode.LIVE:
            broker_positions = self.fetcher.get_open_positions() or {}

        for pos_id, pos in list(self.risk_manager.positions.items()):
            # Get current option premium
            if self.mode == RunMode.LIVE:
                # Check if position was manually exited by user
                info = self.fetcher.find_nearest_expiry_token(
                    self.instrument.symbol, pos.strike, pos.option_type
                )
                if info and broker_positions:
                    broker_qty = broker_positions.get(info["symbol"], 0)
                    expected_qty = pos.lots * pos.lot_size
                    if broker_qty < expected_qty:
                        # Position no longer at broker — user manually exited
                        logger.warning(
                            f"MANUAL EXIT DETECTED: {pos_id} | {info['symbol']} | "
                            f"Broker qty={broker_qty}, expected={expected_qty}"
                        )
                        manually_exited.append(pos_id)
                        continue

                current_premium = self.fetcher.get_option_ltp(
                    self.instrument.symbol, pos.strike,
                    pos.option_type, "",  # Expiry resolved dynamically
                    self.instrument.option_exchange
                )
                if current_premium is None:
                    continue
            else:
                # Paper mode fallback: keep the last known premium so candle/time-based
                # exits still progress even without option tick data.
                current_premium = pos.current_premium

            # Update position and check exit
            exit_reason = self.risk_manager.update_position(
                pos_id, current_premium, st_val, st_dir
            )

            if exit_reason:
                positions_to_close.append((pos_id, current_premium, exit_reason))

        # Handle manually exited positions — close in risk manager WITHOUT selling
        for pos_id in manually_exited:
            pos = self.risk_manager.positions.get(pos_id)
            if pos:
                logger.info(f"Cleaning up manually exited position: {pos_id}")
                # Use last known premium for P&L calculation
                pnl = self.risk_manager.close_position(
                    pos_id, pos.current_premium, "manual_exit"
                )
                self._log_trade(pos, pos.current_premium, pnl, "manual_exit")
                self.recovery_manager.record_trade_result(pnl)
        if manually_exited:
            self.risk_manager.save_positions()

        # Execute exits for algo-triggered closes
        for pos_id, exit_premium, reason in positions_to_close:
            self._exit_position(pos_id, exit_premium, reason)

    def _exit_position(self, pos_id: str, exit_premium: float, reason: str):
        """Exit a position — place sell order and record P&L.
        Verifies position exists at broker before selling to prevent
        unintended short positions from manual exits."""
        pos = self.risk_manager.positions.get(pos_id)
        if pos is None:
            return

        if self.mode == RunMode.LIVE:
            info = self.fetcher.find_nearest_expiry_token(
                self.instrument.symbol, pos.strike, pos.option_type
            )
            if info is None:
                logger.error(
                    f"Cannot resolve token for exit: {pos.option_type} {pos.strike}. "
                    f"Keeping position open in tracker to avoid orphaning live risk."
                )
                return
            if info:
                # Verify position still exists at broker before selling
                expected_qty = pos.lots * pos.lot_size
                if not self.fetcher.verify_position_exists(info["symbol"], expected_qty):
                    logger.warning(
                        f"MANUAL EXIT DETECTED in _exit_position: {pos_id} | "
                        f"{info['symbol']} no longer at broker. "
                        f"Skipping sell order to avoid creating short position."
                    )
                    # Still close internal tracking — use last known premium
                    pnl = self.risk_manager.close_position(pos_id, pos.current_premium, "manual_exit")
                    self._log_trade(pos, pos.current_premium, pnl, "manual_exit")
                    self.recovery_manager.record_trade_result(pnl)
                    self.risk_manager.save_positions()
                    return

                result = self.order_manager.place_option_sell(
                    info["symbol"], info["token"], info["exchange"],
                    pos.lots, pos.lot_size, exit_premium
                )
                if result.success:
                    exit_premium = result.fill_price
                else:
                    # Emergency exit — verify position still exists before retry
                    if self.fetcher.verify_position_exists(info["symbol"], expected_qty):
                        result = self.order_manager.emergency_exit(
                            info["symbol"], info["token"], info["exchange"],
                            pos.lots * pos.lot_size
                        )
                        if result.success:
                            exit_premium = result.fill_price
                    else:
                        logger.warning(
                            f"Position {info['symbol']} gone before emergency exit. "
                            f"Likely manually closed."
                        )
                        pnl = self.risk_manager.close_position(pos_id, pos.current_premium, "manual_exit")
                        self._log_trade(pos, pos.current_premium, pnl, "manual_exit")
                        self.recovery_manager.record_trade_result(pnl)
                        self.risk_manager.save_positions()
                        return

        pnl = self.risk_manager.close_position(pos_id, exit_premium, reason)
        self._log_trade(pos, exit_premium, pnl, reason)
        self.risk_manager.save_positions()

        # V3: Feed results to recovery and re-entry managers
        self.recovery_manager.record_trade_result(pnl)

        if pos.strategy_name.startswith("scalp_"):
            self.scalp_engine.record_scalp_result(pnl, self.risk_manager.current_capital)

        # If stopped out (not target hit or EOD), register for potential re-entry
        if reason in ("sl_hit", "stop_loss", "stop_loss_hit") and pnl < 0:
            direction = 1 if pos.option_type == "CE" else -1
            self.reentry_manager.record_stop_out(
                instrument=self.active_instrument_key,
                direction=direction,
                strategy_name=pos.strategy_name,
                regime_score=self.current_regime.total_score if self.current_regime else 0,
                stop_time=datetime.now(),
                stop_price=pos.strike,
                sl_price=pos.sl_premium,
                entry_price=pos.entry_premium,
                loss_amount=abs(pnl),
            )

    # ──────────────────────────────────────────────
    # New Entry Evaluation
    # ──────────────────────────────────────────────

    def _evaluate_new_entries(self, df_5min: pd.DataFrame, regime_state,
                              spot_price: float, session_config=None):
        """Evaluate recommended strategies and enter if signal is strong enough."""
        # Pre-checks
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"Cannot trade: {reason}")
            return

        # Relax threshold if no trades by configured hour
        threshold = ENTRY_SCORE_THRESHOLD
        if (not self.no_trade_relaxed and
                datetime.now().hour >= NO_TRADE_RELAXATION_HOUR and
                self.risk_manager.daily_state.trades_taken == 0):
            threshold -= 10
            self.no_trade_relaxed = True
            logger.info(f"No trades yet — relaxing threshold to {threshold}")

        # V3: Session-based threshold adjustment
        if session_config:
            threshold = self.session_router.adjust_threshold(
                int(threshold), datetime.now(), self._is_expiry_day()
            )

        # V3: Recovery mode increases threshold
        threshold += self.recovery_manager.get_threshold_adjustment()

        # V3: Recovery mode minimum score override
        min_score = self.recovery_manager.get_min_score_override()
        if min_score and threshold < min_score:
            threshold = min_score

        # V3: Expiry-day max-pain gravity filter
        #   On expiry days, if the ExpiryDayStrategy indicates strong gravity
        #   toward max pain, filter out signals that go AGAINST that pull.
        expiry_mp_info = None
        if self._is_expiry_day() and self._last_oi_signal:
            max_pain = self._last_oi_signal.max_pain
            if max_pain > 0:
                expiry_mp_info = self.expiry_strategy.should_trade_toward_max_pain(
                    spot_price, max_pain, datetime.now().hour
                )
                if expiry_mp_info and expiry_mp_info["gravity_strength"] >= 0.5:
                    logger.info(f"EXPIRY MAX-PAIN: {expiry_mp_info['recommendation']}")

        # Filter strategies by session
        allowed = None
        if session_config:
            allowed = self.session_router.get_allowed_strategies(
                datetime.now(), self._is_expiry_day()
            )

        # Intersect regime recommendations with session-allowed strategies
        if allowed is not None:
            filtered = [s for s in regime_state.recommended_strategies if s in allowed]
        else:
            filtered = regime_state.recommended_strategies

        # Get best signal
        signal = self.strategy_selector.select_best(
            df_5min=df_5min,
            regime_score=regime_state.total_score,
            recommended=filtered,
            spot_price=spot_price,
            strike_gap=self.instrument.strike_gap,
            pdh=self.prev_day.get("prev_high"),
            pdl=self.prev_day.get("prev_low")
        )

        if signal is None:
            logger.debug("No signal generated")
            return

        # Expiry-day max-pain gravity adjustment:
        # Boost signals that align with max-pain pull, penalize opposing signals.
        if expiry_mp_info and expiry_mp_info["gravity_strength"] >= 0.5:
            mp_direction = expiry_mp_info["direction"]  # "bullish" / "bearish" / "neutral"
            gravity = expiry_mp_info["gravity_strength"]
            signal_leg = signal.legs[0] if signal.legs else None

            if signal_leg and mp_direction != "neutral":
                signal_is_bullish = signal_leg.option_type.value == "CE"
                mp_is_bullish = mp_direction == "bullish"

                if signal_is_bullish == mp_is_bullish:
                    # Signal aligns with max-pain pull — boost score
                    boost = int(10 * gravity)
                    signal.score += boost
                    logger.info(f"EXPIRY BOOST: +{boost} pts (aligned with MP gravity)")
                else:
                    # Signal opposes max-pain pull — penalize
                    penalty = int(15 * gravity)
                    signal.score -= penalty
                    logger.info(f"EXPIRY PENALTY: -{penalty} pts (opposing MP gravity)")

        if signal.score < threshold:
            logger.debug(f"Signal score {signal.score} below threshold {threshold}")
            return

        logger.info(f"SIGNAL: {signal.strategy_name} | Score={signal.score} | "
                     f"{signal.reason}")

        # Execute the trade
        self._execute_entry(signal, spot_price, df_5min)

    def _execute_entry(self, signal, spot_price: float, df_5min: pd.DataFrame):
        """Execute a trade signal.
        Uses SmartStrikeSelector to pick optimal strike — adapts if ATM
        is too costly or has poor liquidity/delta."""
        leg = signal.legs[0]

        # Duplicate signal guard — prevent multiple entries on the same
        # strike + direction in the same loop iteration (across strategies,
        # micro-patterns, scalps, and re-entries)
        entry_key = (leg.option_type.value, leg.strike, self.active_instrument_key)
        if entry_key in self._recent_entry_keys:
            logger.debug(f"Duplicate entry suppressed: {entry_key}")
            return
        self._recent_entry_keys.add(entry_key)

        atr_val = atr(df_5min).iloc[-1] if not df_5min.empty else 50

        # Use SmartStrikeSelector to pick the best strike
        # (adapts to cost, liquidity, delta, OI — not hardcoded ATM)
        if self.mode == RunMode.LIVE and self._last_chain_data:
            try:
                days_to_expiry = self._estimate_days_to_expiry()
                selection = self.strike_selector.select_optimal_strike(
                    spot_price=spot_price,
                    option_type=leg.option_type.value,
                    strike_gap=self.instrument.strike_gap,
                    lot_size=self.instrument.lot_size,
                    chain_data=self._last_chain_data,
                    days_to_expiry=days_to_expiry,
                    regime_strength=abs(self.current_regime.total_score) if self.current_regime else 50,
                    strategy_name=signal.strategy_name,
                    risk_per_trade=self.risk_manager.current_capital * 0.03,
                    iv=self.india_vix,
                )
                # Override the strategy's default strike with selector's choice
                if selection.strike != leg.strike:
                    logger.info(
                        f"STRIKE ADAPTED: {leg.strike} → {selection.strike} | "
                        f"Delta={selection.estimated_delta} | "
                        f"Liquidity={selection.liquidity_score} | "
                        f"{selection.reason}"
                    )
                    leg.strike = selection.strike
            except Exception as e:
                logger.debug(f"Strike selector error, using default: {e}")

        # Fetch real option LTP for accurate sizing (LIVE mode)
        estimated_premium = atr_val * 0.8  # Fallback estimate
        if self.mode == RunMode.LIVE:
            leg = signal.legs[0]
            info = self.fetcher.find_nearest_expiry_token(
                self.active_instrument_key, leg.strike, leg.option_type.value
            )
            if info:
                real_ltp = self.fetcher.get_ltp(
                    info["exchange"], info["symbol"], info["token"]
                )
                if real_ltp and real_ltp > 0:
                    estimated_premium = real_ltp
                    logger.info(f"Real LTP for sizing: ₹{real_ltp}")

        # Calculate SL distance based on strategy
        if signal.strategy_name == "momentum_buy":
            from strategies import MomentumBuyStrategy
            strat = MomentumBuyStrategy()
            sl_dist, _ = strat.compute_sl_target(estimated_premium, atr_val)
        elif signal.strategy_name == "debit_spread":
            sl_dist = estimated_premium * 0.50  # 50% of debit
        elif signal.strategy_name == "orb":
            from indicators import opening_range as calc_or
            or_data = calc_or(df_5min)
            or_range = or_data.get("or_range", atr_val)
            from strategies import ORBStrategy
            strat = ORBStrategy()
            sl_dist, _ = strat.compute_sl_target(
                estimated_premium, or_range, spot_price
            )
        else:  # mean_reversion
            sl_dist = estimated_premium * 0.25 + 2

        # Position sizing
        lots = self.risk_manager.calculate_lots(
            signal, self.instrument.lot_size,
            estimated_premium, sl_dist
        )

        if lots <= 0:
            logger.warning("Position sizing returned 0 lots — skipping")
            return

        # Reduce size after consecutive losses
        if self.risk_manager.should_reduce_size():
            lots = max(1, lots // 2)
            logger.info(f"Reducing size due to consecutive losses: {lots} lots")

        if self.mode == RunMode.LIVE:
            results = self.order_manager.execute_signal(
                signal, self.instrument.lot_size, lots,
                instrument_key=self.active_instrument_key
            )

            # Check if all legs filled
            if not results or not all(r.success for r in results):
                logger.error("Not all legs filled — trade aborted")
                return

            # Use actual fill price
            entry_premium = results[0].fill_price

            # Recompute SL relative to actual fill, not estimate
            sl_ratio = sl_dist / estimated_premium if estimated_premium > 0 else 0.30
            sl_dist = entry_premium * sl_ratio
        else:
            # Paper mode — use estimate
            entry_premium = estimated_premium
            logger.info(f"PAPER TRADE: {signal.strategy_name} | "
                         f"Entry=₹{entry_premium:.1f} | "
                         f"SL_dist=₹{sl_dist:.1f} | Lots={lots}")

        # Register position
        leg = signal.legs[0]
        pos_id = self.risk_manager.open_position(
            strategy_name=signal.strategy_name,
            entry_premium=entry_premium,
            strike=leg.strike,
            option_type=leg.option_type.value,
            lots=lots,
            lot_size=self.instrument.lot_size,
            sl_premium=sl_dist,
            target_premium=signal.target_premium if signal.target_premium else None,
            trail_type=signal.trail_type,
            time_stop_candles=signal.time_stop_candles
        )
        self.risk_manager.save_positions()

        logger.info(f"POSITION OPENED: {pos_id}")

    # ──────────────────────────────────────────────
    # V3: Multi-Instrument Scanning
    # ──────────────────────────────────────────────

    def _scan_instruments(self) -> str:
        """Scan all instruments and return the best one to trade.
        Sequential fetch with rate-limiting to respect Angel API limits
        (getCandleData: 3 req/sec, ltpData: 10 req/sec)."""
        if self.mode != RunMode.LIVE:
            return self.instrument_key  # Paper mode stays on default

        try:
            data_dict = {}
            prev_days = {}

            for key, config in INSTRUMENTS.items():
                try:
                    df_5 = self.fetcher.get_5min_data(config.token, config.exchange, 3)
                    time_module.sleep(0.35)  # getCandleData: 3 req/sec

                    df_15 = self.fetcher.get_15min_data(config.token, config.exchange, 5)
                    time_module.sleep(0.35)

                    spot = self.fetcher.get_spot_price(config.token, config.exchange)
                    time_module.sleep(0.15)  # ltpData has higher limit

                    prev = self.fetcher.get_previous_day_ohlc(config.token, config.exchange)
                    time_module.sleep(0.35)

                    if (df_5 is not None and not df_5.empty and
                            spot is not None and
                            df_15 is not None and not df_15.empty):
                        data_dict[key] = {"5min": df_5, "15min": df_15, "spot": spot}
                        prev_days[key] = prev
                except Exception as e:
                    logger.debug(f"Scan fetch error for {key}: {e}")

            if not data_dict:
                return self.active_instrument_key

            results = self.scanner.scan_all(data_dict, prev_days, self.india_vix)
            best = self.scanner.get_best_instrument(results)
            if best:
                return best.instrument_key
        except Exception as e:
            logger.debug(f"Scanner error: {e}")

        return self.active_instrument_key

    # ──────────────────────────────────────────────
    # V3: Micro-Pattern Detection
    # ──────────────────────────────────────────────

    def _check_micro_patterns(self, df_5min: pd.DataFrame, spot_price: float,
                               regime_state, session_config):
        """Scan for micro-patterns and trade if found."""
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return

        patterns = self.micro_detector.scan(
            df_5min, spot_price,
            pdh=self.prev_day.get("prev_high"),
            pdl=self.prev_day.get("prev_low"),
        )

        if not patterns:
            return

        # Pick strongest pattern
        best = max(patterns, key=lambda p: p.strength)

        # Pattern must align with regime
        if regime_state.total_score > 20 and best.direction < 0:
            return  # Bullish regime, bearish pattern — skip
        if regime_state.total_score < -20 and best.direction > 0:
            return  # Bearish regime, bullish pattern — skip

        # On expiry days, skip patterns opposing strong max-pain gravity
        if self._is_expiry_day() and self._last_oi_signal:
            max_pain = self._last_oi_signal.max_pain
            if max_pain > 0:
                mp_info = self.expiry_strategy.should_trade_toward_max_pain(
                    spot_price, max_pain, datetime.now().hour
                )
                if mp_info and mp_info["gravity_strength"] >= 0.7:
                    mp_bullish = mp_info["direction"] == "bullish"
                    pattern_bullish = best.direction > 0
                    if mp_bullish != pattern_bullish:
                        logger.info(f"EXPIRY FILTER: Skipping {best.pattern.value} "
                                    f"(opposes strong MP gravity)")
                        return

        # Apply session target/SL adjustments
        target = self.session_router.adjust_target(
            best.target_price, datetime.now(), self._is_expiry_day()
        )
        sl = self.session_router.adjust_sl(
            best.sl_price, datetime.now(), self._is_expiry_day()
        )

        logger.info(f"MICRO-PATTERN: {best.pattern.value} | "
                     f"Strength={best.strength} | Dir={'BUY' if best.direction > 0 else 'SELL'} | "
                     f"{best.reason}")

        # Create a signal-like object and execute
        self._execute_pattern_trade(best, spot_price, df_5min)

    def _execute_pattern_trade(self, pattern, spot_price, df_5min):
        """Execute a micro-pattern or scalp trade."""
        option_type = "CE" if pattern.direction > 0 else "PE"
        strike = round(spot_price / self.instrument.strike_gap) * self.instrument.strike_gap

        # Duplicate signal guard
        entry_key = (option_type, strike, self.active_instrument_key)
        if entry_key in self._recent_entry_keys:
            logger.debug(f"Duplicate pattern entry suppressed: {entry_key}")
            return
        self._recent_entry_keys.add(entry_key)

        atr_val = atr(df_5min).iloc[-1] if not df_5min.empty else 50
        estimated_premium = atr_val * 0.8
        original_estimate = estimated_premium

        # Fetch real LTP for accurate sizing (LIVE mode)
        info = None
        if self.mode == RunMode.LIVE:
            info = self.fetcher.find_nearest_expiry_token(
                self.active_instrument_key, strike, option_type
            )
            if info:
                real_ltp = self.fetcher.get_ltp(
                    info["exchange"], info["symbol"], info["token"]
                )
                if real_ltp and real_ltp > 0:
                    estimated_premium = real_ltp

        sl_points = abs(pattern.entry_price - pattern.sl_price)
        sl_dist = self._convert_spot_points_to_premium(
            spot_points=sl_points,
            option_type=option_type,
            strike=strike,
            spot_price=spot_price,
            min_premium_move=0.5,
        )
        lots = self.risk_manager.calculate_lots(
            None, self.instrument.lot_size,
            estimated_premium, sl_dist
        )

        if lots <= 0:
            logger.warning("Pattern trade skipped: risk sizing returned 0 lots")
            return

        # Recovery mode — reduce size
        lots = max(1, int(lots * self.recovery_manager.get_size_multiplier()))

        if self.mode == RunMode.LIVE:
            if info:
                result = self.order_manager.place_option_buy(
                    info["symbol"], info["token"], info["exchange"],
                    lots, self.instrument.lot_size, estimated_premium
                )
                if result.success:
                    # Recompute SL relative to actual fill price
                    fill_ratio = result.fill_price / estimated_premium if estimated_premium > 0 else 1.0
                    sl_dist = sl_dist * fill_ratio
                    estimated_premium = result.fill_price
                else:
                    return
            else:
                logger.error(f"Could not find token for pattern trade: {option_type} {strike}")
                return
        else:
            pattern_name = pattern.pattern.value if hasattr(pattern, 'pattern') else "scalp"
            logger.info(f"PAPER TRADE: {pattern_name} | "
                         f"Entry=₹{estimated_premium:.1f} | Lots={lots}")

        if hasattr(pattern, "target_price"):
            target_points = abs(pattern.target_price - pattern.entry_price)
            target_premium = self._convert_spot_points_to_premium(
                spot_points=target_points,
                option_type=option_type,
                strike=strike,
                spot_price=spot_price,
                min_premium_move=0.5,
            )
        else:
            target_premium = estimated_premium * 0.15

        pos_id = self.risk_manager.open_position(
            strategy_name=pattern.pattern.value if hasattr(pattern, 'pattern') else "scalp",
            entry_premium=estimated_premium,
            strike=strike,
            option_type=option_type,
            lots=lots,
            lot_size=self.instrument.lot_size,
            sl_premium=sl_dist,
            target_premium=target_premium,
            trail_type="breakeven",
            time_stop_candles=6,
        )
        self.risk_manager.save_positions()
        self.session_router.record_trade(datetime.now())
        logger.info(f"POSITION OPENED: {pos_id}")

    # ──────────────────────────────────────────────
    # V3: Scalping
    # ──────────────────────────────────────────────

    def _check_scalps(self, df_5min: pd.DataFrame, spot_price: float,
                       current_time: datetime):
        """Check for scalp opportunities."""
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return

        can_scalp, scalp_reason = self.scalp_engine.can_scalp(
            current_time, self.risk_manager.current_capital,
            self._is_expiry_day()
        )
        if not can_scalp:
            return

        is_favorable, env_score = self.scalp_engine.is_scalp_environment(
            df_5min, self.india_vix
        )
        if not is_favorable:
            return

        signals = self.scalp_engine.scan_scalps(
            df_5min, spot_price,
            is_expiry_day=self._is_expiry_day(),
            vix=self.india_vix
        )

        best = self.scalp_engine.get_best_scalp(signals)
        if not best:
            return

        logger.info(f"SCALP SIGNAL: {best.scalp_type.value} | "
                     f"Confidence={best.confidence} | {best.reason}")

        self._execute_scalp_trade(best, spot_price, df_5min)

    def _execute_scalp_trade(self, scalp, spot_price, df_5min):
        """Execute a scalp trade with tight targets."""
        option_type = "CE" if scalp.direction > 0 else "PE"
        strike = round(spot_price / self.instrument.strike_gap) * self.instrument.strike_gap

        # Duplicate signal guard
        entry_key = (option_type, strike, self.active_instrument_key)
        if entry_key in self._recent_entry_keys:
            logger.debug(f"Duplicate scalp entry suppressed: {entry_key}")
            return
        self._recent_entry_keys.add(entry_key)

        atr_val = atr(df_5min).iloc[-1] if not df_5min.empty else 50
        estimated_premium = atr_val * 0.7  # Scalps use slightly ITM for delta

        # Fetch real LTP for accurate sizing (LIVE mode)
        info = None
        if self.mode == RunMode.LIVE:
            info = self.fetcher.find_nearest_expiry_token(
                self.active_instrument_key, strike, option_type
            )
            if info:
                real_ltp = self.fetcher.get_ltp(
                    info["exchange"], info["symbol"], info["token"]
                )
                if real_ltp and real_ltp > 0:
                    estimated_premium = real_ltp

        sl_dist = estimated_premium * (scalp.premium_sl_pct / 100)
        lots = self.risk_manager.calculate_lots(
            None, self.instrument.lot_size,
            estimated_premium, sl_dist
        )

        if lots <= 0:
            logger.warning("Scalp skipped: risk sizing returned 0 lots")
            return

        lots = max(1, int(lots * self.recovery_manager.get_size_multiplier()))

        if self.mode == RunMode.LIVE:
            if info:
                result = self.order_manager.place_option_buy(
                    info["symbol"], info["token"], info["exchange"],
                    lots, self.instrument.lot_size, estimated_premium
                )
                if result.success:
                    # Recompute SL and target from actual fill price
                    estimated_premium = result.fill_price
                    sl_dist = estimated_premium * (scalp.premium_sl_pct / 100)
                else:
                    return
            else:
                logger.error(f"Could not find token for scalp: {option_type} {strike}")
                return
        else:
            logger.info(f"PAPER SCALP: {scalp.scalp_type.value} | "
                         f"Entry=₹{estimated_premium:.1f} | Lots={lots}")

        target_premium = estimated_premium * (scalp.premium_target_pct / 100)

        pos_id = self.risk_manager.open_position(
            strategy_name=f"scalp_{scalp.scalp_type.value}",
            entry_premium=estimated_premium,
            strike=strike,
            option_type=option_type,
            lots=lots,
            lot_size=self.instrument.lot_size,
            sl_premium=sl_dist,
            target_premium=target_premium,
            trail_type="fixed",
            time_stop_candles=scalp.max_hold_candles,
        )
        self.session_router.record_trade(datetime.now())
        # scalps_today is tracked on close via record_scalp_result to avoid
        # double counting open+close events.
        logger.info(f"SCALP OPENED: {pos_id} | Target={target_premium:.1f} SL={sl_dist:.1f}")
        self.risk_manager.save_positions()

    # ──────────────────────────────────────────────
    # V3: Re-Entry After Stop-Outs
    # ──────────────────────────────────────────────

    def _check_reentries(self, df_5min: pd.DataFrame, spot_price: float,
                          regime_state):
        """Check if any stopped-out trade qualifies for re-entry."""
        can_trade, _ = self.risk_manager.can_trade()
        if not can_trade:
            return

        ema21_val = None
        vwap_val = None

        if len(df_5min) > 21:
            ema21_series = ema(df_5min["close"], 21)
            ema21_val = ema21_series.iloc[-1]

        if "volume" in df_5min.columns and df_5min["volume"].sum() > 0:
            vwap_series = vwap(df_5min)
            if vwap_series is not None and not pd.isna(vwap_series.iloc[-1]):
                vwap_val = vwap_series.iloc[-1]

        signal = self.reentry_manager.check_reentry(
            instrument=self.active_instrument_key,
            current_time=datetime.now(),
            current_price=spot_price,
            current_regime_score=regime_state.total_score,
            vwap_price=vwap_val,
            ema21_price=ema21_val,
        )

        if signal is None:
            return

        logger.info(f"RE-ENTRY SIGNAL: {signal.reason} | "
                     f"Size={signal.size_multiplier:.0%}")

        # Execute at reduced size
        atr_val = atr(df_5min).iloc[-1] if not df_5min.empty else 50
        estimated_premium = atr_val * 0.8
        option_type = "CE" if signal.original_trade.direction > 0 else "PE"
        strike = round(spot_price / self.instrument.strike_gap) * self.instrument.strike_gap

        sl_points = abs(signal.re_entry_price - signal.new_sl_price)
        sl_dist = self._convert_spot_points_to_premium(
            spot_points=sl_points,
            option_type=option_type,
            strike=strike,
            spot_price=spot_price,
            min_premium_move=0.5,
        )

        lots = self.risk_manager.calculate_lots(
            None, self.instrument.lot_size,
            estimated_premium, sl_dist
        )

        if lots <= 0:
            logger.warning("Re-entry skipped: risk sizing returned 0 lots")
            return

        lots = max(1, int(lots * signal.size_multiplier))

        if self.mode == RunMode.LIVE:
            info = self.fetcher.find_nearest_expiry_token(
                self.active_instrument_key, strike, option_type
            )
            if info:
                # Fetch real LTP for accurate sizing
                real_ltp = self.fetcher.get_ltp(
                    info["exchange"], info["symbol"], info["token"]
                )
                if real_ltp and real_ltp > 0:
                    estimated_premium = real_ltp

                pre_order_premium = estimated_premium

                result = self.order_manager.place_option_buy(
                    info["symbol"], info["token"], info["exchange"],
                    lots, self.instrument.lot_size, estimated_premium
                )
                if result.success:
                    # Recompute SL from actual fill using same premium-ratio basis
                    # used in pre-trade sizing.
                    if pre_order_premium > 0 and sl_dist > 0:
                        sl_ratio = sl_dist / pre_order_premium
                        estimated_premium = result.fill_price
                        sl_dist = estimated_premium * sl_ratio
                    else:
                        estimated_premium = result.fill_price
                else:
                    logger.error(f"Re-entry order failed: {result.message}")
                    return
            else:
                logger.error(f"Could not find token for re-entry: {option_type} {strike}")
                return
        else:
            logger.info(f"PAPER RE-ENTRY: {option_type} {strike} | "
                         f"Lots={lots} | Premium=₹{estimated_premium:.1f}")

        pos_id = self.risk_manager.open_position(
            strategy_name=f"reentry_{signal.original_trade.strategy_name}",
            entry_premium=estimated_premium,
            strike=strike,
            option_type=option_type,
            lots=lots,
            lot_size=self.instrument.lot_size,
            sl_premium=sl_dist,
            target_premium=None,
            trail_type="breakeven",
            time_stop_candles=8,
        )
        self.session_router.record_trade(datetime.now())
        logger.info(f"RE-ENTRY OPENED: {pos_id}")
        self.risk_manager.save_positions()

    # ──────────────────────────────────────────────
    # V3: Helpers
    # ──────────────────────────────────────────────

    def _estimate_abs_delta(self, option_type: str, spot_price: float, strike: float) -> float:
        """Estimate absolute option delta from moneyness + days-to-expiry.

        Lightweight heuristic to avoid unit mixing when converting spot-point
        stops into option-premium stops. Clamped to a safe practical band.
        """
        gap = max(float(self.instrument.strike_gap), 1.0)
        dte = self._estimate_days_to_expiry()

        # Near-expiry ATM options react faster; farther expiry reacts slower.
        if dte <= 1:
            base = 0.58
        elif dte <= 3:
            base = 0.52
        elif dte <= 7:
            base = 0.48
        else:
            base = 0.44

        if option_type == "CE":
            otm_steps = max((strike - spot_price) / gap, 0.0)
            itm_steps = max((spot_price - strike) / gap, 0.0)
        else:  # PE
            otm_steps = max((spot_price - strike) / gap, 0.0)
            itm_steps = max((strike - spot_price) / gap, 0.0)

        delta = base - 0.12 * otm_steps + 0.08 * itm_steps
        return min(max(delta, 0.15), 0.85)

    def _convert_spot_points_to_premium(self, spot_points: float, option_type: str,
                                        strike: float, spot_price: float,
                                        min_premium_move: float = 0.5) -> float:
        """Convert spot/index points to premium points using estimated delta."""
        if spot_points <= 0:
            return min_premium_move

        abs_delta = self._estimate_abs_delta(option_type, spot_price, strike)
        premium_points = spot_points * abs_delta
        return max(round(premium_points, 2), min_premium_move)

    def _is_expiry_day(self) -> bool:
        """Check if today is any expiry (weekly or monthly) for current instrument."""
        config = INSTRUMENTS.get(self.active_instrument_key)
        if config is None:
            return False
        return self.scanner.is_expiry_today(self.active_instrument_key, config)

    def _estimate_days_to_expiry(self) -> int:
        """Estimate days to the nearest expiry for current instrument."""
        today = datetime.now().date()
        config = INSTRUMENTS.get(self.active_instrument_key)
        if config is None:
            return 3  # Safe default

        day_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6}
        expiry_weekday = day_map.get(config.expiry_day.lower(), 3)
        current_weekday = today.weekday()

        days_ahead = expiry_weekday - current_weekday
        if days_ahead <= 0:
            days_ahead += 7

        return max(days_ahead, 0)

    # ──────────────────────────────────────────────
    # Close All & EOD
    # ──────────────────────────────────────────────

    def _close_all_positions(self, reason: str):
        """Force-close all open positions. Retries if any fail."""
        max_retries = 3
        for attempt in range(max_retries):
            remaining = list(self.risk_manager.positions.keys())
            if not remaining:
                return

            logger.info(f"Closing {len(remaining)} positions (attempt {attempt + 1}/{max_retries})")
            for pos_id in remaining:
                pos = self.risk_manager.positions.get(pos_id)
                if pos:
                    exit_premium = pos.current_premium
                    self._exit_position(pos_id, exit_premium, reason)

            if not self.risk_manager.positions:
                return

            # Some failed — wait and retry
            time_module.sleep(2)

        if self.risk_manager.positions:
            logger.error(
                f"CRITICAL: {len(self.risk_manager.positions)} positions still open "
                f"after {max_retries} close attempts! Manual intervention needed."
            )

    def _end_of_day(self):
        """End of day logging and summary."""
        summary = self.risk_manager.get_daily_summary()
        logger.info("=" * 60)
        logger.info("END OF DAY SUMMARY")
        logger.info(f"  Date:         {summary['date']}")
        logger.info(f"  P&L:          ₹{summary['realized_pnl']:,.0f}")
        logger.info(f"  Trades:       {summary['trades']}")
        logger.info(f"  Win Rate:     {summary['win_rate']:.0f}%")
        logger.info(f"  Capital:      ₹{summary['capital']:,.0f}")
        logger.info(f"  Slippage:     ₹{self.order_manager.total_slippage:.0f}")
        logger.info("=" * 60)

        self._log_daily_pnl(summary)
        self.risk_manager.reset_daily()

        # V3: Reset all daily state
        self.scalp_engine.reset_daily()
        self.session_router.reset_daily()
        self.reentry_manager.reset_daily()
        self.recovery_manager.reset_daily()
        self.oi_analyzer.reset_daily()
        self.no_trade_relaxed = False

        # Cleanup position persistence file only if all positions are closed
        if not self.risk_manager.positions:
            self.risk_manager._remove_positions_file()
        else:
            # Positions still open (close failed) — keep file for crash recovery
            self.risk_manager.save_positions()
            logger.warning("Positions file kept — some positions still open")

    # ──────────────────────────────────────────────
    # Trade Logging
    # ──────────────────────────────────────────────

    def _log_trade(self, pos, exit_premium, pnl, reason):
        """Log trade to CSV."""
        os.makedirs(LOG_DIR, exist_ok=True)

        file_exists = os.path.exists(TRADE_LOG_FILE)
        with open(TRADE_LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "date", "time", "strategy", "option_type", "strike",
                    "entry_premium", "exit_premium", "lots", "pnl", "reason",
                    "candles_held", "capital_after"
                ])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S"),
                pos.strategy_name, pos.option_type, pos.strike,
                pos.entry_premium, exit_premium, pos.lots,
                round(pnl, 0), reason, pos.candles_held,
                round(self.risk_manager.current_capital, 0)
            ])

    def _log_daily_pnl(self, summary):
        os.makedirs(LOG_DIR, exist_ok=True)

        file_exists = os.path.exists(DAILY_PNL_FILE)
        with open(DAILY_PNL_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["date", "pnl", "trades", "wins", "losses",
                                 "win_rate", "capital"])
            writer.writerow([
                summary["date"], summary["realized_pnl"],
                summary["trades"], summary["wins"], summary["losses"],
                round(summary["win_rate"], 1), summary["capital"]
            ])


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

def main():
    """Run the trading engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Indian Options Trading Bot")
    parser.add_argument("--mode", choices=["live", "paper"], default="paper",
                        help="Trading mode: live (real orders) or paper (simulated)")
    parser.add_argument("--instrument", choices=list(INSTRUMENTS.keys()),
                        default=DEFAULT_INSTRUMENT,
                        help="Instrument to trade")
    args = parser.parse_args()

    setup_logging()

    mode = RunMode.LIVE if args.mode == "live" else RunMode.PAPER
    engine = TradingEngine(mode=mode, instrument_key=args.instrument)
    engine.run()


if __name__ == "__main__":
    main()
