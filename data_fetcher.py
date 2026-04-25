"""
Data Fetcher — SmartAPI integration for market data.

Fetches:
1. Historical candle data (5-min, 15-min) for indicators
2. Current LTP for options
3. India VIX
4. Option chain for strike selection
5. Previous day OHLC

Handles API rate limits and connection errors gracefully.
"""
import logging
import time as time_module
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# SmartAPI imports (install: pip install smartapi-python)
try:
    from SmartApi import SmartConnect
    from SmartApi.smartWebSocketV2 import SmartWebSocketV2
    SMARTAPI_AVAILABLE = True
except ImportError:
    SMARTAPI_AVAILABLE = False
    logger.warning("SmartApi not installed. Run: pip install smartapi-python")


class DataFetcher:
    """
    Fetches all required market data from Angel SmartAPI.
    Includes retry logic and rate limit handling.
    """

    def __init__(self, api_key: str, client_id: str, password: str,
                 totp_secret: str = ""):
        self.api_key = api_key
        self.client_id = client_id
        self.password = password
        self.totp_secret = totp_secret
        self.smart_api: Optional[SmartConnect] = None
        self.auth_token = None
        self.feed_token = None
        self._symbol_cache = {}

    # ──────────────────────────────────────────────
    # Authentication
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Authenticate with Angel SmartAPI."""
        if not SMARTAPI_AVAILABLE:
            logger.error("SmartApi package not installed")
            return False

        try:
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP if secret provided
            totp_value = ""
            if self.totp_secret:
                import pyotp
                totp_value = pyotp.TOTP(self.totp_secret).now()

            data = self.smart_api.generateSession(
                self.client_id,
                self.password,
                totp_value
            )

            if data.get("status"):
                self.auth_token = data["data"]["jwtToken"]
                self.feed_token = self.smart_api.getfeedToken()
                self._last_successful_call = time_module.time()
                self._consecutive_failures = 0
                logger.info("SmartAPI connected successfully")
                return True
            else:
                logger.error(f"SmartAPI login failed: {data.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            logger.error(f"SmartAPI connection error: {e}")
            return False

    def reconnect(self, max_attempts: int = 3, backoff_base: float = 5.0) -> bool:
        """
        Re-authenticate after session drop.
        Uses exponential backoff: 5s, 10s, 20s between attempts.
        Returns True if reconnected successfully.
        """
        for attempt in range(1, max_attempts + 1):
            wait = backoff_base * (2 ** (attempt - 1))
            logger.warning(f"Reconnect attempt {attempt}/{max_attempts} "
                           f"(waiting {wait:.0f}s)...")
            time_module.sleep(wait)

            if self.connect():
                # Re-load symbol master so token cache is valid
                if self._symbol_cache:
                    logger.info("Reloading symbol master after reconnect...")
                    self.load_symbol_master()
                logger.info(f"Reconnected on attempt {attempt}")
                return True

        logger.error(f"Failed to reconnect after {max_attempts} attempts")
        return False

    def _record_api_success(self):
        """Track successful API call for health monitoring."""
        self._last_successful_call = time_module.time()
        self._consecutive_failures = 0

    def _record_api_failure(self):
        """Track failed API call. Returns True if reconnect is needed."""
        self._consecutive_failures = getattr(self, '_consecutive_failures', 0) + 1
        return self._consecutive_failures >= 3

    def is_session_healthy(self) -> bool:
        """Check if the API session appears healthy.
        Returns False if >3 consecutive failures or >5 min since last success."""
        failures = getattr(self, '_consecutive_failures', 0)
        last_ok = getattr(self, '_last_successful_call', time_module.time())
        stale = (time_module.time() - last_ok) > 300  # 5 min
        return failures < 3 and not stale

    def _ensure_connected(self):
        if self.smart_api is None:
            raise ConnectionError("Not connected to SmartAPI. Call connect() first.")

    # ──────────────────────────────────────────────
    # Historical Candle Data
    # ──────────────────────────────────────────────

    def get_candle_data(self, symbol_token: str, exchange: str,
                        interval: str = "FIVE_MINUTE",
                        days_back: int = 5) -> pd.DataFrame:
        """
        Fetch historical candle data.

        Args:
            symbol_token: Angel symbol token
            exchange: "NSE" / "BSE" / "NFO"
            interval: "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", etc.
            days_back: Number of days of history

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self._ensure_connected()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)

        params = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        for attempt in range(3):
            try:
                data = self.smart_api.getCandleData(params)

                if data.get("status") and data.get("data"):
                    df = pd.DataFrame(
                        data["data"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    df = df.astype({
                        "open": float, "high": float, "low": float,
                        "close": float, "volume": float
                    })
                    self._record_api_success()
                    return df
                else:
                    logger.warning(f"No candle data: {data.get('message', '')}")
                    return pd.DataFrame()

            except Exception as e:
                logger.warning(f"Candle data fetch attempt {attempt + 1} failed: {e}")
                if self._record_api_failure():
                    logger.warning("Multiple consecutive API failures — session may be stale")
                time_module.sleep(1)

        return pd.DataFrame()

    def get_5min_data(self, symbol_token: str, exchange: str,
                      days_back: int = 3) -> pd.DataFrame:
        """Convenience: Fetch 5-minute candles."""
        return self.get_candle_data(symbol_token, exchange, "FIVE_MINUTE", days_back)

    def get_15min_data(self, symbol_token: str, exchange: str,
                       days_back: int = 5) -> pd.DataFrame:
        """Convenience: Fetch 15-minute candles."""
        return self.get_candle_data(symbol_token, exchange, "FIFTEEN_MINUTE", days_back)

    # ──────────────────────────────────────────────
    # LTP & Market Data
    # ──────────────────────────────────────────────

    def get_ltp(self, exchange: str, trading_symbol: str,
                symbol_token: str) -> Optional[float]:
        """Get Last Traded Price for a symbol."""
        self._ensure_connected()

        try:
            data = self.smart_api.ltpData(exchange, trading_symbol, symbol_token)
            if data.get("status") and data.get("data"):
                self._record_api_success()
                return float(data["data"]["ltp"])
        except Exception as e:
            if self._record_api_failure():
                logger.warning("Multiple consecutive LTP failures — session may be stale")
            logger.warning(f"LTP fetch failed for {trading_symbol}: {e}")

        return None

    def get_option_ltp(self, index_symbol: str, strike: float,
                       option_type: str, expiry_str: str,
                       option_exchange: str = None) -> Optional[float]:
        """
        Get LTP for a specific option contract.

        Args:
            index_symbol: "NIFTY", "BANKNIFTY", or "SENSEX"
            strike: Strike price (e.g., 24000)
            option_type: "CE" or "PE"
            expiry_str: e.g., "24APR2026" format
            option_exchange: "NFO" or "BFO" (auto-detected if None)
        """
        # Auto-detect exchange: Sensex uses BFO, others use NFO
        if option_exchange is None:
            from config import INSTRUMENTS
            inst = INSTRUMENTS.get(index_symbol)
            option_exchange = inst.option_exchange if inst else "NFO"

        # Build trading symbol: e.g., "NIFTY24APR2026C24000"
        trading_symbol = f"{index_symbol}{expiry_str}{option_type[0]}{int(strike)}"

        token = self._get_symbol_token(trading_symbol, option_exchange)
        if token is None:
            logger.warning(f"Token not found for {option_exchange}:{trading_symbol}")
            return None

        return self.get_ltp(option_exchange, trading_symbol, token)

    def get_option_quote(self, exchange: str, trading_symbol: str,
                         symbol_token: str) -> dict:
        """Get detailed quote including bid/ask for slippage estimation."""
        self._ensure_connected()

        try:
            data = self.smart_api.getMarketData(
                mode="FULL",
                exchangeTokens={exchange: [symbol_token]}
            )
            if data.get("status") and data.get("data"):
                fetched = data["data"]["fetched"]
                if fetched:
                    item = fetched[0]
                    return {
                        "ltp": float(item.get("ltp", 0)),
                        "best_bid": float(item.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
                        "best_ask": float(item.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
                        "bid_qty": int(item.get("depth", {}).get("buy", [{}])[0].get("quantity", 0)),
                        "ask_qty": int(item.get("depth", {}).get("sell", [{}])[0].get("quantity", 0)),
                        "oi": int(item.get("opnInterest", 0)),
                        "volume": int(item.get("tradeVolume", 0)),
                    }
        except Exception as e:
            logger.warning(f"Quote fetch failed: {e}")

        return {}

    # ──────────────────────────────────────────────
    # India VIX
    # ──────────────────────────────────────────────

    def get_india_vix(self) -> float:
        """Fetch current India VIX. Returns 15.0 as default if unavailable."""
        try:
            # India VIX token on NSE
            vix_ltp = self.get_ltp("NSE", "India VIX", "26017")
            if vix_ltp is not None:
                return vix_ltp
        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        return 15.0  # Safe default

    # ──────────────────────────────────────────────
    # Previous Day Data
    # ──────────────────────────────────────────────

    def get_previous_day_ohlc(self, symbol_token: str,
                               exchange: str) -> dict:
        """Get previous trading day's OHLC for key levels."""
        # Fetch 2 days of daily data
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)

        try:
            params = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": "ONE_DAY",
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }

            data = self.smart_api.getCandleData(params)

            if data.get("status") and data.get("data") and len(data["data"]) >= 2:
                prev_day = data["data"][-2]  # Second-to-last = previous day
                return {
                    "prev_high": float(prev_day[2]),
                    "prev_low": float(prev_day[3]),
                    "prev_close": float(prev_day[4]),
                }
        except Exception as e:
            logger.warning(f"Previous day OHLC fetch failed: {e}")

        return {"prev_high": np.nan, "prev_low": np.nan, "prev_close": np.nan}

    # ──────────────────────────────────────────────
    # Symbol Token Resolution
    # ──────────────────────────────────────────────

    def load_symbol_master(self):
        """
        Load Angel's instrument master file for token lookup.
        Uses a daily disk cache so restarts don't re-download the ~60 MB file.
        Call once at startup.
        """
        import json, os, glob

        cache_dir = "logs"
        os.makedirs(cache_dir, exist_ok=True)
        today_str = datetime.now().strftime("%Y%m%d")
        cache_path = os.path.join(cache_dir, f"symbol_master_{today_str}.json")

        instruments = None

        # ── Try loading from today's disk cache first ──
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    instruments = json.load(f)
                logger.info(f"Symbol master loaded from disk cache ({cache_path})")
            except Exception as e:
                logger.warning(f"Cache file corrupt, will re-download: {e}")
                instruments = None

        # ── Download fresh if no valid cache ──
        if instruments is None:
            try:
                import requests
                url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                instruments = response.json()

                # Save to disk for subsequent restarts today
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(instruments, f)
                logger.info(f"Symbol master downloaded and cached to {cache_path}")
            except Exception as e:
                logger.error(f"Failed to download symbol master: {e}")
                return

        # Index by trading symbol for fast lookup
        for inst in instruments:
            key = f"{inst.get('exch_seg', '')}:{inst.get('symbol', '')}"
            self._symbol_cache[key] = inst.get("token", "")

        logger.info(f"Symbol master indexed: {len(self._symbol_cache)} instruments")

        # ── Cleanup stale cache files (older than 3 days) ──
        for old_file in glob.glob(os.path.join(cache_dir, "symbol_master_*.json")):
            if today_str not in old_file:
                try:
                    file_age_days = (datetime.now() - datetime.strptime(
                        os.path.basename(old_file).replace("symbol_master_", "").replace(".json", ""),
                        "%Y%m%d"
                    )).days
                    if file_age_days > 3:
                        os.remove(old_file)
                        logger.debug(f"Removed stale cache: {old_file}")
                except (ValueError, OSError):
                    pass

    def _get_symbol_token(self, trading_symbol: str, exchange: str) -> Optional[str]:
        """Look up token for a trading symbol."""
        key = f"{exchange}:{trading_symbol}"
        return self._symbol_cache.get(key)

    def find_nearest_expiry_token(self, index_symbol: str, strike: float,
                                   option_type: str) -> Optional[dict]:
        """
        Find the nearest expiry option token.
        Works for both weekly (Nifty/Sensex) and monthly (BankNifty) expiries.
        Returns dict with 'token', 'symbol', 'exchange' or None.
        """
        # Determine option exchange
        from config import INSTRUMENTS
        inst = INSTRUMENTS.get(index_symbol)
        opt_exchange = inst.option_exchange if inst else "NFO"

        today = datetime.now().date()
        prefix = f"{opt_exchange}:{index_symbol}"

        candidates = []
        for key, token in self._symbol_cache.items():
            if key.startswith(prefix) and str(int(strike)) in key:
                if option_type[0] in key:  # C or P
                    candidates.append((key, token))

        if not candidates:
            return None

        # Sort candidates by expiry date embedded in symbol name
        # Try to extract date from symbol for proper ordering
        import re
        dated_candidates = []
        for key, token in candidates:
            sym = key.split(":")[1]
            # Try common patterns: NIFTY25APR2026CE24000, NIFTY2504252400CE
            match = re.search(r'(\d{2}[A-Z]{3}\d{4})', sym)
            if match:
                try:
                    exp_date = datetime.strptime(match.group(1), "%d%b%Y").date()
                    if exp_date >= today:  # Skip expired contracts
                        dated_candidates.append((key, token, exp_date))
                except ValueError:
                    dated_candidates.append((key, token, today + timedelta(days=365)))
            else:
                dated_candidates.append((key, token, today + timedelta(days=365)))

        if not dated_candidates:
            # Fallback: return first candidate even if we can't parse date
            if candidates:
                sym = candidates[0][0].split(":")[1]
                return {"token": candidates[0][1], "symbol": sym, "exchange": opt_exchange}
            return None

        # Return the nearest future expiry
        dated_candidates.sort(key=lambda x: x[2])
        best = dated_candidates[0]
        sym = best[0].split(":")[1]
        return {
            "token": best[1],
            "symbol": sym,
            "exchange": opt_exchange
        }

    # ──────────────────────────────────────────────
    # Spot Price
    # ──────────────────────────────────────────────

    def get_spot_price(self, symbol_token: str, exchange: str) -> Optional[float]:
        """Get current spot/index price."""
        # Some brokers require both token and symbol for ltpData.
        # Resolve a stable symbol from configured instruments when possible.
        from config import INSTRUMENTS

        trading_symbol = None
        for cfg in INSTRUMENTS.values():
            if cfg.token == symbol_token and cfg.exchange == exchange:
                trading_symbol = cfg.symbol
                break

        return self.get_ltp(exchange, trading_symbol or "", symbol_token)

    # ──────────────────────────────────────────────
    # Option Chain (for OI Analysis)
    # ──────────────────────────────────────────────

    def get_option_chain(self, index_symbol: str, spot_price: float,
                          strike_gap: int,
                          strikes_each_side: int = 10) -> dict:
        """
        Fetch full option chain with OI, volume, IV for nearby strikes.
        Sequential with rate-limiting to stay within Angel API limits
        (getMarketData: 10 req/sec).

        Returns: Dict[strike_float -> StrikeOI]
        Used by OIAnalyzer for max pain, PCR, OI walls.
        """
        from oi_analyzer import StrikeOI

        self._ensure_connected()

        atm = round(spot_price / strike_gap) * strike_gap
        strikes = [atm + i * strike_gap
                    for i in range(-strikes_each_side, strikes_each_side + 1)]
        chain = {}

        for i, strike in enumerate(strikes):
            data = StrikeOI(strike=strike)

            ce_info = self.find_nearest_expiry_token(index_symbol, strike, "CE")
            if ce_info:
                ce_quote = self.get_option_quote(
                    ce_info["exchange"], ce_info["symbol"], ce_info["token"]
                )
                if ce_quote:
                    data.ce_oi = ce_quote.get("oi", 0)
                    data.ce_volume = ce_quote.get("volume", 0)
                    data.ce_ltp = ce_quote.get("ltp", 0)

            pe_info = self.find_nearest_expiry_token(index_symbol, strike, "PE")
            if pe_info:
                pe_quote = self.get_option_quote(
                    pe_info["exchange"], pe_info["symbol"], pe_info["token"]
                )
                if pe_quote:
                    data.pe_oi = pe_quote.get("oi", 0)
                    data.pe_volume = pe_quote.get("volume", 0)
                    data.pe_ltp = pe_quote.get("ltp", 0)

            chain[strike] = data

            # Rate limit: 2 getMarketData calls per strike (CE + PE)
            # Angel limit = 10 req/sec → space at ~4 strikes/sec = 0.25s per strike
            if i < len(strikes) - 1:
                time_module.sleep(0.25)

        logger.info(f"Option chain loaded: {len(chain)} strikes around {atm}")
        return chain

    def get_market_depth(self, exchange: str, trading_symbol: str,
                          symbol_token: str) -> dict:
        """
        Get 5-level market depth (bid/ask) for order book analysis.
        Returns structured depth data for OrderBookAnalyzer.
        """
        from orderbook_analyzer import OrderBookSnapshot, DepthLevel

        self._ensure_connected()

        try:
            data = self.smart_api.getMarketData(
                mode="FULL",
                exchangeTokens={exchange: [symbol_token]}
            )

            if data.get("status") and data.get("data"):
                fetched = data["data"].get("fetched", [])
                if fetched:
                    item = fetched[0]
                    depth = item.get("depth", {})

                    bids = []
                    for level in depth.get("buy", [])[:5]:
                        bids.append(DepthLevel(
                            price=float(level.get("price", 0)),
                            quantity=int(level.get("quantity", 0)),
                            orders=int(level.get("orders", 0))
                        ))

                    asks = []
                    for level in depth.get("sell", [])[:5]:
                        asks.append(DepthLevel(
                            price=float(level.get("price", 0)),
                            quantity=int(level.get("quantity", 0)),
                            orders=int(level.get("orders", 0))
                        ))

                    total_bid = sum(b.quantity for b in bids)
                    total_ask = sum(a.quantity for a in asks)

                    return OrderBookSnapshot(
                        bids=bids,
                        asks=asks,
                        ltp=float(item.get("ltp", 0)),
                        total_bid_qty=total_bid,
                        total_ask_qty=total_ask,
                        timestamp=datetime.now().strftime("%H:%M:%S")
                    )

        except Exception as e:
            logger.warning(f"Market depth fetch failed: {e}")

        return OrderBookSnapshot()

    # ──────────────────────────────────────────────
    # Broker Position Verification
    # ──────────────────────────────────────────────

    def get_open_positions(self) -> dict:
        """
        Fetch current open positions from broker.
        Returns dict keyed by trading_symbol → net quantity.
        Used to detect manually exited positions before placing sell orders.
        """
        self._ensure_connected()
        positions = {}

        try:
            data = self.smart_api.position()
            if data and data.get("status") and data.get("data"):
                for pos in data["data"]:
                    symbol = pos.get("tradingsymbol", "")
                    net_qty = int(pos.get("netqty", 0))
                    if symbol:
                        positions[symbol] = net_qty
        except Exception as e:
            logger.warning(f"Position fetch failed: {e}")

        return positions

    def verify_position_exists(self, trading_symbol: str,
                                expected_qty: int) -> bool:
        """
        Check if a position still exists in the broker with expected quantity.
        Returns True if position exists with net qty >= expected_qty.
        Returns True on error (fail-safe: don't skip exit on API errors).
        """
        try:
            positions = self.get_open_positions()
            net_qty = positions.get(trading_symbol, 0)
            return net_qty >= expected_qty
        except Exception as e:
            logger.warning(f"Position verification failed, assuming exists: {e}")
            return True  # Fail-safe: attempt exit if we can't verify

    # ──────────────────────────────────────────────
    # WebSocket for Real-time Data (optional upgrade)
    # ──────────────────────────────────────────────

    def setup_websocket(self, tokens: list, callback):
        """
        Optional: Setup WebSocket for real-time data.
        Reduces API calls and latency.
        """
        if not SMARTAPI_AVAILABLE or not self.feed_token:
            logger.warning("Cannot setup WebSocket — not connected")
            return None

        try:
            ws = SmartWebSocketV2(
                self.auth_token,
                self.api_key,
                self.client_id,
                self.feed_token
            )

            def on_data(wsapp, message):
                callback(message)

            def on_open(wsapp):
                # Subscribe to tokens
                token_list = [{"exchangeType": 2, "tokens": tokens}]  # 2 = NFO
                ws.subscribe("abc123", 3, token_list)  # 3 = SnapQuote mode

            ws.on_data = on_data
            ws.on_open = on_open

            return ws

        except Exception as e:
            logger.error(f"WebSocket setup failed: {e}")
            return None
