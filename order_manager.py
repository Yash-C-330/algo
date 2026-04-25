"""
Order Manager — Handles order placement with slippage protection.

Key features:
1. Limit orders with buffer (not market orders)
2. Auto-modify to fill if not filled within timeout
3. Bracket order support (entry + SL + target)
4. Order status tracking
5. Slippage accounting
"""
import logging
import time as time_module
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import LIMIT_ORDER_BUFFER, FILL_TIMEOUT_SEC, ORDER_RETRY_COUNT, SLIPPAGE_POINTS
from strategies import TradeSignal, OptionLeg, OrderSide
from data_fetcher import DataFetcher

logger = logging.getLogger(__name__)
trade_logger = logging.getLogger("trades")


@dataclass
class OrderResult:
    """Result of an order attempt."""
    success: bool
    order_id: str
    fill_price: float
    slippage: float       # Actual slippage from expected price
    trading_symbol: str
    token: str
    message: str
    exchange: str = "NFO"  # NFO or BFO


class OrderManager:
    """
    Places and manages orders through Angel SmartAPI with slippage protection.

    Slippage Handling Strategy:
    1. NEVER use market orders directly
    2. Place limit at best_ask + buffer for buys (best_bid - buffer for sells)
    3. If not filled in FILL_TIMEOUT_SEC → modify to aggressive limit
    4. Track actual slippage for performance analysis
    """

    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self.active_orders = {}  # order_id -> status
        self.total_slippage = 0.0

    def place_option_buy(self, trading_symbol: str, token: str,
                         exchange: str, lots: int, lot_size: int,
                         expected_premium: float) -> OrderResult:
        """
        Place a buy order for an option with slippage protection.

        Steps:
        1. Get current bid/ask
        2. Place limit at ask + buffer
        3. Wait for fill
        4. If not filled, modify to more aggressive price
        """
        api = self.fetcher.smart_api
        if api is None:
            return OrderResult(False, "", 0, 0, trading_symbol, token,
                               "API not connected", exchange)

        # Get current quote for slippage-aware pricing
        quote = self.fetcher.get_option_quote(exchange, trading_symbol, token)
        if quote:
            best_ask = quote.get("best_ask", 0)
            if best_ask <= 0:
                best_ask = quote.get("ltp", expected_premium)
        else:
            best_ask = expected_premium

        # Safety: never place order at near-zero price
        if best_ask <= 0:
            ltp = self.fetcher.get_ltp(exchange, trading_symbol, token)
            if ltp and ltp > 0:
                best_ask = ltp
            else:
                logger.error(f"Cannot determine price for {trading_symbol} — aborting buy")
                return OrderResult(False, "", 0, 0, trading_symbol, token,
                                   "No price available", exchange)

        # Limit price = best ask + small buffer
        limit_price = round(best_ask + LIMIT_ORDER_BUFFER, 1)
        quantity = lots * lot_size

        logger.info(f"ORDER BUY: {trading_symbol} qty={quantity} limit=₹{limit_price} "
                     f"(ask=₹{best_ask})")

        for attempt in range(ORDER_RETRY_COUNT + 1):
            try:
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": trading_symbol,
                    "symboltoken": token,
                    "transactiontype": "BUY",
                    "exchange": exchange,
                    "ordertype": "LIMIT",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": str(limit_price),
                    "quantity": str(quantity),
                }

                result = api.placeOrder(order_params)

                # Angel API returns order_id string on success, or None/dict on failure
                if result and not isinstance(result, dict) and str(result) != "None":
                    order_id = str(result)
                    logger.info(f"Order placed: {order_id}")

                    # Wait for fill
                    fill_price = self._wait_for_fill(order_id, limit_price, attempt)

                    if fill_price is not None:
                        slippage = fill_price - best_ask
                        self.total_slippage += abs(slippage)

                        trade_logger.info(
                            f"BUY  | {trading_symbol} | qty={quantity} | "
                            f"fill=₹{fill_price} | ask=₹{best_ask:.1f} | "
                            f"slip=₹{slippage:.1f} | oid={order_id}"
                        )
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            fill_price=fill_price,
                            slippage=slippage,
                            trading_symbol=trading_symbol,
                            token=token,
                            message=f"Filled at ₹{fill_price} (slippage: ₹{slippage:.1f})",
                            exchange=exchange
                        )

                    # Not filled — modify to more aggressive price
                    if attempt < ORDER_RETRY_COUNT:
                        limit_price = round(limit_price + LIMIT_ORDER_BUFFER, 1)
                        logger.info(f"Order not filled, modifying to ₹{limit_price}")
                        self._modify_order(order_id, limit_price, quantity)

                else:
                    err_msg = result.get("message", str(result)) if isinstance(result, dict) else str(result)
                    logger.warning(f"Order placement returned: {err_msg}")

            except Exception as e:
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                time_module.sleep(0.5)

        return OrderResult(False, "", 0, 0, trading_symbol, token,
                           "All order attempts failed", exchange)

    def place_option_sell(self, trading_symbol: str, token: str,
                          exchange: str, lots: int, lot_size: int,
                          expected_premium: float) -> OrderResult:
        """
        Place a sell order (for exit or spread leg) with slippage protection.
        """
        api = self.fetcher.smart_api
        if api is None:
            return OrderResult(False, "", 0, 0, trading_symbol, token,
                               "API not connected", exchange)

        quote = self.fetcher.get_option_quote(exchange, trading_symbol, token)
        if quote:
            best_bid = quote.get("best_bid", 0)
            if best_bid <= 0:
                best_bid = quote.get("ltp", expected_premium)
        else:
            best_bid = expected_premium

        # Safety: never place sell at near-zero price
        if best_bid <= 0:
            ltp = self.fetcher.get_ltp(exchange, trading_symbol, token)
            if ltp and ltp > 0:
                best_bid = ltp
            else:
                logger.error(f"Cannot determine price for {trading_symbol} — aborting sell")
                return OrderResult(False, "", 0, 0, trading_symbol, token,
                                   "No price available", exchange)

        # Limit price = best bid - small buffer (willing to sell slightly lower)
        limit_price = round(best_bid - LIMIT_ORDER_BUFFER, 1)
        limit_price = max(limit_price, 0.05)  # Don't go below tick size
        quantity = lots * lot_size

        logger.info(f"ORDER SELL: {trading_symbol} qty={quantity} limit=₹{limit_price} "
                     f"(bid=₹{best_bid})")

        for attempt in range(ORDER_RETRY_COUNT + 1):
            try:
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": trading_symbol,
                    "symboltoken": token,
                    "transactiontype": "SELL",
                    "exchange": exchange,
                    "ordertype": "LIMIT",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": str(limit_price),
                    "quantity": str(quantity),
                }

                result = api.placeOrder(order_params)

                if isinstance(result, dict):
                    err_msg = result.get("message", str(result))
                    logger.warning(f"Sell placeOrder returned dict (error): {err_msg}")
                elif result and str(result) != "None":
                    order_id = str(result)

                    fill_price = self._wait_for_fill(order_id, limit_price, attempt)

                    if fill_price is not None:
                        slippage = best_bid - fill_price
                        self.total_slippage += abs(slippage)

                        trade_logger.info(
                            f"SELL | {trading_symbol} | qty={quantity} | "
                            f"fill=₹{fill_price} | bid=₹{best_bid:.1f} | "
                            f"slip=₹{slippage:.1f} | oid={order_id}"
                        )
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            fill_price=fill_price,
                            slippage=slippage,
                            trading_symbol=trading_symbol,
                            token=token,
                            message=f"Filled at ₹{fill_price}",
                            exchange=exchange
                        )

                    if attempt < ORDER_RETRY_COUNT:
                        limit_price = round(limit_price - LIMIT_ORDER_BUFFER, 1)
                        limit_price = max(limit_price, 0.05)
                        self._modify_order(order_id, limit_price, quantity)
                else:
                    logger.warning(f"Sell placeOrder returned empty/None: {result}")

            except Exception as e:
                logger.error(f"Sell order attempt {attempt + 1} failed: {e}")
                time_module.sleep(0.5)

        return OrderResult(False, "", 0, 0, trading_symbol, token,
                           "All sell order attempts failed", exchange)

    def emergency_exit(self, trading_symbol: str, token: str,
                       exchange: str, quantity: int) -> OrderResult:
        """
        Emergency market exit — use only when speed matters more than slippage.
        For daily close-all or stop loss that must execute.
        """
        api = self.fetcher.smart_api
        if api is None:
            return OrderResult(False, "", 0, 0, trading_symbol, token,
                               "API not connected", exchange)

        try:
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": trading_symbol,
                "symboltoken": token,
                "transactiontype": "SELL",
                "exchange": exchange,
                "ordertype": "MARKET",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": str(quantity),
            }

            result = api.placeOrder(order_params)
            if result:
                logger.info(f"EMERGENCY EXIT placed: {result}")

                # Wait briefly for fill
                time_module.sleep(2)
                status = self._get_order_status(str(result))
                fill_price = status.get("averageprice", 0)

                actual_fill = float(fill_price) if fill_price else 0
                trade_logger.info(
                    f"EXIT | {trading_symbol} | qty={quantity} | "
                    f"fill=₹{actual_fill} | EMERGENCY_MARKET | oid={result}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(result),
                    fill_price=actual_fill,
                    slippage=SLIPPAGE_POINTS,  # Assume worst case
                    trading_symbol=trading_symbol,
                    token=token,
                    message="Emergency market exit",
                    exchange=exchange
                )

        except Exception as e:
            logger.error(f"Emergency exit FAILED: {e}")

        return OrderResult(False, "", 0, 0, trading_symbol, token,
                           "Emergency exit failed", exchange)

    # ──────────────────────────────────────────────
    # Order Status & Fill Monitoring
    # ──────────────────────────────────────────────

    def _wait_for_fill(self, order_id: str, expected_price: float,
                       attempt: int) -> Optional[float]:
        """Wait for order to fill, return fill price or None.
        Uses progressive polling to reduce API load (orderBook: 10 req/sec)."""
        timeout = FILL_TIMEOUT_SEC
        start = time_module.time()
        poll_interval = 1.0  # Start at 1s (was 0.5s — too aggressive for API)

        while time_module.time() - start < timeout:
            status = self._get_order_status(order_id)
            if status:
                order_status = status.get("orderstatus", "")
                if order_status == "complete":
                    fill_price = float(status.get("averageprice", expected_price))
                    logger.info(f"Order {order_id} FILLED at ₹{fill_price}")
                    return fill_price
                elif order_status in ("rejected", "cancelled"):
                    logger.warning(f"Order {order_id} {order_status}: "
                                   f"{status.get('text', '')}")
                    return None

            time_module.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.3, 2.0)  # Backoff: 1.0, 1.3, 1.7, 2.0

        logger.info(f"Order {order_id} not filled within {timeout}s")
        return None

    def _get_order_status(self, order_id: str) -> dict:
        """Get status of a specific order."""
        try:
            order_book = self.fetcher.smart_api.orderBook()
            if order_book.get("status") and order_book.get("data"):
                for order in order_book["data"]:
                    if str(order.get("orderid")) == order_id:
                        return order
        except Exception as e:
            logger.warning(f"Order status check failed: {e}")
        return {}

    def _modify_order(self, order_id: str, new_price: float, quantity: int):
        """Modify an open order to a new price."""
        try:
            self.fetcher.smart_api.modifyOrder({
                "variety": "NORMAL",
                "orderid": order_id,
                "ordertype": "LIMIT",
                "price": str(new_price),
                "quantity": str(quantity),
                "duration": "DAY",
            })
            logger.info(f"Order {order_id} modified to ₹{new_price}")
        except Exception as e:
            logger.warning(f"Order modification failed: {e}")

    def cancel_order(self, order_id: str):
        """Cancel an open order."""
        try:
            self.fetcher.smart_api.cancelOrder(order_id, "NORMAL")
            logger.info(f"Order {order_id} cancelled")
        except Exception as e:
            logger.warning(f"Order cancellation failed: {e}")

    # ──────────────────────────────────────────────
    # Execute Full Trade Signal
    # ──────────────────────────────────────────────

    def execute_signal(self, signal: TradeSignal, lot_size: int,
                       lots: int, instrument_key: str = "") -> list:
        """
        Execute all legs of a trade signal.
        Returns list of OrderResult.

        Args:
            instrument_key: e.g. "NIFTY", "BANKNIFTY", "SENSEX"
                            Required for correct token resolution.
        """
        if not instrument_key:
            logger.error("instrument_key is required for execute_signal()")
            return [OrderResult(False, "", 0, 0, "", "",
                                "Missing instrument_key")]

        results = []

        for leg in signal.legs:
            # Resolve the option token
            info = self.fetcher.find_nearest_expiry_token(
                instrument_key, leg.strike, leg.option_type.value
            )

            if info is None:
                logger.error(f"Could not find token for {leg.option_type.value} "
                             f"{leg.strike}")
                results.append(OrderResult(
                    False, "", 0, 0, "", "",
                    f"Token not found for {leg.option_type.value} {leg.strike}"
                ))
                continue

            if leg.side == OrderSide.BUY:
                result = self.place_option_buy(
                    info["symbol"], info["token"], info["exchange"],
                    lots, lot_size, leg.expected_premium
                )
            else:
                result = self.place_option_sell(
                    info["symbol"], info["token"], info["exchange"],
                    lots, lot_size, leg.expected_premium
                )

            results.append(result)

            # If any leg fails, cancel previous legs
            if not result.success:
                logger.error(f"Leg failed: {result.message}. Cancelling previous legs.")
                for prev in results[:-1]:
                    if prev.success:
                        # Reverse the previous leg
                        self.emergency_exit(
                            prev.trading_symbol, prev.token,
                            prev.exchange, lots * lot_size
                        )
                break

        return results
