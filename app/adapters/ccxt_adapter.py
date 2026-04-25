"""
Generic CCXT-based adapter. We use plain ``ccxt`` (sync lib wrapped in asyncio.to_thread)
by default so we don't require ``ccxt.pro`` to run. WebSocket streaming can be added by
subclassing and overriding ``watch_orderbook``.

All network calls are guarded by ``asyncio.wait_for`` so a hung exchange can never
starve the async loop. Orderbook fetch is pinned to a shallow depth so we stay
well under per-endpoint rate-limit budgets.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from app.adapters.base import ExchangeAdapter
from app.common.clock import utcnow
from app.common.enums import OrderStatus, OrderType, Side
from app.common.exceptions import AuthError, PermanentError, RateLimitError, TransientError
from app.common.ids import new_order_id
from app.common.logging import get_logger
from app.models.balance import BalanceSnapshot
from app.models.order import OrderIntent, UnifiedOrderState
from app.models.orderbook import OrderBookLevel, OrderBookSnapshot

log = get_logger("adapter.ccxt")

# Orderbook depth used by the REST fallback. With probe sizes now matching
# ``max_notional_per_trade`` (up to 400 USDT default), 5 levels is often not
# enough to fill the probe on thin small-cap books — VWAP gets truncated and
# reports an artificially low max_tradable_base, making the risk engine cap
# trades far below what the market actually offers. 10 is the sweet spot:
# Binance/OKX/Bybit/Gate/Bitget/HTX all treat limit=10 as weight=1 (same as
# limit=5), so no rate-limit impact.
_ORDERBOOK_DEPTH = 10
# Some exchanges reject small limits on fetchOrderBook. Known per-exchange
# minimum accepted limits (ccxt raises ``ExchangeError`` if below):
#   KuCoin     : must be 20 or 100
#   Coinbase   : supports 1/50/... ; 5 works but 20 is safer for adv-trade
# We keep the global default at 5 (cheap) and override only where the
# exchange refuses it.
_ORDERBOOK_DEPTH_OVERRIDE = {
    "kucoin": 20,
}

# WebSocket-side per-exchange minimum accepted depth. Some exchanges
# enforce a different (and stricter) allowed-limit set on the ws feed
# than on the REST endpoint. Sourced from ccxt error messages observed
# on the live VPS:
#   bybit  : spot ws accepts only 1 / 50 / 200 / 1000
#   kraken : ws accepts only 10 / 25 / 100 / 500 / 1000
#   kucoin : ws accepts 20 / 100 (same as REST)
# Coinbase, OKX, Binance, Gate, Bitget, HTX accept depth=5 on ws so the
# default applies.
_ORDERBOOK_DEPTH_WS_OVERRIDE = {
    "bybit": 50,
    "kraken": 10,
    "kucoin": 20,
}

# Default hard ceilings on ccxt round-trips. These guard against hung sockets.
_DEFAULT_ORDERBOOK_TIMEOUT_S = 5.0
_DEFAULT_CREATE_ORDER_TIMEOUT_S = 8.0
_DEFAULT_CANCEL_TIMEOUT_S = 5.0
_DEFAULT_FETCH_ORDER_TIMEOUT_S = 5.0
_DEFAULT_FETCH_BALANCE_TIMEOUT_S = 8.0


def _d(x: Any) -> Decimal:
    if x is None:
        return Decimal(0)
    return Decimal(str(x))


_STATUS_MAP = {
    "open": OrderStatus.SUBMITTED,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}

_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


async def _bounded(coro, timeout_s: float, kind: str):
    """Wrap ``asyncio.to_thread(...)`` in ``asyncio.wait_for`` so a hung ccxt
    HTTP call cannot pin the event loop. Always raises TransientError on
    timeout — callers surface it as a recoverable adapter failure."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise TransientError(f"{kind} timed out after {timeout_s}s") from e


class CcxtExchangeAdapter(ExchangeAdapter):
    """
    Wraps a ``ccxt`` sync exchange instance with asyncio.to_thread.
    """

    def __init__(
        self,
        name: str,
        ccxt_client: Any,
        default_fee_bps: Decimal = Decimal("10"),
        pro_client: Any | None = None,
    ):
        self.name = name
        self._client = ccxt_client
        self._default_fee_bps = default_fee_bps
        self._connected = False
        # Optional ccxt.pro async client used by ``watch_orderbook_ws``.
        # When None, ws is disabled and the manager will REST-poll instead.
        self._pro_client = pro_client

    @property
    def is_configured(self) -> bool:
        apikey = getattr(self._client, "apiKey", "") or ""
        return bool(apikey)

    async def connect(self) -> None:
        try:
            await _bounded(
                asyncio.to_thread(self._client.load_markets),
                timeout_s=10.0,
                kind="load_markets",
            )
            self._connected = True
            log.info("exchange_connected", exchange=self.name)
        except Exception as e:
            log.warning("exchange_connect_failed", exchange=self.name, error=str(e))
            # We don't raise: scanner will mark this exchange unhealthy instead.

    async def close(self) -> None:
        try:
            close = getattr(self._client, "close", None)
            if callable(close):
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:  # noqa: BLE001
            pass
        # ccxt.pro client owns a live WS + http session; close it too so
        # we don't leak file descriptors on shutdown.
        if self._pro_client is not None:
            try:
                pclose = getattr(self._pro_client, "close", None)
                if callable(pclose):
                    res = pclose()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:  # noqa: BLE001
                pass
        self._connected = False

    def supports_websocket(self) -> bool:
        return self._pro_client is not None and hasattr(self._pro_client, "watch_order_book")

    async def watch_orderbook_ws(self, symbol: str) -> OrderBookSnapshot:
        """Block on a ccxt.pro WS update for ``symbol``. Raises TransientError
        on disconnect / timeout; the manager falls back to REST in that case.

        ccxt.pro's ``watch_order_book`` resolves on every push update from
        the exchange, so the caller's ``while True: await self.watch_orderbook_ws``
        is equivalent to a long-lived subscription with internal reconnect.
        """
        if self._pro_client is None:
            raise NotImplementedError("pro_client not configured")
        depth = _ORDERBOOK_DEPTH_WS_OVERRIDE.get(
            self.name, _ORDERBOOK_DEPTH_OVERRIDE.get(self.name, _ORDERBOOK_DEPTH)
        )
        try:
            # 30s timeout: most exchanges push within ~1s, but inactive
            # symbols (small caps with no recent trades) can stall longer.
            # 15s was too aggressive on the live VPS.
            ob = await asyncio.wait_for(
                self._pro_client.watch_order_book(symbol, depth),
                timeout=30.0,
            )
        except asyncio.TimeoutError as e:
            raise TransientError(f"watch_order_book[{self.name}/{symbol}] no update in 30s") from e
        except Exception as e:  # noqa: BLE001
            cls_name = type(e).__name__
            if "RateLimit" in cls_name:
                raise RateLimitError(str(e)) from e
            if "Auth" in cls_name:
                raise AuthError(str(e)) from e
            raise TransientError(f"ws update failed: {e}") from e

        now = utcnow()
        ts_ex = None
        if ob.get("timestamp"):
            ts_ex = datetime.fromtimestamp(ob["timestamp"] / 1000, tz=timezone.utc)
        latency_ms = None
        if ts_ex is not None:
            latency_ms = max(0, int((now - ts_ex).total_seconds() * 1000))
        bids = [OrderBookLevel(_d(lv[0]), _d(lv[1])) for lv in (ob.get("bids") or [])]
        asks = [OrderBookLevel(_d(lv[0]), _d(lv[1])) for lv in (ob.get("asks") or [])]
        return OrderBookSnapshot(
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
            ts_local=now,
            ts_exchange=ts_ex,
            latency_ms=latency_ms,
        )

    def supports_symbol(self, symbol: str) -> bool:
        """After ``load_markets``, consult the ccxt markets table.

        Before connect() runs ``markets`` is empty / None, so default to
        True (optimistic) so the first poll attempt can itself populate /
        discover the market. Post-connect this returns False for symbols
        the exchange genuinely does not list.
        """
        try:
            markets = getattr(self._client, "markets", None)
        except Exception:  # noqa: BLE001
            return True
        if not markets:
            return True
        return symbol in markets

    async def watch_orderbook(self, symbol: str) -> OrderBookSnapshot:
        """One-shot REST fetch. Long-running watchers poll this in a loop."""
        depth = _ORDERBOOK_DEPTH_OVERRIDE.get(self.name, _ORDERBOOK_DEPTH)
        try:
            ob = await _bounded(
                asyncio.to_thread(self._client.fetch_order_book, symbol, depth),
                timeout_s=_DEFAULT_ORDERBOOK_TIMEOUT_S,
                kind="fetch_order_book",
            )
        except TransientError:
            raise
        except Exception as e:  # noqa: BLE001
            cls_name = type(e).__name__
            if "RateLimit" in cls_name:
                raise RateLimitError(str(e)) from e
            if "Auth" in cls_name:
                raise AuthError(str(e)) from e
            raise TransientError(f"orderbook fetch failed: {e}") from e

        now = utcnow()
        ts_ex = None
        if ob.get("timestamp"):
            ts_ex = datetime.fromtimestamp(ob["timestamp"] / 1000, tz=timezone.utc)
        latency_ms = None
        if ts_ex is not None:
            latency_ms = max(0, int((now - ts_ex).total_seconds() * 1000))

        # OKX returns [price, size, liquidated_orders, num_orders] while Binance
        # returns [price, size]; index into the first two to stay compatible.
        bids = [OrderBookLevel(_d(lv[0]), _d(lv[1])) for lv in (ob.get("bids") or [])]
        asks = [OrderBookLevel(_d(lv[0]), _d(lv[1])) for lv in (ob.get("asks") or [])]
        return OrderBookSnapshot(
            exchange=self.name,
            symbol=symbol,
            bids=bids,
            asks=asks,
            ts_local=now,
            ts_exchange=ts_ex,
            latency_ms=latency_ms,
        )

    async def fetch_balances(self) -> list[BalanceSnapshot]:
        if not self.is_configured:
            return []
        try:
            bals = await _bounded(
                asyncio.to_thread(self._client.fetch_balance),
                timeout_s=_DEFAULT_FETCH_BALANCE_TIMEOUT_S,
                kind="fetch_balance",
            )
        except TransientError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"balance fetch failed: {e}") from e
        out: list[BalanceSnapshot] = []
        now = utcnow()
        total_map = bals.get("total", {}) or {}
        free_map = bals.get("free", {}) or {}
        used_map = bals.get("used", {}) or {}
        for asset in total_map:
            total = _d(total_map.get(asset))
            if total == 0:
                continue
            out.append(
                BalanceSnapshot(
                    exchange=self.name,
                    asset=asset,
                    free=_d(free_map.get(asset)),
                    locked=_d(used_map.get(asset)),
                    total=total,
                    ts_local=now,
                )
            )
        return out

    def _ccxt_order_type(self, order_type: OrderType) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        if order_type == OrderType.MARKET:
            return "market", params
        if order_type == OrderType.IOC_LIMIT:
            params["timeInForce"] = "IOC"
            return "limit", params
        if order_type == OrderType.FOK_LIMIT:
            params["timeInForce"] = "FOK"
            return "limit", params
        return "limit", params

    def _truncate_coid(self, coid: str | None) -> str | None:
        """Binance accepts up to 32 chars; OKX is 36. Trim defensively."""
        if not coid:
            return coid
        limit = 32 if self.name == "binance" else 36
        return coid[:limit]

    async def create_order(self, intent: OrderIntent) -> UnifiedOrderState:
        t, params = self._ccxt_order_type(intent.order_type)
        coid = self._truncate_coid(intent.client_order_id)
        if coid:
            params["clientOrderId"] = coid
        try:
            raw = await _bounded(
                asyncio.to_thread(
                    self._client.create_order,
                    intent.symbol,
                    t,
                    intent.side.value,
                    float(intent.amount),
                    float(intent.price) if intent.price is not None else None,
                    params,
                ),
                timeout_s=_DEFAULT_CREATE_ORDER_TIMEOUT_S,
                kind="create_order",
            )
        except TransientError:
            raise
        except Exception as e:  # noqa: BLE001
            cls_name = type(e).__name__
            if "RateLimit" in cls_name:
                raise RateLimitError(str(e)) from e
            if "Auth" in cls_name or "Permission" in cls_name:
                raise AuthError(str(e)) from e
            if "Insufficient" in cls_name or "InvalidOrder" in cls_name:
                raise PermanentError(str(e)) from e
            raise TransientError(str(e)) from e

        state = self._normalize_order(raw, intent)

        # Some exchanges return ``open`` even for IOC/FOK because the fill
        # confirmation is async on their side. Poll the order until it reaches
        # a terminal state so downstream code sees accurate fill numbers.
        if state.status not in _TERMINAL and state.exchange_order_id:
            state = await self._poll_until_terminal(state)
        return state

    async def cancel_order(self, exchange_order_id: str, symbol: str) -> UnifiedOrderState:
        try:
            raw = await _bounded(
                asyncio.to_thread(self._client.cancel_order, exchange_order_id, symbol),
                timeout_s=_DEFAULT_CANCEL_TIMEOUT_S,
                kind="cancel_order",
            )
        except TransientError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"cancel_order failed: {e}") from e
        return self._normalize_order(raw, intent=None, symbol=symbol)

    async def fetch_order(self, exchange_order_id: str, symbol: str) -> UnifiedOrderState:
        try:
            raw = await _bounded(
                asyncio.to_thread(self._client.fetch_order, exchange_order_id, symbol),
                timeout_s=_DEFAULT_FETCH_ORDER_TIMEOUT_S,
                kind="fetch_order",
            )
        except TransientError:
            raise
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"fetch_order failed: {e}") from e
        return self._normalize_order(raw, intent=None, symbol=symbol)

    def fee_rate(self, symbol: str, side: str) -> Decimal:
        """
        Prefer ccxt's ``markets[symbol]['taker']`` if present.
        """
        try:
            m = self._client.markets.get(symbol) if getattr(self._client, "markets", None) else None
            if m and "taker" in m and m["taker"] is not None:
                return _d(m["taker"])
        except Exception:  # noqa: BLE001
            pass
        return self._default_fee_bps / Decimal("10000")

    async def _poll_until_terminal(
        self,
        state: UnifiedOrderState,
        max_wait_s: float = 2.0,
        interval_s: float = 0.25,
    ) -> UnifiedOrderState:
        """Re-fetch the order until it settles or we exhaust the time budget.
        On persistent UNKNOWN we return the last observed state — the caller
        treats that as non-terminal and will hand it to the repair engine."""
        deadline = asyncio.get_event_loop().time() + max_wait_s
        current = state
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(interval_s)
            try:
                if not current.exchange_order_id:
                    return current
                latest = await self.fetch_order(current.exchange_order_id, current.symbol)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "poll_until_terminal_error",
                    exchange=self.name,
                    order_id=current.exchange_order_id,
                    error=str(e),
                )
                continue
            # Preserve identity fields the exchange may not echo back.
            latest.internal_order_id = current.internal_order_id
            latest.hedge_group_id = current.hedge_group_id
            latest.is_repair = current.is_repair
            current = latest
            if current.status in _TERMINAL:
                return current
        return current

    # --- helpers ---
    def _normalize_order(
        self,
        raw: dict[str, Any],
        intent: OrderIntent | None,
        symbol: str | None = None,
    ) -> UnifiedOrderState:
        status_s = (raw.get("status") or "").lower()
        status = _STATUS_MAP.get(status_s, OrderStatus.UNKNOWN)
        filled = _d(raw.get("filled"))
        amount = _d(raw.get("amount") or (intent.amount if intent else 0))
        remaining = _d(raw.get("remaining") or max(amount - filled, Decimal(0)))
        if status == OrderStatus.SUBMITTED and filled > 0 and remaining > 0:
            status = OrderStatus.PARTIALLY_FILLED

        fee = raw.get("fee") or {}
        fee_amt = _d(fee.get("cost"))
        fee_asset = fee.get("currency")

        now = utcnow()
        return UnifiedOrderState(
            internal_order_id=new_order_id(),
            hedge_group_id=intent.hedge_group_id if intent else "",
            exchange=self.name,
            exchange_order_id=str(raw.get("id")) if raw.get("id") is not None else None,
            client_order_id=raw.get("clientOrderId"),
            symbol=raw.get("symbol") or symbol or (intent.symbol if intent else ""),
            side=Side((raw.get("side") or (intent.side.value if intent else "buy")).lower()),
            price=_d(raw.get("price")) if raw.get("price") else (intent.price if intent else None),
            amount=amount,
            filled=filled,
            remaining=remaining,
            avg_fill_price=_d(raw.get("average")) if raw.get("average") else None,
            status=status,
            fee_amount=fee_amt if fee_amt else None,
            fee_asset=fee_asset,
            is_repair=intent.is_repair if intent else False,
            created_at=now,
            updated_at=now,
        )
