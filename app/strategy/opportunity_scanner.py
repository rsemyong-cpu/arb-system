"""
OpportunityScanner: for each (symbol, exchange_pair, direction), computes SpreadEstimate,
filters by min-edge and min-profit, emits ArbitrageOpportunity objects.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import AsyncIterator, Callable

from app.accounts.balance_manager import BalanceManager
from app.common.clock import utcnow
from app.common.ids import new_opportunity_id
from app.common.logging import get_logger
from app.config.settings import Settings
from app.marketdata.orderbook_manager import OrderBookManager
from app.models.opportunity import ArbitrageOpportunity
from app.strategy.spread_calculator import SpreadCalculator

log = get_logger("strategy.scanner")


class OpportunityScanner:
    def __init__(
        self,
        settings: Settings,
        book_mgr: OrderBookManager,
        balance_mgr: BalanceManager,
        calc: SpreadCalculator,
        on_opportunity: Callable[[ArbitrageOpportunity], "asyncio.Future | None"] | None = None,
    ):
        self._settings = settings
        self._books = book_mgr
        self._balances = balance_mgr
        self._calc = calc
        self._on_opportunity = on_opportunity
        self._running = False
        self._scan_count = 0
        self._last_scan_at = None
        # Per-process counters surfaced to /opportunities/recent so the UI
        # can show real "本次会话机会数" / "拒绝原因分布" without paging
        # through the DB. Reset on container restart.
        self._session_count = 0
        self._accepted_count = 0
        self._reject_counts: dict[str, int] = {}

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def scan_count(self) -> int:
        return self._scan_count

    def last_scan_at(self):
        return self._last_scan_at

    def session_count(self) -> int:
        return self._session_count

    def accepted_count(self) -> int:
        return self._accepted_count

    def reject_counts(self) -> dict[str, int]:
        return dict(self._reject_counts)

    def record_detected(self) -> None:
        """Called by the bootstrap scanner loop for every opp the scanner
        emits. Kept on the scanner (not the loop) so future callers (tests,
        replay) get the same counters without duplicating logic.
        """
        self._session_count += 1

    def record_decision(self, accepted: bool, reason: str | None = None) -> None:
        """Mirror of ``record_detected`` for after-risk-evaluation. We track
        accepted vs rejected separately so the UI can show the funnel:
        detected → accepted; detected → rejected (with reason breakdown).
        """
        if accepted:
            self._accepted_count += 1
            return
        key = reason or "unknown"
        self._reject_counts[key] = self._reject_counts.get(key, 0) + 1

    async def run(self, exchanges: list[str]) -> None:
        """Poll loop — for every pair in the whitelist, evaluate both directions."""
        self.start()
        interval = self._settings.scan_interval_ms / 1000.0
        while self.is_running():
            try:
                for symbol in self._settings.enabled_symbol_list:
                    await self._scan_symbol(symbol, exchanges)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.error("scanner_loop_error", error=str(e))
            await asyncio.sleep(interval)

    async def scan_once(self, exchanges: list[str]) -> list[ArbitrageOpportunity]:
        self._scan_count += 1
        self._last_scan_at = utcnow()
        out: list[ArbitrageOpportunity] = []
        for symbol in self._settings.enabled_symbol_list:
            out.extend(await self._scan_symbol(symbol, exchanges))
        return out

    async def _scan_symbol(self, symbol: str, exchanges: list[str]) -> list[ArbitrageOpportunity]:
        if len(exchanges) < 2:
            return []
        out: list[ArbitrageOpportunity] = []
        for i in range(len(exchanges)):
            for j in range(len(exchanges)):
                if i == j:
                    continue
                buy_ex, sell_ex = exchanges[i], exchanges[j]
                buy_book = self._books.get(buy_ex, symbol)
                sell_book = self._books.get(sell_ex, symbol)
                if not buy_book or not sell_book:
                    continue
                if self._books.is_stale(buy_ex, symbol) or self._books.is_stale(sell_ex, symbol):
                    continue

                probe = self._probe_size(symbol, buy_book, sell_book)
                est = self._calc.evaluate_direction(symbol, buy_book, sell_book, probe)
                if est is None:
                    continue
                # Pre-risk scanner filters. We record these as if they were
                # risk-gate rejects (same reason strings) so the UI's
                # 拒绝原因分布 panel sees the full funnel — without this,
                # almost every iteration's filtering happens here and the
                # histogram would be permanently empty.
                if est.net_edge_bps < self._settings.min_net_edge_bps:
                    self._session_count += 1
                    self._reject_counts["below_min_edge"] = self._reject_counts.get("below_min_edge", 0) + 1
                    continue
                if est.expected_profit_quote < self._settings.min_profit_quote:
                    self._session_count += 1
                    self._reject_counts["below_min_profit"] = (
                        self._reject_counts.get("below_min_profit", 0) + 1
                    )
                    continue
                # Liquidity filter: reject opportunities where the fillable
                # notional (in USDT) is below the floor. Guards against
                # "ghost" spreads with tiny top-of-book depth that would
                # slip massively on actual execution.
                min_liq = getattr(self._settings, "min_liquidity_usdt", Decimal("0"))
                if min_liq > 0:
                    tradable_notional = est.max_tradable_base * est.buy_leg.effective_price
                    if tradable_notional < min_liq:
                        self._session_count += 1
                        self._reject_counts["below_min_size"] = (
                            self._reject_counts.get("below_min_size", 0) + 1
                        )
                        continue

                opp = ArbitrageOpportunity(
                    opportunity_id=new_opportunity_id(),
                    symbol=symbol,
                    buy_exchange=buy_ex,
                    sell_exchange=sell_ex,
                    buy_price=est.buy_leg.effective_price,
                    sell_price=est.sell_leg.effective_price,
                    gross_spread_bps=est.gross_spread_bps,
                    buy_fee_bps=est.buy_leg.fee_bps,
                    sell_fee_bps=est.sell_leg.fee_bps,
                    slippage_bps=est.slippage_bps_total,
                    buffer_bps=est.buffer_bps,
                    net_edge_bps=est.net_edge_bps,
                    max_tradable_size=est.max_tradable_base,
                    expected_profit_quote=est.expected_profit_quote,
                    detected_at=utcnow(),
                )
                out.append(opp)
                if self._on_opportunity is not None:
                    res = self._on_opportunity(opp)
                    if asyncio.iscoroutine(res):
                        await res
        return out

    def _probe_size(self, symbol: str, buy_book, sell_book) -> Decimal:
        """
        Probe size for VWAP evaluation. Must match the max notional the risk
        engine may approve (``max_notional_per_trade``), otherwise we estimate
        slippage at depth N but actually fill at depth M > N, which silently
        turns positive-edge opportunities into loss-making fills on thin
        small-cap books (PEPE / BONK / FLOKI / SHIB etc. where levels 2-5
        can span tens of bps).
        """
        mid = buy_book.mid_price or Decimal(1)
        if mid == 0:
            mid = Decimal(1)
        return self._settings.max_notional_per_trade / mid
