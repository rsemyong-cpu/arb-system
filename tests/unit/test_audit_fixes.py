"""Regression tests for the 5 fixes from the 2026-04 code audit:

1. probe_size equals max_notional_per_trade (no longer silently clamped at 200)
2. scanner loop fire-and-forget hedge exec (covered by event-loop behaviour,
   unit-tested via _execute_hedge_safe wrapper contract)
3. REST orderbook depth bumped to 10 (static constant)
4. RepairEngine applies ioc_price_buffer_bps to its IOC reference price
5. HedgeCoordinator re-reads latest book before submit and aborts on
   edge-collapse
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from app.accounts.balance_manager import BalanceManager
from app.adapters.ccxt_adapter import _ORDERBOOK_DEPTH
from app.adapters.mock_adapter import MockExchangeAdapter
from app.adapters.registry import AdapterRegistry
from app.common.enums import HedgeState
from app.config.settings import Settings
from app.execution.hedge_coordinator import HedgeCoordinator
from app.execution.order_router import OrderRouter
from app.execution.order_tracker import OrderTracker
from app.execution.paper_fill_engine import PaperFillConfig, PaperFillEngine
from app.execution.repair_engine import RepairEngine
from app.marketdata.orderbook_manager import OrderBookManager
from app.models.hedge import HedgeGroupState
from app.models.opportunity import ArbitrageOpportunity
from app.risk.exposure_manager import ExposureManager
from app.strategy.fee_model import FeeModel
from app.strategy.opportunity_scanner import OpportunityScanner
from app.strategy.spread_calculator import SpreadCalculator


def _opp(
    *,
    buy_price: Decimal = Decimal("100"),
    sell_price: Decimal = Decimal("101"),
    gross_bps: Decimal = Decimal("100"),
    net_edge_bps: Decimal = Decimal("90"),
) -> ArbitrageOpportunity:
    return ArbitrageOpportunity(
        opportunity_id="opp_test",
        symbol="BTC/USDT",
        buy_exchange="a",
        sell_exchange="b",
        buy_price=buy_price,
        sell_price=sell_price,
        gross_spread_bps=gross_bps,
        buy_fee_bps=Decimal("1"),
        sell_fee_bps=Decimal("1"),
        slippage_bps=Decimal("0"),
        buffer_bps=Decimal("0"),
        net_edge_bps=net_edge_bps,
        max_tradable_size=Decimal("1"),
        expected_profit_quote=Decimal("1"),
        detected_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Issue 1: probe_size
# ---------------------------------------------------------------------------


def test_probe_size_uses_max_notional_per_trade():
    """probe_quote should equal max_notional_per_trade (not min(.., 200))."""
    s = Settings(
        max_notional_per_trade=Decimal("400"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )

    books = OrderBookManager(max_stale_ms=999999)
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    a.set_orderbook("BTC/USDT", bids=[(99, 1)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(101, 10)], asks=[(102, 1)])
    import asyncio

    books.update(asyncio.get_event_loop().run_until_complete(a.watch_orderbook("BTC/USDT")))
    books.update(asyncio.get_event_loop().run_until_complete(b.watch_orderbook("BTC/USDT")))

    bm = BalanceManager(AdapterRegistry([a, b]))
    fee_model = FeeModel(s)
    spread_calc = SpreadCalculator(fee_model)

    sc = OpportunityScanner(s, books, bm, spread_calc)
    probe = sc._probe_size("BTC/USDT", books.get("a", "BTC/USDT"), books.get("b", "BTC/USDT"))
    # mid ~ 100, max_notional 400 -> probe = 4.0 base (NOT 2.0 which would be 200/mid)
    assert probe == Decimal("400") / books.get("a", "BTC/USDT").mid_price


def test_probe_size_scales_with_higher_max_notional():
    """Changing max_notional_per_trade changes probe size proportionally."""
    s_small = Settings(
        max_notional_per_trade=Decimal("100"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    s_big = Settings(
        max_notional_per_trade=Decimal("4000"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    books = OrderBookManager(max_stale_ms=999999)
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    a.set_orderbook("BTC/USDT", bids=[(99, 1)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(101, 10)], asks=[(102, 1)])
    import asyncio

    books.update(asyncio.get_event_loop().run_until_complete(a.watch_orderbook("BTC/USDT")))
    books.update(asyncio.get_event_loop().run_until_complete(b.watch_orderbook("BTC/USDT")))
    bm = BalanceManager(AdapterRegistry([a, b]))
    fm = FeeModel(s_small)
    sc_small = OpportunityScanner(s_small, books, bm, SpreadCalculator(fm))
    sc_big = OpportunityScanner(s_big, books, bm, SpreadCalculator(fm))
    p_small = sc_small._probe_size("BTC/USDT", books.get("a", "BTC/USDT"), books.get("b", "BTC/USDT"))
    p_big = sc_big._probe_size("BTC/USDT", books.get("a", "BTC/USDT"), books.get("b", "BTC/USDT"))
    # 4000 / 100 = 40x larger probe (allow small rounding in Decimal division)
    ratio = p_big / p_small
    assert abs(ratio - Decimal(40)) < Decimal("0.001")


# ---------------------------------------------------------------------------
# Issue 3: orderbook depth
# ---------------------------------------------------------------------------


def test_rest_orderbook_depth_is_ten():
    assert _ORDERBOOK_DEPTH == 10


# ---------------------------------------------------------------------------
# Issue 4: repair IOC buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_applies_ioc_buffer_buy_side():
    """Repair BUY should use price HIGHER than best_ask by buffer bps."""
    s = Settings(
        mode="paper-trade",
        max_repair_attempts=3,
        ioc_price_buffer_bps=Decimal("10"),  # 10 bps aggressive buffer
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    a.set_orderbook("BTC/USDT", bids=[(99, 10)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(101, 10)], asks=[(102, 10)])
    reg = AdapterRegistry([a, b])

    books = OrderBookManager(max_stale_ms=999999)
    books.update(await a.watch_orderbook("BTC/USDT"))
    books.update(await b.watch_orderbook("BTC/USDT"))

    paper = PaperFillEngine(books, cfg=PaperFillConfig(partial_fill_probability=0.0))
    tracker = OrderTracker()
    router = OrderRouter(s, reg, paper)
    repair = RepairEngine(s, router, tracker, books)

    # Simulate a group that needs BUY repair (net_position_base < 0 => buy side)
    group = HedgeGroupState(
        hedge_group_id="h_buy",
        opportunity_id="opp_x",
        symbol="BTC/USDT",
        buy_exchange="a",
        sell_exchange="b",
        target_amount=Decimal("1"),
        state=HedgeState.FAILED_NEEDS_REPAIR,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        executed_buy_amount=Decimal("0.5"),
        executed_sell_amount=Decimal("1.0"),
        net_position_base=Decimal("-0.5"),  # short -> buy to cover
    )
    # Capture submitted intent
    captured: list = []
    orig_submit = router.submit

    async def _capture(intent):
        captured.append(intent)
        return await orig_submit(intent)

    router.submit = _capture  # type: ignore[assignment]

    await repair.repair(group)
    assert len(captured) == 1
    intent = captured[0]
    # best_ask on 'a' is 100. Buffer 10 bps -> 100 * 1.001 = 100.1
    assert intent.price == Decimal("100") * (Decimal(1) + Decimal("10") / Decimal("10000"))


@pytest.mark.asyncio
async def test_repair_applies_ioc_buffer_sell_side():
    """Repair SELL should use price LOWER than best_bid by buffer bps."""
    s = Settings(
        mode="paper-trade",
        max_repair_attempts=3,
        ioc_price_buffer_bps=Decimal("10"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    a.set_orderbook("BTC/USDT", bids=[(99, 10)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(101, 10)], asks=[(102, 10)])
    reg = AdapterRegistry([a, b])

    books = OrderBookManager(max_stale_ms=999999)
    books.update(await a.watch_orderbook("BTC/USDT"))
    books.update(await b.watch_orderbook("BTC/USDT"))

    paper = PaperFillEngine(books, cfg=PaperFillConfig(partial_fill_probability=0.0))
    tracker = OrderTracker()
    router = OrderRouter(s, reg, paper)
    repair = RepairEngine(s, router, tracker, books)

    # net_position_base > 0 => sell side
    group = HedgeGroupState(
        hedge_group_id="h_sell",
        opportunity_id="opp_y",
        symbol="BTC/USDT",
        buy_exchange="a",
        sell_exchange="b",
        target_amount=Decimal("1"),
        state=HedgeState.FAILED_NEEDS_REPAIR,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        executed_buy_amount=Decimal("1.0"),
        executed_sell_amount=Decimal("0.5"),
        net_position_base=Decimal("0.5"),  # long -> sell to cover
    )
    captured: list = []
    orig_submit = router.submit

    async def _capture(intent):
        captured.append(intent)
        return await orig_submit(intent)

    router.submit = _capture  # type: ignore[assignment]

    await repair.repair(group)
    assert len(captured) == 1
    intent = captured[0]
    # sell on 'b', best_bid = 101. Buffer 10 bps -> 101 * 0.999 = 100.899
    assert intent.price == Decimal("101") * (Decimal(1) - Decimal("10") / Decimal("10000"))


# ---------------------------------------------------------------------------
# Issue 5: pre-submit re-check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presubmit_aborts_when_edge_collapses():
    """If the book has moved against us between scan and submit, abort."""
    s = Settings(
        mode="paper-trade",
        min_net_edge_bps=Decimal("3"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    # Current book has NO spread (both tops identical)
    a.set_orderbook("BTC/USDT", bids=[(99, 10)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(100, 10)], asks=[(100.5, 10)])
    reg = AdapterRegistry([a, b])

    books = OrderBookManager(max_stale_ms=999999)
    books.update(await a.watch_orderbook("BTC/USDT"))
    books.update(await b.watch_orderbook("BTC/USDT"))

    paper = PaperFillEngine(books, cfg=PaperFillConfig(partial_fill_probability=0.0))
    tracker = OrderTracker()
    router = OrderRouter(s, reg, paper)
    repair = RepairEngine(s, router, tracker, books)
    hedge = HedgeCoordinator(s, router, tracker, repair, ExposureManager(), book_mgr=books)

    # opp claims big edge from a stale scan (gross=100bps, net=90bps)
    opp = _opp(
        buy_price=Decimal("100"),
        sell_price=Decimal("101"),
        gross_bps=Decimal("100"),
        net_edge_bps=Decimal("90"),
    )
    # required fresh gross = 100 - 90 + 3 = 13 bps. Actual fresh gross = (100 - 100)/100 = 0.
    group = await hedge.execute(opp, approved_amount=Decimal("0.1"))
    assert group.state == HedgeState.ABORTED
    assert any("pre_submit_edge_collapsed" in n for n in group.notes)


@pytest.mark.asyncio
async def test_presubmit_proceeds_when_edge_still_intact():
    """If the fresh book still shows enough edge, hedge proceeds normally."""
    s = Settings(
        mode="paper-trade",
        min_net_edge_bps=Decimal("3"),
        postgres_dsn="postgresql+asyncpg://x:x@localhost/none",
        redis_url="redis://localhost:6379/0",
    )
    a = MockExchangeAdapter("a")
    b = MockExchangeAdapter("b")
    # Book STILL shows the spread from scan time
    a.set_orderbook("BTC/USDT", bids=[(99, 10)], asks=[(100, 10)])
    b.set_orderbook("BTC/USDT", bids=[(101, 10)], asks=[(102, 10)])
    reg = AdapterRegistry([a, b])

    books = OrderBookManager(max_stale_ms=999999)
    books.update(await a.watch_orderbook("BTC/USDT"))
    books.update(await b.watch_orderbook("BTC/USDT"))

    bm = BalanceManager(reg)
    bm.set_virtual_balance("a", "USDT", Decimal("10000"))
    bm.set_virtual_balance("b", "BTC", Decimal("10"))

    paper = PaperFillEngine(books, cfg=PaperFillConfig(partial_fill_probability=0.0))
    tracker = OrderTracker()
    router = OrderRouter(s, reg, paper)
    repair = RepairEngine(s, router, tracker, books)
    hedge = HedgeCoordinator(s, router, tracker, repair, ExposureManager(), book_mgr=books)

    opp = _opp(
        buy_price=Decimal("100"),
        sell_price=Decimal("101"),
        gross_bps=Decimal("100"),
        net_edge_bps=Decimal("90"),
    )
    group = await hedge.execute(opp, approved_amount=Decimal("0.1"))
    assert group.state != HedgeState.ABORTED
