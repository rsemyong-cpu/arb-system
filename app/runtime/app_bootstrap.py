"""
Assembles and wires all components into a Container. Also provides a start/stop
lifecycle for the background loops (market data, balances, scanner).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from app.accounts.account_reconciler import AccountReconciler
from app.accounts.balance_manager import BalanceManager
from app.adapters.perp_registry import build_default_perp_registry
from app.adapters.registry import build_default_registry
from app.common.clock import utcnow
from app.common.enums import Mode, Severity
from app.common.logging import configure_logging, get_logger
from app.config.settings import Settings, get_settings
from app.execution.funding_executor import FundingExecutor
from app.execution.hedge_coordinator import HedgeCoordinator
from app.execution.order_router import OrderRouter
from app.execution.order_tracker import OrderTracker
from app.execution.paper_fill_engine import PaperFillEngine
from app.execution.repair_engine import RepairEngine
from app.execution.triangular_executor import TriangularExecutor
from app.marketdata.orderbook_manager import OrderBookManager
from app.risk.circuit_breaker import CircuitBreaker
from app.risk.exposure_manager import ExposureManager
from app.risk.health_guard import HealthGuard
from app.risk.kill_switch import KillSwitch
from app.risk.rules import RiskEngine
from app.runtime.dependency_container import Container
from app.services.alert_service import AlertService
from app.services.config_service import ConfigService
from app.services.metrics_service import get_metrics
from app.services.report_service import ReportService
from app.storage.db import Database
from app.storage.repositories.config_snapshots import ConfigSnapshotRepo
from app.storage.repositories.events import EventRepo
from app.storage.repositories.hedges import HedgeRepo
from app.storage.repositories.opportunities import OpportunityRepo
from app.storage.repositories.orders import OrderRepo
from app.strategy.fee_model import FeeModel
from app.strategy.funding_rate_scanner import FundingRateScanner
from app.strategy.opportunity_scanner import OpportunityScanner
from app.strategy.spread_calculator import SpreadCalculator
from app.strategy.triangular_scanner import TriangularScanner

log = get_logger("runtime.bootstrap")


_bg_tasks: list[asyncio.Task] = []


def _build_container(settings: Settings) -> Container:
    registry = build_default_registry(settings)
    perp_registry = build_default_perp_registry(settings)
    book_mgr = OrderBookManager(
        max_stale_ms=settings.max_marketdata_staleness_ms,
        poll_interval_ms=settings.scan_interval_ms,
        marketdata_mode=settings.marketdata_mode,
    )
    balance_mgr = BalanceManager(registry, refresh_interval_sec=15, settings=settings)
    reconciler = AccountReconciler(balance_mgr)

    # The override getter reads *current* settings on every call so that
    # operators can toggle the override at runtime without a restart.
    # Seed the fee table with per-exchange defaults from the catalog so that
    # the scanner has plausible fees even when ccxt's ``markets[sym].taker``
    # is missing (common on exchanges that don't return per-symbol fees).
    from app.adapters.exchanges_catalog import SUPPORTED_EXCHANGES
    from app.strategy.fee_model import FeeTable

    fee_table = FeeTable(per_exchange_bps={s.id: s.default_taker_bps for s in SUPPORTED_EXCHANGES})
    fee_model = FeeModel(
        registry,
        table=fee_table,
        override_bps_getter=lambda: settings.fee_override_bps,
    )
    spread_calc = SpreadCalculator(fee_model, buffer_getter=lambda: settings.scan_buffer_bps)
    kill = KillSwitch(initial=settings.kill_switch)
    breaker = CircuitBreaker(max_consecutive_failures=settings.max_consecutive_failures)
    health = HealthGuard(book_mgr, balance_mgr, settings)
    exposure = ExposureManager()
    risk = RiskEngine(settings, kill, breaker, health, exposure, balance_mgr)

    paper = PaperFillEngine(book_mgr, balance_mgr=balance_mgr)
    tracker = OrderTracker()
    router = OrderRouter(settings, registry, paper)
    repair = RepairEngine(settings, router, tracker, book_mgr)

    hedge = HedgeCoordinator(
        settings,
        router,
        tracker,
        repair,
        exposure,
        breaker=breaker,
        registry=registry,
        book_mgr=book_mgr,
    )
    from app.execution.maker_taker_executor import MakerTakerExecutor

    maker_taker = MakerTakerExecutor(settings, router, book_mgr)
    scanner = OpportunityScanner(settings, book_mgr, balance_mgr, spread_calc)
    triangular = TriangularScanner(settings, registry, book_mgr, fee_model)
    funding = FundingRateScanner(settings, book_mgr)
    triangular_exec = TriangularExecutor(router=router, paper=paper)
    funding_exec = FundingExecutor(settings, book_mgr, paper, balance_mgr)

    # Configure the funding executor with registry lookups (late-bound to
    # keep the executor independent of the container layout).
    def _spot_lookup(name: str):
        try:
            return registry.get(name)
        except KeyError:
            return None

    def _perp_lookup(name: str):
        return perp_registry.get(name)

    funding_exec.configure(_spot_lookup, _perp_lookup)

    # Bridge: when the triangular scanner finds a qualifying opportunity
    # AND we're running in paper-trade or live mode, kick off the 3-leg
    # sequential executor. dry-run is skipped — the scanner's detect
    # stream is enough for audit purposes.
    async def _maybe_execute_triangular(opp):
        if settings.mode == Mode.DRY_RUN.value:
            return
        if not settings.strategy_triangular_same_exchange_enabled:
            return
        probe = Decimal(settings.max_notional_per_trade)
        try:
            await triangular_exec.execute(opp, probe, mode_label=settings.mode)
        except Exception as e:  # noqa: BLE001
            log.error("triangular_exec_error", error=str(e))

    triangular.set_on_opportunity(_maybe_execute_triangular)

    async def _maybe_execute_funding(opp):
        # paper-trade and live both execute; dry-run is skipped inside
        # FundingExecutor.execute by checking settings.mode.
        try:
            await funding_exec.execute(opp)
        except Exception as e:  # noqa: BLE001
            log.error("funding_exec_error", error=str(e))

    funding.set_on_opportunity(_maybe_execute_funding)
    metrics = get_metrics()
    alerts = AlertService(settings)
    config_service = ConfigService(settings)

    return Container(
        settings=settings,
        registry=registry,
        perp_registry=perp_registry,
        book_mgr=book_mgr,
        balance_mgr=balance_mgr,
        reconciler=reconciler,
        fee_model=fee_model,
        spread_calc=spread_calc,
        scanner=scanner,
        triangular=triangular,
        funding=funding,
        kill=kill,
        breaker=breaker,
        health=health,
        exposure=exposure,
        risk=risk,
        paper=paper,
        router=router,
        tracker=tracker,
        repair=repair,
        hedge=hedge,
        maker_taker=maker_taker,
        metrics=metrics,
        alerts=alerts,
        config_service=config_service,
        db=None,
        report=None,
        opp_repo=None,
        hedge_repo=None,
        order_repo=None,
        event_repo=None,
    )


async def bootstrap(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    log.info(
        "bootstrap_start",
        env=settings.env,
        mode=settings.mode,
        symbols=settings.enabled_symbol_list,
    )

    # ccxt is sync — every fetch_order_book / fetch_time runs on the
    # default ThreadPoolExecutor (asyncio.to_thread). Python's default has
    # only ``min(32, os.cpu_count() + 4)`` workers, which on a 2-vCPU box
    # is just 6. With 9 exchanges × 18 symbols = 162 polls every
    # scan_interval_ms, that pool saturates and even cheap calls (like the
    # fetch_time used by /health/latency) queue behind orderbook fetches,
    # producing inflated app-level "latency" of 1.5–2.5 s while the
    # underlying network RTT is single-digit ms. Bumping to 256 workers
    # decouples latency probes from the orderbook poll fan-out.
    pool = ThreadPoolExecutor(max_workers=256, thread_name_prefix="ccxt")
    asyncio.get_event_loop().set_default_executor(pool)
    log.info("threadpool_configured", max_workers=256)

    c = _build_container(settings)

    # DB optional — if unreachable, we keep running but without persistence.
    db: Database | None = None
    try:
        db = Database(settings.postgres_dsn)
        await db.create_all()
        c.db = db
        c.report = ReportService(db)
        c.opp_repo = OpportunityRepo(db)
        c.hedge_repo = HedgeRepo(db)
        c.order_repo = OrderRepo(db)
        c.event_repo = EventRepo(db)
        # Wire the persistent config snapshot. We load it BEFORE adapter
        # connect / scanner start so the operator's last-saved selections
        # (enabled symbols, threshold tweaks, strategy on/off, selected
        # exchanges) are in effect from the very first scan cycle.
        config_repo = ConfigSnapshotRepo(db)
        c.config_service.attach_repo(config_repo)
        try:
            applied = await c.config_service.load_from_db()
            if applied:
                log.info("config_snapshot_restored", fields=applied)
        except Exception as e:  # noqa: BLE001
            log.warning("config_snapshot_restore_failed", error=str(e))
        log.info("db_ready")
    except Exception as e:  # noqa: BLE001
        log.warning("db_unavailable_running_in_memory", error=str(e))

    # Startup reconciliation: any hedge group the DB still has as active
    # (i.e. not completed/aborted) outlived its parent process. We don't try
    # to auto-resume — instead we mark them with a note, surface them in /events
    # so an operator can decide what to do.
    if c.hedge_repo is not None:
        try:
            active = await c.hedge_repo.active()
            if active:
                log.warning("startup_found_active_hedges", count=len(active))
                for row in active:
                    if c.event_repo is not None:
                        await _safe(
                            c.event_repo.log(
                                event_type="startup_orphan_hedge",
                                severity="warning",
                                component="bootstrap",
                                message=f"hedge_group={row.id} still {row.state} at startup",
                                payload={
                                    "hedge_group_id": row.id,
                                    "symbol": row.symbol,
                                    "state": row.state,
                                    "executed_buy": str(row.executed_buy_amount),
                                    "executed_sell": str(row.executed_sell_amount),
                                    "net_position_base": str(row.net_position_base),
                                },
                            )
                        )
        except Exception as e:  # noqa: BLE001
            log.warning("startup_reconcile_error", error=str(e))

    # Virtual balances for paper-trade so the risk engine has something to gate on.
    #
    # Rule: each base asset is seeded with ≥ 10_000 USDT-equivalent at an
    # approximate recent market price, so small-cap coins (SHIB/PEPE etc.)
    # don't run out after the very first paper fill. Prices below are rough
    # snapshots — we deliberately overshoot on the conservative side so even
    # after 2-3x price moves the virtual budget is still meaningful.
    # Unknown assets fall back to 100_000 units (a safe over-provision).
    if settings.mode in (Mode.PAPER_TRADE.value, Mode.DRY_RUN.value):
        _APPROX_USD_PRICE = {
            "BTC": Decimal("60000"),
            "ETH": Decimal("3000"),
            "SOL": Decimal("150"),
            "BNB": Decimal("600"),
            "XRP": Decimal("0.55"),
            "ADA": Decimal("0.5"),
            "LINK": Decimal("15"),
            "DOGE": Decimal("0.12"),
            "ARB": Decimal("1.0"),
            "OP": Decimal("2.0"),
            "SUI": Decimal("1.0"),
            "WIF": Decimal("2.5"),
            "PEPE": Decimal("0.000012"),
            "SHIB": Decimal("0.000025"),
            "BONK": Decimal("0.000025"),
            "FLOKI": Decimal("0.00015"),
            "JTO": Decimal("3.5"),
            "TIA": Decimal("8"),
            "ORDI": Decimal("45"),
        }
        TARGET_USD = Decimal("10000")
        for ex in c.registry.names():
            c.balance_mgr.set_virtual_balance(ex, "USDT", TARGET_USD)
            for s in settings.enabled_symbol_list:
                base = s.split("/")[0]
                px = _APPROX_USD_PRICE.get(base)
                if px is not None and px > 0:
                    # round to 6 decimals so display stays readable on big coins
                    amount = (TARGET_USD / px).quantize(Decimal("0.000001"))
                else:
                    amount = Decimal("100000")
                c.balance_mgr.set_virtual_balance(ex, base, amount)

    # Connect adapters (non-fatal if fails)
    try:
        await c.registry.connect_all()
    except Exception as e:  # noqa: BLE001
        log.warning("adapter_connect_issue", error=str(e))

    # Background tasks
    _bg_tasks.append(
        asyncio.create_task(
            c.book_mgr.run(c.registry.all(), settings.enabled_symbol_list),
            name="orderbook_loop",
        )
    )
    # Balance loop runs always, but the BalanceManager itself skips the
    # real-exchange fetch whenever mode != live — so dry-run / paper-trade
    # never need API keys and virtual balances are never overwritten.
    _bg_tasks.append(asyncio.create_task(c.balance_mgr.run(), name="balance_loop"))
    _bg_tasks.append(asyncio.create_task(_scanner_loop(c), name="scanner_loop"))
    _bg_tasks.append(asyncio.create_task(_triangular_supervisor(c), name="triangular_supervisor"))
    _bg_tasks.append(asyncio.create_task(_funding_supervisor(c), name="funding_supervisor"))

    return c


async def _funding_supervisor(c: Container) -> None:
    """Start/stop funding-rate scanner based on the strategy flag.

    Same pattern as ``_triangular_supervisor`` — poll every 2 seconds and
    manage a single child task.
    """
    task: asyncio.Task | None = None
    while True:
        try:
            enabled = c.settings.strategy_funding_rate_spot_perp_enabled
            if enabled and (task is None or task.done()):
                c.funding.start()
                task = asyncio.create_task(c.funding.run(), name="funding_scan")
            elif not enabled and task is not None and not task.done():
                c.funding.stop()
        except asyncio.CancelledError:
            if task and not task.done():
                c.funding.stop()
            raise
        except Exception as e:  # noqa: BLE001
            log.error("funding_supervisor_error", error=str(e))
        await asyncio.sleep(2.0)


async def _triangular_supervisor(c: Container) -> None:
    """Start/stop the triangular scanner based on the strategy flag.

    Poll settings every 2 s. When the flag flips on and no scanner task is
    currently running, spawn one; when it flips off, the scanner's own run
    loop will exit cleanly. We never kill mid-iteration here.
    """
    task: asyncio.Task | None = None
    while True:
        try:
            enabled = c.settings.strategy_triangular_same_exchange_enabled
            if enabled and (task is None or task.done()):
                c.triangular.start()
                task = asyncio.create_task(c.triangular.run(), name="triangular_scan")
            elif not enabled and task is not None and not task.done():
                c.triangular.stop()  # loop will exit on next iteration
        except asyncio.CancelledError:
            if task and not task.done():
                c.triangular.stop()
            raise
        except Exception as e:  # noqa: BLE001
            log.error("triangular_supervisor_error", error=str(e))
        await asyncio.sleep(2.0)


async def _scanner_loop(c: Container) -> None:
    settings = c.settings
    c.scanner.start()
    interval = settings.scan_interval_ms / 1000.0
    from app.common.enums import Mode as _M

    while c.scanner.is_running():
        # Respect the operator-set pause flag without dropping out of the loop
        # (so unpausing is instantaneous).
        if settings.paused:
            await asyncio.sleep(interval)
            continue
        # Strategy-level master switch. We only run cross_exchange_spot today;
        # when its toggle is off the loop idles (but still obeys pause/mode).
        if not getattr(settings, "strategy_cross_exchange_spot_enabled", True):
            await asyncio.sleep(interval)
            continue
        try:
            # Restrict scanning to the operator-selected exchanges for the
            # cross-exchange-spot strategy. Defaults to all registered if the
            # setting is empty (first boot).
            raw = (getattr(settings, "strategy_cross_exchange_spot_exchanges", "") or "").strip()
            selected = [x.strip() for x in raw.split(",") if x.strip()]
            registry_names = c.registry.names()
            ex_list = [n for n in registry_names if n in selected] if selected else registry_names
            if len(ex_list) < 2:
                # Under-constrained — cross-ex arb needs ≥2 venues.
                await asyncio.sleep(interval)
                continue
            opps = await c.scanner.scan_once(ex_list)
            for opp in opps:
                c.scanner.record_detected()
                c.metrics.opp_detected_total.labels(symbol=opp.symbol).inc()
                c.metrics.net_edge_bps_hist.labels(symbol=opp.symbol).observe(float(opp.net_edge_bps))
                decision = c.risk.evaluate(opp)
                if decision.approved:
                    opp.decision = "accepted"
                    c.scanner.record_decision(accepted=True)
                    c.metrics.opp_accepted_total.labels(symbol=opp.symbol).inc()
                    if c.opp_repo:
                        await _safe(c.opp_repo.save(opp, decision="accepted"))
                    # In dry-run we still go through execute so the full audit trail is produced.
                    if settings.mode in (_M.DRY_RUN.value, _M.PAPER_TRADE.value, _M.LIVE.value):
                        # Set cooldown synchronously BEFORE spawning the task so
                        # the next scan_once won't produce another opp for the
                        # same symbol while this one is still submitting.
                        c.risk.set_cooldown(opp.symbol)
                        # Route to maker-taker executor when that mode is selected
                        # AND we're not dry-run (maker-taker is runtime-only; dry
                        # run always goes through the audit path in hedge.execute
                        # so the risk/repair chain is exercised).
                        if (
                            getattr(settings, "execution_mode", "taker_taker") == "maker_taker"
                            and settings.mode != _M.DRY_RUN.value
                        ):
                            asyncio.create_task(
                                _execute_maker_taker_safe(c, opp, decision.approved_amount),
                                name=f"mt_exec_{opp.symbol}",
                            )
                        else:
                            # Fire-and-forget: scanner keeps looking for new opps
                            # while this hedge's 2x network submit + poll + any
                            # repair runs concurrently. Blocking here (the prior
                            # behavior) lost up to 50-75 scan cycles per trade on
                            # a 10-15s-long hedge execution — enough for typical
                            # sub-second cross-exchange windows to have closed.
                            asyncio.create_task(
                                _execute_hedge_safe(c, opp, decision.approved_amount),
                                name=f"hedge_exec_{opp.symbol}",
                            )
                else:
                    opp.decision = "rejected"
                    opp.decision_reason = decision.reason.value if decision.reason else "unknown"
                    c.scanner.record_decision(accepted=False, reason=opp.decision_reason)
                    c.metrics.opp_rejected_total.labels(symbol=opp.symbol, reason=opp.decision_reason).inc()
                    c.metrics.risk_reject_total.labels(reason=opp.decision_reason).inc()
                    if c.opp_repo:
                        await _safe(c.opp_repo.save(opp, decision="rejected", reason=opp.decision_reason))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.error("scanner_iteration_error", error=str(e))
        await asyncio.sleep(interval)


async def _safe(coro) -> None:
    try:
        await coro
    except Exception as e:  # noqa: BLE001
        log.warning("persistence_error", error=str(e))


async def _execute_hedge_safe(c: Container, opp, approved_amount) -> None:
    """Background hedge execution. Runs in its own task so the scanner loop
    can keep polling while this hedge's network submissions + settlement
    poll + any repair attempt complete (10-15s worst case per hedge).
    All errors are caught so a buggy task can't bring down the scheduler.
    """
    try:
        group = await c.hedge.execute(opp, approved_amount)
        if c.hedge_repo:
            await _safe(c.hedge_repo.upsert(group))
    except Exception as e:  # noqa: BLE001
        log.error("hedge_exec_error", symbol=opp.symbol, error=str(e))


async def _execute_maker_taker_safe(c: Container, opp, approved_amount) -> None:
    """Background maker-taker execution. Same safety contract as
    ``_execute_hedge_safe``.
    """
    try:
        await c.maker_taker.execute(opp, approved_amount)
    except Exception as e:  # noqa: BLE001
        log.error("maker_taker_exec_error", symbol=opp.symbol, error=str(e))


async def teardown(c: Container) -> None:
    c.scanner.stop()
    await c.book_mgr.stop()
    await c.balance_mgr.stop()
    for t in _bg_tasks:
        t.cancel()
    for t in _bg_tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    _bg_tasks.clear()
    await c.registry.close_all()
    if c.db:
        await c.db.close()
    log.info("teardown_complete")
