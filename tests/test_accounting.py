import asyncio

import pytest

from gateway.accounting import Ledger, SpendCapExceeded
from memory.store import Scope

SCOPE = Scope("acct-team", "dave")


def test_cost_from_exact_tokens():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=2.0, output_price_per_m=10.0)
    assert ledger.cost(1_000_000, 100_000) == pytest.approx(3.0)


async def test_cap_counts_recorded_spend(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    ledger = Ledger(pool, cap_usd=1.0, input_price_per_m=1e6, output_price_per_m=0)
    await ledger.record(SCOPE, "muse", "escalate_compact", {"prompt_tokens": 1}, 10, paid=True)
    assert await ledger.spent() == pytest.approx(1.0)
    with pytest.raises(SpendCapExceeded):
        async with ledger.reserve(0.01):
            pass
    # Free (local) calls are recorded at zero cost.
    await ledger.record(SCOPE, "llama", "local", {"prompt_tokens": 50}, 10, paid=False)
    assert await ledger.spent() == pytest.approx(1.0)


async def test_reservations_block_concurrent_overspend(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    ledger = Ledger(pool, cap_usd=1.0, input_price_per_m=0, output_price_per_m=0)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first():
        async with ledger.reserve(0.8):
            entered.set()
            await release.wait()

    task = asyncio.create_task(first())
    await entered.wait()
    with pytest.raises(SpendCapExceeded):
        async with ledger.reserve(0.3):
            pass
    release.set()
    await task
    async with ledger.reserve(0.3):
        pass
