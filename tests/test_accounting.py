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


async def test_paid_call_without_usage_is_charged_the_estimate(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    ledger = Ledger(pool, cap_usd=1.0, input_price_per_m=1.0, output_price_per_m=1.0)
    await ledger.record(SCOPE, "muse", "escalate_compact", {}, 10, paid=True, estimate_usd=0.25)
    assert await ledger.spent() == pytest.approx(0.25)


async def test_cached_tokens_are_recorded(pool):
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE usage")
    ledger = Ledger(pool, cap_usd=1.0, input_price_per_m=0, output_price_per_m=0)
    usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 64}}
    await ledger.record(SCOPE, "llama", "local", usage, 10, paid=False)
    assert (await ledger.summary())[0]["cached_tokens"] == 64


def test_worst_case_is_an_upper_bound_for_code():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=1e6, output_price_per_m=0)
    code = "x=f(a,b);" * 100  # ~1 token per 2-3 characters
    assert ledger.worst_case_cost({"messages": [{"content": code}]}) >= len(code) / 2


def test_output_is_limited_to_what_is_reserved():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=0, output_price_per_m=1e6,
                    max_output_tokens=100)
    assert ledger.limited({"messages": []})["max_tokens"] == 100
    assert ledger.limited({"messages": [], "max_tokens": 10})["max_tokens"] == 10
    limited = ledger.limited({"messages": [], "max_completion_tokens": 5000})
    assert limited["max_tokens"] == 100 and "max_completion_tokens" not in limited
    # Each of n choices can write the whole limit.
    assert ledger.worst_case_cost(ledger.limited({"messages": [], "n": 3})) == pytest.approx(300)


def test_prompt_bound_counts_every_byte():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=1e6, output_price_per_m=0)
    text = "é" * 1000  # 2 bytes each; dense text can tokenize at close to a token per byte
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1}
    assert ledger.worst_case_cost(body) >= 2000


def test_cached_prompt_tokens_at_their_own_price():
    ledger = Ledger(pool=None, cap_usd=1, input_price_per_m=1.25, output_price_per_m=4.25,
                    cached_input_price_per_m=0.15)
    assert ledger.cost(1_000_000, 0, cached_tokens=1_000_000) == pytest.approx(0.15)
    assert ledger.cost(1_000_000, 1_000_000, cached_tokens=0) == pytest.approx(5.5)
