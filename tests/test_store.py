from memory.store import MemoryStore, Scope
from tests.conftest import fake_embed

ALICE = Scope(team_id="payments", user_id="alice")
BOB = Scope(team_id="payments", user_id="bob")
EVE = Scope(team_id="search", user_id="eve")


async def test_scopes(pool):
    store = MemoryStore(pool, fake_embed)
    await store.add(ALICE, "Refunds are processed by the ledger service",
                    ["refunds", "ledger"], "def refund(): ...", source="ledger.py")
    await store.add(ALICE, "Alice prefers refunds explained with examples",
                    ["refunds"], "chat transcript", private=True)

    # Team-shared memory is visible to the team; private memory only to its owner.
    assert len(await store.search(ALICE, "refunds ledger")) == 2
    bob_hits = await store.search(BOB, "refunds ledger")
    assert [m.source for m in bob_hits] == ["ledger.py"]
    # Another team sees nothing.
    assert await store.search(EVE, "refunds ledger") == []


async def test_pages_follow_summaries(pool):
    store = MemoryStore(pool, fake_embed)
    summary_id = await store.add(BOB, "Webhook retries back off exponentially",
                                 ["webhook", "retry"], "RETRY_DELAYS = [1, 2, 4, 8]")
    assert [p.content for p in await store.pages(BOB, [summary_id])] == ["RETRY_DELAYS = [1, 2, 4, 8]"]
    assert await store.pages(EVE, [summary_id]) == []


async def test_cannot_write_into_another_team(pool):
    store = MemoryStore(pool, fake_embed)
    async with store.scoped(EVE) as conn:
        try:
            await conn.execute(
                "INSERT INTO memories (team_id, kind, content, token_count) "
                "VALUES ('payments', 'summary', 'injected', 1)"
            )
        except Exception as e:
            assert "row-level security" in str(e)
        else:
            raise AssertionError("RLS allowed a cross-team write")


async def test_no_scope_sees_nothing(pool):
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SET LOCAL ROLE gateway_app")
        cur = await conn.execute("SELECT count(*) FROM memories")
        assert (await cur.fetchone())[0] == 0


async def test_other_teams_rows_do_not_crowd_out_search(pool):
    store = MemoryStore(pool, fake_embed)
    async with pool.connection() as conn:
        vector = (await store._vectors(["refunds ledger noise"]))[0]
        await conn.cursor().executemany(
            "INSERT INTO memories (team_id, kind, content, token_count, embedding)"
            " VALUES (%s, 'summary', 'refunds ledger noise', 3, %s)",
            [(f"crowd-{i}", vector) for i in range(500)],
        )
    scope = Scope("crowded", "zoe")
    await store.add(scope, "Refunds go through the ledger", ["refunds"], "page")
    assert [m.content for m in await store.search(scope, "refunds ledger")] == [
        "Refunds go through the ledger"
    ]


async def test_has_source(pool):
    store = MemoryStore(pool, fake_embed)
    scope = Scope("sources", "yan")
    await store.add(scope, "s", [], "p", source="a.py")
    assert await store.has_source(scope, "a.py")
    assert not await store.has_source(Scope("other", "yan"), "a.py")
