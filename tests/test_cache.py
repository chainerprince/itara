from uuid import UUID

from cache.lmcache import cache_scope, scoped_body, with_memory
from memory.store import Memory, Scope


def mem(n, source, owner=None):
    return Memory(UUID(int=n), owner, "summary", None, source, f"content {n}", [], 1)


SHARED = [mem(1, "b.py"), mem(2, "a.py")]
PRIVATE = [*SHARED, mem(3, "notes", owner="ann")]


def test_same_team_shares_a_key_and_private_memory_does_not():
    ann, ben, eve = Scope("web", "ann"), Scope("web", "ben"), Scope("ops", "eve")
    assert cache_scope(ann, SHARED) == cache_scope(ben, SHARED)
    assert cache_scope(ann, SHARED) != cache_scope(eve, SHARED)
    assert cache_scope(ann, PRIVATE) != cache_scope(ann, SHARED)


def test_layout_is_memory_first_and_stable_across_rank_order():
    caller = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "q"}]
    a = with_memory(caller, SHARED)
    b = with_memory(caller, list(reversed(SHARED)))
    assert a == b and len(a) == 2
    system = a[0]["content"]
    assert system.index("[a.py]") < system.index("[b.py]") < system.index("Be brief.")


def test_backends_scope_differently():
    body = {"messages": [{"role": "user", "content": "q"}]}
    vllm = scoped_body(body, Scope("web", "ann"), SHARED, "vllm")
    ollama = scoped_body(body, Scope("web", "ann"), SHARED, "ollama")
    assert vllm["cache_salt"] == cache_scope(Scope("web", "ann"), SHARED)
    assert "[scope:" not in vllm["messages"][0]["content"]
    assert "cache_salt" not in ollama
    assert ollama["messages"][0]["content"].startswith(f"[scope:{vllm['cache_salt']}]")
