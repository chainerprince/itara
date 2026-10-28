-- Memory Gateway schema: shared text memory with row-level security, plus usage accounting.
--
-- The app connects as an admin user but runs every memory query as the `gateway_app`
-- role (SET LOCAL ROLE), so row-level security always applies. Superusers and table owners
-- bypass RLS, which is why the role switch matters.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'gateway_app') THEN
        CREATE ROLE gateway_app NOLOGIN;
    END IF;
END
$$;
GRANT gateway_app TO CURRENT_USER;  -- needed on Neon, where the owner isn't a superuser

-- One row per memory. A document becomes one summary row plus raw page rows that
-- point at it (parent_id). owner_id NULL means shared with the whole team; otherwise
-- only that user can read it (personal conversation memory is private by default).
CREATE TABLE IF NOT EXISTS memories (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id     text NOT NULL,
    owner_id    text,
    kind        text NOT NULL CHECK (kind IN ('summary', 'page')),
    parent_id   uuid REFERENCES memories(id) ON DELETE CASCADE,
    source      text,
    content     text NOT NULL,
    keywords    text[] NOT NULL DEFAULT '{}',
    token_count int NOT NULL,
    embedding   vector(384),
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories (team_id, owner_id, kind);
CREATE INDEX IF NOT EXISTS memories_parent_idx ON memories (parent_id);
CREATE INDEX IF NOT EXISTS memories_tsv_idx ON memories USING gin (tsv);
CREATE INDEX IF NOT EXISTS memories_embedding_idx ON memories USING hnsw (embedding vector_cosine_ops);

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS memories_scope ON memories;
CREATE POLICY memories_scope ON memories
    USING (
        team_id = current_setting('app.team_id', true)
        AND (owner_id IS NULL OR owner_id = current_setting('app.user_id', true))
    )
    WITH CHECK (
        team_id = current_setting('app.team_id', true)
        AND (owner_id IS NULL OR owner_id = current_setting('app.user_id', true))
    );

-- Exact token counts from each model response. Not scoped: the spend cap is global.
CREATE TABLE IF NOT EXISTS usage (
    id                bigserial PRIMARY KEY,
    team_id           text NOT NULL,
    user_id           text NOT NULL,
    model             text NOT NULL,
    route             text NOT NULL,
    prompt_tokens     int NOT NULL,
    completion_tokens int NOT NULL,
    cost_usd          numeric(12, 6) NOT NULL,
    latency_ms        int NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now()
);
-- cached_tokens: prompt tokens served from the KV cache (reported by vLLM).
-- estimated: a paid response came back without usage, so the reserved worst case was recorded.
ALTER TABLE usage ADD COLUMN IF NOT EXISTS cached_tokens int NOT NULL DEFAULT 0;
ALTER TABLE usage ADD COLUMN IF NOT EXISTS estimated boolean NOT NULL DEFAULT false;

-- API keys. Identity comes only from here, never from client-supplied headers.
-- Not granted to gateway_app; only the hash of each key is stored.
CREATE TABLE IF NOT EXISTS principals (
    key_sha256 text PRIMARY KEY,
    team_id    text NOT NULL,
    user_id    text NOT NULL,
    is_admin   boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON memories TO gateway_app;
