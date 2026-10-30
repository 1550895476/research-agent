CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS research_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    value_score DOUBLE PRECISION NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    compressed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_memories_user_type
    ON research_memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_value
    ON research_memories(user_id, value_score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON research_memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS episodic_memories (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    stage TEXT NOT NULL,
    task_id TEXT,
    decision TEXT,
    failure_reason TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_user_time
    ON episodic_memories(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_run_time
    ON episodic_memories(run_id, occurred_at);

CREATE TABLE IF NOT EXISTS blackboard_results (
    result_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    payload_type TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blackboard_task
    ON blackboard_results(task_id, created_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    dense_embedding vector(1024) NOT NULL,
    sparse_embedding JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', COALESCE(title, '') || ' ' || content)
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON knowledge_chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_dense
    ON knowledge_chunks USING hnsw (dense_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_sparse
    ON knowledge_chunks USING gin (sparse_embedding);
CREATE INDEX IF NOT EXISTS idx_chunks_search
    ON knowledge_chunks USING gin (search_vector);

CREATE TABLE IF NOT EXISTS saga_journal (
    saga_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (saga_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_saga_run
    ON saga_journal(run_id, updated_at DESC);
