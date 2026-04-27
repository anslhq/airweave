# RAG Improvements Plan — NexusRAG Parity & Beyond

**Branch:** `feat/local-dev-improvements`
**Status:** Implemented and deployed (local dev stack)
**Date:** April 2026

## Background

Airweave's RAG pipeline was benchmarked against NexusRAG (a competitor using LightRAG + ChromaDB + cross-encoder reranking). Initial evaluation showed NexusRAG winning 3-1 on retrieval quality. After implementing the improvements documented here, the result improved to a 2-2-1 tie, with Airweave gaining advantages in source diversity and federated search that NexusRAG lacks.

### Root Causes Identified

1. **Chunk size too large** — 4096-token chunks caused diluted relevance; most of each chunk was irrelevant padding around the actual answer.
2. **No entity deduplication** — Multiple chunks from the same document dominated results, reducing diversity.
3. **No reranking** — Vector similarity alone is a weak relevance signal; cross-encoder reranking dramatically improves precision.
4. **No knowledge graph** — Pure vector search misses entity relationships and structured knowledge that LightRAG captures.
5. **MCP response bloat** — Full document text in MCP responses wasted context window tokens for consuming AI agents.

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │     Source Connectors     │
                          │  (Google Drive, Slack...) │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │    Sync Pipeline          │
                          │  Phase 1: Extract         │
                          │  Phase 2: Transform (DAG) │
                          │  Phase 2.5: Chunk + Embed │
                          │  Phase 2.7: KG Ingest ◄───── NEW
                          │  Phase 3: Load to Vespa   │
                          └────────────┬─────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                   │
          ┌────────▼────────┐ ┌───────▼────────┐ ┌───────▼────────┐
          │   Vespa (Vector) │ │  LightRAG (KG)  │ │ Federated Srcs │
          │   Hybrid Search  │ │  Entity/Relation │ │ (Live Search)  │
          └────────┬────────┘ └───────┬────────┘ └───────┬────────┘
                    │                  │                   │
                    └──────────────────┼──────────────────┘
                                       │ Parallel fetch
                          ┌────────────▼─────────────┐
                          │    Entity Deduplication   │  ◄── NEW
                          │    Cross-Encoder Rerank   │  ◄── NEW
                          │    RRF Merge (if fed.)    │
                          └────────────┬─────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │   SearchResults + KG ctx  │
                          │   → MCP (compact 500ch)   │
                          └──────────────────────────┘
```

---

## Tier 1: Chunking & Deduplication

### Semantic Chunking Tuning

**File:** `backend/airweave/platform/chunkers/semantic.py`

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `SEMANTIC_CHUNK_SIZE` | 4096 | 512 | Smaller chunks = more precise retrieval. Each chunk is closer to a single concept. |
| `OVERLAP_TOKENS` | 128 | 64 | Reduced proportionally. 64 tokens is ~2 sentences of context bleed. |
| `SIMILARITY_THRESHOLD` | 0.01 | 0.3 | Higher threshold = more aggressive splitting at semantic boundaries. 0.01 was effectively "never split". |

The chunker uses a two-stage approach:
1. **SemanticChunker** (chonkie library) with `minishlab/potion-base-8M` embedding model detects semantic boundaries
2. **TokenChunker** fallback force-splits any oversized chunks at token boundaries (8192 hard limit for OpenAI embedding model)

### Entity Deduplication

**File:** `backend/airweave/domains/search/executor.py` — `_deduplicate_by_entity()`

When a single document produces multiple chunks (`{entity_id}__chunk_0`, `__chunk_1`, ...), only the highest-relevance chunk per parent entity is kept. This improves result diversity for RAG consumers — instead of 5 chunks from the same doc, you get 5 different documents.

```python
def _deduplicate_by_entity(results, limit):
    """Keep only the best-scoring chunk per parent entity."""
    seen: dict[str, SearchResult] = {}
    for result in results:
        orig_id = result.airweave_system_metadata.original_entity_id
        if orig_id not in seen:
            seen[orig_id] = result
        elif result.relevance_score > seen[orig_id].relevance_score:
            seen[orig_id] = result
    # ... preserve original ordering ...
```

### Compact MCP Responses

**File:** `mcp/src/utils/error-handling.ts`

MCP responses now truncate to 500 characters per result (`EXCERPT_MAX_CHARS = 500`), with word-boundary truncation and ellipsis. Full content only included when `limit <= 5`. Knowledge graph context rendered as a dedicated section before results.

---

## Tier 2: Cross-Encoder Reranking

### Implementation

**File:** `backend/airweave/adapters/reranker/cross_encoder.py`

Model: `BAAI/bge-reranker-v2-m3` — a multilingual cross-encoder that scores query-document pairs with much higher precision than embedding cosine similarity.

Key design decisions:
- **Lazy loading** — Model loaded on first use, not at startup. Avoids GPU memory allocation if reranking isn't needed.
- **Thread pool execution** — Cross-encoder inference is CPU-bound; runs via `asyncio.run_in_executor()` to avoid blocking the event loop.
- **Max length 512** — Cross-encoder input is capped at 512 tokens. This is sufficient since chunks are now 512 tokens.

### Overfetch Strategy

**File:** `backend/airweave/domains/search/executor.py`

```python
RERANK_OVERFETCH_MULTIPLIER = 3
```

When a reranker is configured, the executor fetches 3x the requested limit from Vespa. This gives the cross-encoder a wider candidate pool to score. The pipeline:

1. Vespa returns `3 * limit` results (e.g., 30 for limit=10)
2. Entity deduplication collapses chunks → ~15-20 unique documents
3. Cross-encoder scores each query-document pair
4. Top `limit` results returned, sorted by cross-encoder score

Observed cross-encoder scores: 0.994, 0.860, 0.72 (vs. Vespa's cosine similarity: 0.85, 0.83, 0.82 — much flatter distribution).

### Infrastructure

**File:** `docker/Dockerfile.backend-local`

```dockerfile
FROM ghcr.io/airweave-ai/airweave-backend:latest
RUN pip install --no-cache-dir sentence-transformers==4.1.0
```

The pre-built backend image doesn't include sentence-transformers (large dependency). The local dev Dockerfile extends it with the reranker dependency. The worker image does NOT need this — reranking happens at search time, not sync time.

---

## Tier 3: Knowledge Graph (LightRAG)

### Architecture Decision: Dedicated Container

We evaluated two approaches:
1. **Embedded** — Import LightRAG as a Python library into the backend/worker
2. **Dedicated container** — Run LightRAG as a separate Docker service with REST API

**Chose (2)** because:
- LightRAG's `ainsert()` queues work internally; cleanup() before extraction completes loses data
- Embedded approach requires the same process to stay alive for the full extraction pipeline
- Dedicated container handles its own lifecycle, retries, and caching
- Per-collection isolation via `LIGHTRAG-WORKSPACE` HTTP header scales to N collections without N containers
- LLM/embedding provider configuration lives in `lightrag.env`, independent of Airweave's config

### LightRAG Container

**Docker Compose service:** `docker/docker-compose.yml`

```yaml
lightrag:
  container_name: airweave-lightrag
  image: ghcr.io/hkuds/lightrag:latest
  ports:
    - "9621:9621"
  env_file:
    - lightrag.env
  volumes:
    - lightrag_data:/app/data/rag_storage
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

### LightRAG Configuration

**File:** `docker/lightrag.env`

```
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:8317/v1   # or SGLang at 192.168.1.106:30000/v1
LLM_MODEL=gpt-5.2                                       # or QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
EMBEDDING_BINDING=gemini
EMBEDDING_MODEL=gemini-embedding-2-preview
EMBEDDING_DIM=3072
MAX_ASYNC=16
MAX_PARALLEL_INSERT=8
```

LLM options tested and benchmarked:

| Model | Quality Score | Rate Limit | Notes |
|-------|--------------|------------|-------|
| qwen/qwen3.6-plus-preview:free (OpenRouter) | 100/100 | 1000 req/day ($10 credits) | Best free option |
| gpt-5.2 (via cliproxy) | High | Abundant user quota | Fastest cloud option |
| Qwen3-Coder-30B-A3B (SGLang local) | Good | Unlimited | ~3 docs/min, GPU-bound |
| gpt-oss-120b (OpenRouter) | 85/100 | Free tier | Good but slower |

### HTTP Client Adapter

**File:** `backend/airweave/adapters/knowledge_graph/lightrag_adapter.py`

```python
LIGHTRAG_BASE_URL = "http://lightrag:9621"
LIGHTRAG_INGEST_TIMEOUT = 300.0   # 5 min for large docs
LIGHTRAG_QUERY_TIMEOUT = 10.0     # Fast fail for search queries

class KnowledgeGraphService:
    def _headers(self):
        return {"LIGHTRAG-WORKSPACE": self.collection_readable_id}
```

Per-collection isolation: Every HTTP request includes `LIGHTRAG-WORKSPACE` header. LightRAG creates separate graph storage, vector indices, and KV stores per workspace. This scales to any number of collections without running separate containers.

Split timeouts:
- **Ingestion: 300s** — Large documents can take minutes to extract entities/relationships
- **Query: 10s** — Search queries should fail fast; KG context is supplementary, not critical

### Sync Pipeline Integration

**File:** `backend/airweave/domains/sync_pipeline/orchestrator.py`

New **Phase 2.7** added between chunk+embed and vector DB load:

```
Phase 1: Extract entities from source
Phase 2: Transform via DAG
Phase 2.5: Chunk + Embed
Phase 2.7: KG Ingestion  ← NEW
Phase 3: Load to Vespa
```

Entity text is collected during chunk+embed (`runtime.kg_texts`), then batch-ingested into LightRAG. Fault-tolerant: KG failure never blocks the sync pipeline.

**File:** `backend/airweave/domains/sync_pipeline/processors/chunk_embed.py:65-68`

```python
# Step 2.5: Collect parent entity texts for KG ingestion
for entity in processed:
    if entity.textual_representation:
        runtime.kg_texts.append(entity.textual_representation)
```

### Search-Time KG Integration

**File:** `backend/airweave/domains/search/executor.py:140-162`

KG query runs in parallel with Vespa vector search via `asyncio.create_task()`:

```python
vector_task = asyncio.create_task(self._execute_vector_search(...))
kg_task = asyncio.create_task(_query_knowledge_graph(query, collection_readable_id))

vector_results = await vector_task
kg_context = await kg_task
```

KG context is returned as `SearchResults.knowledge_graph_context` — a plain text string with entity/relationship context. Graceful degradation: if KG is unavailable (import error, timeout, or empty), returns empty string.

### KG Extraction Stats (as of April 2026)

- **Total documents:** 616
- **Successfully processed:** 537 (87%)
- **Persistently failed:** 79 → Resolved by bumping SGLang context from 8192 to 16384
- **Graph size:** 13,303 nodes, 23,353 edges
- **Average extraction:** ~12 entities + 10 relationships per chunk
- **Processing rate:** ~3 docs/min on RTX 5090 via SGLang

---

## Supporting Infrastructure

### OpenAI-Compatible LLM Adapter

**File:** `backend/airweave/adapters/llm/openai_compat.py`

Generic adapter that works with any OpenAI-compatible endpoint (local LLMs, vLLM, SGLang, Ollama, etc.):
- Uses `TOGETHER_BASE_URL` and `TOGETHER_API_KEY` settings (reuses existing config)
- Strips markdown code fences from JSON responses (common with smaller models)
- Handles `reasoning_content` field for models that separate reasoning from output
- `response_format={"type": "json_object"}` for structured output

### SGLang Local Inference Server

**Location:** `~/sglang/docker-compose.yml` on `wade@xmen` (192.168.1.106)

```yaml
command: >
  python3 -m sglang.launch_server
    --model-path QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ
    --quantization awq_marlin    # NOT awq — awq causes OOM on 32GB
    --dtype float16
    --mem-fraction-static 0.82
    --context-length 16384       # Bumped from 8192 to handle large docs
```

Hardware: NVIDIA RTX 5090 (32GB VRAM)
- Model: Qwen3-Coder-30B-A3B (MoE, 30B total / 3B active params)
- Quantization: AWQ_MARLIN (15.78 GB weights, loads directly in quantized format)
- KV Cache: ~4.5 GB (97K tokens)
- GPU utilization: 100% during batch extraction

**Critical lesson:** `--quantization awq` causes OOM because it loads weights in fp16 first then converts. `awq_marlin` loads directly in quantized format, saving ~15GB.

### LLM Model Evaluation Script

**File:** `scripts/eval_llm_models.py`

PEP 723 inline-deps script for comparing LLM models across four tasks:
1. **Search strategy planning** — JSON schema adherence for search plans
2. **KG entity extraction** — Entity/relationship extraction quality
3. **Query expansion** — Synonym and reformulation generation
4. **Factual accuracy** — Simple knowledge questions

Supports OpenRouter (free tier models) and local proxy endpoints. Scores on: JSON validity, schema completeness, factual accuracy, and reliability (retry success rate).

### RustFS (S3-Compatible Storage)

Added to Docker Compose for local object storage (replacing AWS S3 dependency):
- Persistent volume for data
- Init container creates default bucket
- Internal Docker network only (no exposed ports)

### CORS Fix

**File:** `backend/airweave/api/middleware.py`

Fixed `DynamicCORSMiddleware`: when `CORS_ORIGINS=*`, the middleware checked `origin in self.default_origins` which failed because `"*"` doesn't match a specific origin string. Added `"*" in self.default_origins` check for wildcard CORS.

### force_full_sync Fix

**File:** `backend/airweave/domains/sync_pipeline/factory.py`

`force_full_sync` wasn't actually re-inserting entities because `skip_hash_comparison` wasn't being set. Fixed: `resolved_config.behavior.skip_hash_comparison = True` when force_full_sync is requested.

---

## Commit History

```
ec9bdf5a feat: local dev stack improvements — RustFS, MCP server, OpenAI-compat LLM, CORS fix
817f8c89 feat: RAG precision improvements — smaller chunks, entity dedup, compact MCP responses
0c84add9 feat: Tier 3 — cross-encoder reranking, LightRAG knowledge graph, KG-enhanced search
3c515913 fix: Tier 3 KG — add query method, share volume, fix force_full_sync hash bypass
89e57761 feat: dedicated LightRAG container, Gemini models, KG via HTTP API
dc9d90aa fix: KG workspace isolation, gemini-2.5-flash fallback, query timeout
ca4b61c5 feat: LLM model evaluation script with OpenRouter presets
```

---

## Evaluation Results

### Before Improvements (Baseline)

NexusRAG won 3-1 on retrieval quality. Airweave results were diluted by large chunks, lacked reranking, and had no knowledge graph.

### After Tier 1+2 (Chunks + Reranking)

Improved to competitive but NexusRAG still had edge from KG context.

### After Tier 3 (Full Stack)

Result: **2-2-1 tie** (Airweave vs NexusRAG).

Airweave advantages:
- Federated live search (NexusRAG doesn't have this)
- Source diversity from entity deduplication
- 50+ source connectors vs NexusRAG's manual upload

NexusRAG advantages:
- Simpler stack (single ChromaDB vs Vespa)
- Faster cold start (no Temporal/Redis/Postgres overhead)

---

## Configuration Reference

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TOGETHER_BASE_URL` | OpenAI-compat LLM endpoint for search planning | `http://host.docker.internal:8317/v1` |
| `TOGETHER_API_KEY` | API key for the LLM endpoint | `airweave` |

### LightRAG Environment (`docker/lightrag.env`)

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_BINDING` | LLM provider type | `openai` |
| `LLM_BINDING_HOST` | LLM API endpoint | `http://host.docker.internal:8317/v1` |
| `LLM_MODEL` | Model name | `gpt-5.2` |
| `EMBEDDING_BINDING` | Embedding provider | `gemini` |
| `EMBEDDING_MODEL` | Embedding model | `gemini-embedding-2-preview` |
| `EMBEDDING_DIM` | Embedding dimensions | `3072` |
| `MAX_ASYNC` | Max concurrent LLM calls | `16` |
| `MAX_PARALLEL_INSERT` | Max parallel doc insertions | `8` |

### Files Changed

| File | Change |
|------|--------|
| `backend/airweave/platform/chunkers/semantic.py` | Chunk size 4096→512, overlap 128→64, threshold 0.01→0.3 |
| `backend/airweave/domains/search/executor.py` | Entity dedup, reranker integration, parallel KG query |
| `backend/airweave/adapters/reranker/cross_encoder.py` | NEW — BAAI/bge-reranker-v2-m3 cross-encoder |
| `backend/airweave/adapters/reranker/types.py` | NEW — RerankerResult type |
| `backend/airweave/adapters/reranker/exceptions.py` | NEW — RerankerError |
| `backend/airweave/core/protocols/reranker.py` | NEW — RerankerProtocol |
| `backend/airweave/adapters/knowledge_graph/lightrag_adapter.py` | NEW — LightRAG HTTP client |
| `backend/airweave/adapters/llm/openai_compat.py` | NEW — OpenAI-compatible LLM adapter |
| `backend/airweave/domains/sync_pipeline/orchestrator.py` | Phase 2.7 KG ingestion |
| `backend/airweave/domains/sync_pipeline/processors/chunk_embed.py` | Collect kg_texts |
| `backend/airweave/domains/sync_pipeline/contexts/runtime.py` | kg_texts field |
| `backend/airweave/domains/sync_pipeline/factory.py` | force_full_sync fix |
| `backend/airweave/api/middleware.py` | CORS wildcard fix |
| `docker/docker-compose.yml` | LightRAG container, RustFS, MCP server |
| `docker/Dockerfile.backend-local` | NEW — extends backend with sentence-transformers |
| `docker/lightrag.env` | NEW — LightRAG configuration |
| `mcp/src/utils/error-handling.ts` | 500-char excerpts, KG section rendering |
| `scripts/eval_llm_models.py` | NEW — LLM model evaluation script |
