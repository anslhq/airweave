# PRD: Memvid Integration into Airweave

**Branch:** `feat/memvid-multimodal`
**Status:** Draft
**Author:** Harsha
**Date:** April 2026

---

## 1. Problem Statement

Airweave has zero capability for audio and video content. When a Google Drive sync encounters an `.mp4`, `.mp3`, or `.wav` file, it's skipped entirely. YouTube transcripts must be pre-processed externally and uploaded as markdown. There is no path from raw media to searchable, indexed content.

Meanwhile, the Airweave fork already has a production-quality RAG pipeline for documents (512-token semantic chunking, BM25+vector hybrid search, LightRAG knowledge graph, bge-reranker-v2-m3 cross-encoder, entity deduplication). **The only gap is media input.**

Additionally, there is no way to export a collection as a portable, offline-queryable artifact. Users who need to query knowledge bases without infrastructure (offline, edge, shipping to clients) have no option.

## 2. Solution

Integrate memvid as a **dedicated sidecar container** that provides two capabilities:

1. **Media transcription** — Audio/video files → timestamped text, returned to Airweave's existing pipeline for chunking, embedding, KG extraction, and Vespa indexing.
2. **Portable export** — Dump a processed Airweave collection into a `.mv2` file for offline querying.

Memvid does NOT replace any existing Airweave component. Documents (PDF, DOCX, MD, etc.) continue through the existing pipeline unchanged.

```
Source file arrives via connector
    │
    ├── Document (.md/.pdf/.docx/.pptx)
    │   └── Airweave pipeline directly (unchanged)
    │
    ├── Audio (.mp3/.wav/.flac/.aac/.ogg)
    │   └── POST to memvid:9622/transcribe
    │       └── Timestamped transcript text returned
    │           └── Enters Airweave pipeline as text entity
    │               └── Chunk → Embed → KG → Vespa
    │
    └── Video (.mp4/.mov/.webm/.mkv/.avi)
        └── POST to memvid:9622/transcribe
            └── Audio track extracted → Whisper transcription
                └── Timestamped transcript text returned
                    └── Enters Airweave pipeline as text entity
                        └── Chunk → Embed → KG → Vespa
```

## 3. Architecture

### 3.1 Container Topology

```yaml
# docker/docker-compose.yml (addition)
memvid:
  container_name: airweave-memvid
  build:
    context: ./memvid-service
    dockerfile: Dockerfile
  ports:
    - "9622:9622"
  volumes:
    - memvid_data:/data/mv2
    - memvid_models:/root/.cache/memvid
  environment:
    - MEMVID_HOST=0.0.0.0
    - MEMVID_PORT=9622
    - WHISPER_MODEL=whisper-small-en
    - DEFAULT_EMBEDDING_MODEL=bge-small
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: "4.0"
      reservations:
        memory: 2G
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:9622/health"]
    interval: 10s
    timeout: 5s
    retries: 5

volumes:
  memvid_data:
  memvid_models:
```

**Why dedicated container (not embedded in worker):**
- Whisper model is 244MB — loading per-worker wastes RAM
- Whisper inference is CPU/GPU heavy — competes with chunking/embedding
- Dedicated container keeps model warm between transcriptions
- Same pattern as LightRAG container (proven, already in the stack)
- Single-writer `.mv2` file lock requires centralized access

### 3.2 Memvid Service (New — ~150 lines)

**Location:** `docker/memvid-service/`

```
docker/memvid-service/
├── Dockerfile
├── server.py            # FastAPI wrapper around memvid-sdk
└── requirements.txt     # memvid-sdk, fastapi, uvicorn
```

**Dockerfile:**
```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN mkdir -p /data/mv2

EXPOSE 9622
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9622"]
```

**requirements.txt:**
```
memvid-sdk>=2.0.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
```

**server.py — HTTP API:**

```python
"""Memvid sidecar service for Airweave.

Provides:
  POST /transcribe         — Audio/video → timestamped transcript
  POST /export             — Collection chunks → .mv2 file
  GET  /search/{collection} — Search a .mv2 file
  GET  /health             — Health check
"""

from fastapi import FastAPI, UploadFile, HTTPException, Header
from pydantic import BaseModel
from memvid import Memvid, PutOptions, SearchRequest, WhisperTranscriber, WhisperConfig
from pathlib import Path
import tempfile, os, json

app = FastAPI(title="Airweave Memvid Service")

# Lazy-loaded Whisper transcriber (keeps model warm)
_transcriber = None

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        model = os.environ.get("WHISPER_MODEL", "whisper-small-en")
        _transcriber = WhisperTranscriber(WhisperConfig(model_name=model))
    return _transcriber


# ── Transcription ──────────────────────────────────────────────

class TranscriptSegment(BaseModel):
    start: float       # seconds
    end: float         # seconds
    text: str

class TranscribeResponse(BaseModel):
    segments: list[TranscriptSegment]
    full_text: str
    duration_seconds: float
    language: str

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile):
    """Transcribe audio/video file. Returns timestamped segments.
    
    Accepts: mp3, wav, flac, ogg, aac, mp4, mov, webm, mkv, avi
    Returns: List of {start, end, text} segments + concatenated full_text
    """
    suffix = Path(file.filename or "audio.mp3").suffix
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        transcriber = get_transcriber()
        result = transcriber.transcribe_file(tmp_path)
        
        segments = [
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip()
            )
            for seg in result.segments
            if seg.text.strip()
        ]
        
        full_text = "\n".join(
            f"[{_fmt_time(s.start)}] {s.text}" for s in segments
        )
        
        return TranscribeResponse(
            segments=segments,
            full_text=full_text,
            duration_seconds=result.duration,
            language=result.language or "en"
        )
    finally:
        os.unlink(tmp_path)


# ── Export ─────────────────────────────────────────────────────

class ExportChunk(BaseModel):
    text: str
    uri: str | None = None
    title: str | None = None
    tags: dict[str, str] | None = None

class ExportRequest(BaseModel):
    collection_id: str
    chunks: list[ExportChunk]

class ExportResponse(BaseModel):
    mv2_path: str
    frame_count: int
    file_size_bytes: int

@app.post("/export", response_model=ExportResponse)
async def export_collection(request: ExportRequest):
    """Export chunks to a portable .mv2 file."""
    mv2_path = f"/data/mv2/{request.collection_id}.mv2"
    
    mem = Memvid.create(mv2_path)
    
    for chunk in request.chunks:
        opts = PutOptions()
        if chunk.uri:
            opts.uri = chunk.uri
        if chunk.title:
            opts.title = chunk.title
        if chunk.tags:
            for k, v in chunk.tags.items():
                opts.tag(k, v)
        mem.put_bytes(chunk.text.encode(), opts)
    
    mem.commit()
    stats = mem.stats()
    
    return ExportResponse(
        mv2_path=mv2_path,
        frame_count=stats.frame_count,
        file_size_bytes=os.path.getsize(mv2_path)
    )


# ── Search ─────────────────────────────────────────────────────

class SearchResult(BaseModel):
    frame_id: int
    text: str
    score: float | None
    uri: str | None
    title: str | None

class SearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchResult]
    elapsed_ms: float

@app.get("/search/{collection_id}", response_model=SearchResponse)
async def search_collection(
    collection_id: str,
    query: str,
    top_k: int = 10
):
    """Search a .mv2 file. For offline/portable MCP use."""
    mv2_path = f"/data/mv2/{collection_id}.mv2"
    
    if not os.path.exists(mv2_path):
        raise HTTPException(404, f"Collection {collection_id} not found")
    
    mem = Memvid.open_read_only(mv2_path)
    
    response = mem.search(SearchRequest(
        query=query,
        top_k=top_k,
        snippet_chars=300
    ))
    
    return SearchResponse(
        query=query,
        total_hits=response.total_hits,
        hits=[
            SearchResult(
                frame_id=hit.frame_id,
                text=hit.text,
                score=hit.score,
                uri=hit.uri if hit.uri else None,
                title=hit.title if hit.title else None
            )
            for hit in response.hits
        ],
        elapsed_ms=response.elapsed_ms
    )


# ── Health ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "whisper_model": os.environ.get("WHISPER_MODEL", "whisper-small-en")}


# ── Helpers ────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
```

### 3.3 Airweave HTTP Client Adapter

**File:** `backend/airweave/adapters/transcription/memvid_adapter.py`

Follows the exact same pattern as `adapters/knowledge_graph/lightrag_adapter.py`:

```python
"""Memvid transcription service adapter.

Mirrors the LightRAG adapter pattern:
- Lazy HTTP client initialization
- Per-collection workspace isolation
- Split timeouts (transcription: 300s, search: 10s)
- Fault-tolerant: transcription failure never blocks sync
"""

from __future__ import annotations
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

MEMVID_BASE_URL = "http://memvid:9622"
MEMVID_TRANSCRIBE_TIMEOUT = 300.0   # 5 min for large media files
MEMVID_SEARCH_TIMEOUT = 10.0        # Fast fail for search queries
MEMVID_EXPORT_TIMEOUT = 600.0       # 10 min for large collections

# MIME types that memvid can transcribe
TRANSCRIBABLE_MIMES = frozenset({
    # Audio
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/flac", "audio/ogg", "audio/aac", "audio/mp4",
    # Video
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-matroska", "video/x-msvideo", "video/avi",
})

TRANSCRIBABLE_EXTENSIONS = frozenset({
    ".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
})


def is_transcribable(mime_type: str | None = None, filename: str | None = None) -> bool:
    """Check if a file can be transcribed by memvid."""
    if mime_type and mime_type.lower() in TRANSCRIBABLE_MIMES:
        return True
    if filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in TRANSCRIBABLE_EXTENSIONS
    return False


class MemvidTranscriptionService:
    """HTTP client for the memvid sidecar container."""

    def __init__(self, collection_readable_id: str):
        self.collection_readable_id = collection_readable_id
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=MEMVID_BASE_URL,
                timeout=httpx.Timeout(MEMVID_TRANSCRIBE_TIMEOUT),
            )
        return self._client

    async def transcribe(self, file_bytes: bytes, filename: str) -> Optional[str]:
        """Transcribe audio/video bytes. Returns timestamped text or None on failure."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/transcribe",
                files={"file": (filename, file_bytes)},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("full_text")
        except Exception as e:
            logger.warning(
                f"[Memvid] Transcription failed for {filename}: {e}. "
                "Entity will be skipped (non-fatal).",
                exc_info=True,
            )
            return None

    async def export_collection(
        self, chunks: list[dict]
    ) -> Optional[dict]:
        """Export chunks to a .mv2 file. Returns {mv2_path, frame_count, file_size_bytes}."""
        try:
            client = await self._get_client()
            client.timeout = httpx.Timeout(MEMVID_EXPORT_TIMEOUT)
            response = await client.post(
                "/export",
                json={
                    "collection_id": self.collection_readable_id,
                    "chunks": chunks,
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[Memvid] Export failed: {e}", exc_info=True)
            return None

    async def search(self, query: str, top_k: int = 10) -> Optional[dict]:
        """Search a .mv2 file. For offline MCP use."""
        try:
            client = await self._get_client()
            client.timeout = httpx.Timeout(MEMVID_SEARCH_TIMEOUT)
            response = await client.get(
                f"/search/{self.collection_readable_id}",
                params={"query": query, "top_k": top_k},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[Memvid] Search failed: {e}", exc_info=True)
            return None

    async def cleanup(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
```

### 3.4 Sync Pipeline Integration

**Where it plugs in:** Between entity extraction and chunk+embed, inside the entity processing pipeline.

**File:** `backend/airweave/domains/sync_pipeline/processors/chunk_embed.py`

Add media detection + transcription before chunking:

```python
# At the top of ChunkEmbedProcessor.process(), before chunking:

async def process(self, entities, sync_context, runtime):
    # ── NEW: Transcribe media entities ──────────────────────
    if runtime.memvid_service:
        for entity in entities:
            mime = getattr(entity, 'mime_type', '') or ''
            name = getattr(entity, 'name', '') or ''
            
            if is_transcribable(mime_type=mime, filename=name):
                # Get the raw file bytes from storage
                file_bytes = await self._fetch_entity_bytes(entity, runtime)
                if file_bytes:
                    transcript = await runtime.memvid_service.transcribe(
                        file_bytes, name
                    )
                    if transcript:
                        # Replace entity's textual representation with transcript
                        entity.textual_representation = transcript
                        sync_context.logger.info(
                            f"[Memvid] Transcribed {name} "
                            f"({len(transcript)} chars)"
                        )
                    else:
                        sync_context.logger.debug(
                            f"[Memvid] Skipping {name} (transcription failed)"
                        )
                        continue
    
    # ── Existing pipeline continues unchanged ──────────────
    # Step 1: Build textual representations
    # Step 2: Collect KG texts
    # Step 3: Chunk
    # Step 4: Embed
    # ...
```

**File:** `backend/airweave/domains/sync_pipeline/contexts/runtime.py`

Add memvid service to runtime:

```python
@dataclass
class SyncRuntime:
    source: BaseSource
    entity_tracker: EntityTracker
    # ... existing fields ...
    kg_texts: list[str] = field(default_factory=list)
    memvid_service: Optional[MemvidTranscriptionService] = None  # NEW
```

**File:** `backend/airweave/domains/sync_pipeline/orchestrator.py`

Initialize memvid service in pipeline setup:

```python
# In _build_runtime() or pipeline initialization:
try:
    from airweave.adapters.transcription.memvid_adapter import MemvidTranscriptionService
    # Health check
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://memvid:9622/health", timeout=2.0)
        resp.raise_for_status()
    runtime.memvid_service = MemvidTranscriptionService(collection_readable_id)
    logger.info("[Memvid] Transcription service available")
except Exception:
    runtime.memvid_service = None
    logger.debug("[Memvid] Transcription service unavailable, media files will be skipped")
```

### 3.5 Export API Endpoint

**File:** `backend/airweave/api/v1/endpoints/collections.py`

Add an export endpoint:

```python
@router.post("/{collection_id}/export/memvid")
async def export_to_memvid(
    collection_id: UUID,
    ctx: ApiContext = Depends(get_api_context),
):
    """Export collection to a portable .mv2 file."""
    # 1. Fetch all chunks from Vespa for this collection
    chunks = await vespa_client.get_all_chunks(collection_id)
    
    # 2. Format for memvid
    export_chunks = [
        {
            "text": chunk.textual_representation,
            "uri": chunk.entity_id,
            "title": chunk.name,
            "tags": {"source": chunk.source_name},
        }
        for chunk in chunks
    ]
    
    # 3. Send to memvid service
    memvid = MemvidTranscriptionService(collection_readable_id)
    result = await memvid.export_collection(export_chunks)
    
    if not result:
        raise HTTPException(500, "Export failed")
    
    return {
        "mv2_path": result["mv2_path"],
        "frame_count": result["frame_count"],
        "file_size_bytes": result["file_size_bytes"],
        "message": f"Exported {result['frame_count']} frames to {result['mv2_path']}"
    }
```

## 4. Data Flow Diagrams

### 4.1 Media Transcription Flow

```
Google Drive sync encounters "meeting-recording.mp4"
    │
    ▼
Source connector yields FileEntity(
    name="meeting-recording.mp4",
    mime_type="video/mp4",
    entity_id="gdrive:abc123"
)
    │
    ▼
ChunkEmbedProcessor.process()
    │
    ├── is_transcribable("video/mp4") → True
    │
    ├── Fetch file bytes from storage
    │
    ├── POST http://memvid:9622/transcribe
    │   Body: multipart file upload
    │   Timeout: 300s
    │       │
    │       ▼
    │   Memvid container:
    │   ├── ffmpeg extracts audio track
    │   ├── Whisper transcribes (whisper-small-en, 244MB model)
    │   └── Returns: {segments: [{start, end, text}...], full_text: "..."}
    │
    ├── entity.textual_representation = full_text
    │   (formatted as "[00:00] text\n[00:15] text\n...")
    │
    ├── runtime.kg_texts.append(full_text)  # KG extraction
    │
    ▼
Existing pipeline continues:
    ├── Semantic chunking (512 tokens)
    ├── Dense + sparse embedding
    ├── Vespa indexing
    └── LightRAG KG ingestion (Phase 2.7)
```

### 4.2 Portable Export Flow

```
User clicks "Export to Memvid" in UI
    │
    ▼
POST /collections/{id}/export/memvid
    │
    ▼
Backend fetches all chunks from Vespa
    │
    ▼
POST http://memvid:9622/export
    Body: {collection_id, chunks: [{text, uri, title, tags}...]}
    Timeout: 600s
    │
    ▼
Memvid container:
    ├── Creates /data/mv2/{collection_id}.mv2
    ├── Writes each chunk as a frame with metadata
    ├── Commits (builds Tantivy + HNSW indices)
    └── Returns: {mv2_path, frame_count, file_size_bytes}
    │
    ▼
.mv2 file available on memvid_data volume
    ├── Queryable via GET /search/{collection_id}
    ├── Downloadable for offline use
    └── Can be served via standalone MCP server
```

## 5. File Manifest

### New Files

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `docker/memvid-service/Dockerfile` | 15 | Container image with memvid-sdk + ffmpeg |
| `docker/memvid-service/server.py` | 150 | FastAPI wrapper: /transcribe, /export, /search, /health |
| `docker/memvid-service/requirements.txt` | 4 | memvid-sdk, fastapi, uvicorn, python-multipart |
| `backend/airweave/adapters/transcription/__init__.py` | 1 | Package init |
| `backend/airweave/adapters/transcription/memvid_adapter.py` | 120 | HTTP client adapter (mirrors lightrag_adapter pattern) |

### Modified Files

| File | Change | Lines changed (est.) |
|------|--------|---------------------|
| `docker/docker-compose.yml` | Add memvid service + volumes | +25 |
| `backend/airweave/domains/sync_pipeline/contexts/runtime.py` | Add `memvid_service` field | +3 |
| `backend/airweave/domains/sync_pipeline/processors/chunk_embed.py` | Add media detection + transcription before chunking | +30 |
| `backend/airweave/domains/sync_pipeline/orchestrator.py` | Initialize memvid service with health check | +15 |
| `backend/airweave/api/v1/endpoints/collections.py` | Add `/export/memvid` endpoint | +40 |

### Total: ~400 lines of new code + ~115 lines of modifications

## 6. Configuration

### Environment Variables

| Variable | Container | Default | Purpose |
|----------|-----------|---------|---------|
| `WHISPER_MODEL` | memvid | `whisper-small-en` | Whisper model for transcription |
| `MEMVID_HOST` | memvid | `0.0.0.0` | Listen address |
| `MEMVID_PORT` | memvid | `9622` | Listen port |
| `DEFAULT_EMBEDDING_MODEL` | memvid | `bge-small` | For .mv2 vector search (export only) |

### Resource Limits

| Container | Memory Limit | CPU Limit | Rationale |
|-----------|-------------|-----------|-----------|
| memvid | 4GB | 4.0 | Whisper model (244MB) + inference headroom |
| memvid (reservation) | 2GB | — | Model always in memory |

### First-Run Downloads

| Model | Size | Cached at |
|-------|------|-----------|
| whisper-small-en | 244MB | `/root/.cache/memvid/` (Docker volume `memvid_models`) |
| bge-small (optional, for export) | 120MB | Same cache directory |

## 7. Fault Tolerance

Follows the LightRAG precedent — **memvid failure never blocks a sync.**

| Failure | Behavior |
|---------|----------|
| Memvid container down | `runtime.memvid_service = None`, media files skipped silently |
| Transcription timeout (>300s) | Warning logged, entity skipped, sync continues |
| Transcription returns empty text | Entity skipped (no textual representation to chunk) |
| Export fails | HTTP 500 returned to caller, no side effects |
| Whisper model download fails | Container health check fails, service marked unavailable |
| .mv2 file lock contention | Export retries with backoff (single-writer constraint) |

## 8. Supported Media Formats

### Audio (via ffmpeg + Whisper)

| Format | Extension | MIME Type |
|--------|-----------|-----------|
| MP3 | .mp3 | audio/mpeg |
| WAV | .wav | audio/wav, audio/x-wav |
| FLAC | .flac | audio/flac |
| OGG/Vorbis | .ogg | audio/ogg |
| AAC | .aac, .m4a | audio/aac, audio/mp4 |

### Video (audio track extracted via ffmpeg, then Whisper)

| Format | Extension | MIME Type |
|--------|-----------|-----------|
| MP4 | .mp4 | video/mp4 |
| QuickTime | .mov | video/quicktime |
| WebM | .webm | video/webm |
| Matroska | .mkv | video/x-matroska |
| AVI | .avi | video/x-msvideo |

## 9. Testing Plan

### Unit Tests

| Test | Validates |
|------|-----------|
| `test_is_transcribable` | MIME type and extension detection |
| `test_memvid_adapter_transcribe` | HTTP client sends correct request |
| `test_memvid_adapter_fault_tolerance` | Returns None on failure, doesn't raise |
| `test_chunk_embed_media_detection` | Media entities routed to transcription |
| `test_chunk_embed_text_passthrough` | Non-media entities skip transcription |

### Integration Tests

| Test | Validates |
|------|-----------|
| `test_transcribe_mp3` | End-to-end: upload mp3 → get timestamped text |
| `test_transcribe_mp4` | End-to-end: upload mp4 → ffmpeg extract → Whisper → text |
| `test_export_and_search` | Export chunks → .mv2 → search returns results |
| `test_sync_with_media` | Google Drive sync with mixed files (docs + audio) processes all |

### Manual Verification

1. Upload a 5-minute MP3 to Google Drive
2. Trigger sync
3. Verify transcript appears in Airweave search results with timestamps
4. Verify KG entities extracted from transcript
5. Export collection to .mv2
6. Search .mv2 via /search endpoint

## 10. Future Extensions (Not In This PRD)

| Extension | Description | Effort |
|-----------|-------------|--------|
| **Keyframe OCR** | Extract text from video frames (slides, UI) via PySceneDetect + PaddleOCR | Medium |
| **CLIP visual search** | Image similarity search via CLIP embeddings | Medium |
| **Speaker diarization** | Who said what (WhisperX upgrade from base Whisper) | Low |
| **YouTube connector** | Native yt-dlp source connector for Airweave | Low |
| **MCP for .mv2** | Standalone MCP server wrapping /search for offline Claude use | Low |
| **Collection snapshots** | Auto-export .mv2 on sync completion (cron) | Low |

## 11. Dependencies

| Dependency | Version | License | Purpose |
|------------|---------|---------|---------|
| `memvid-sdk` | >=2.0.0 | Apache 2.0 | Core library (Whisper, search, .mv2 format) |
| `fastapi` | >=0.104.0 | MIT | HTTP server for sidecar |
| `uvicorn` | >=0.24.0 | BSD | ASGI server |
| `python-multipart` | >=0.0.6 | Apache 2.0 | File upload parsing |
| `ffmpeg` | system | LGPL/GPL | Audio extraction from video |
| `httpx` | (already in Airweave) | BSD | Async HTTP client for adapter |

## 12. Rollout Plan

1. **Build and test memvid service container locally** — verify transcription works on sample files
2. **Add adapter + pipeline integration** — wire into ChunkEmbedProcessor
3. **Test with Google Drive sync** — upload mixed media + docs, verify all indexed
4. **Add export endpoint** — test .mv2 creation and search
5. **Add to docker-compose.yml** — deploy alongside existing stack
6. **Document in rag-improvements.md** — update the improvements doc with Tier 4
