"""Knowledge graph adapter for LightRAG-based entity extraction.

Provides per-collection knowledge graph construction that runs as a
post-processing phase in the sync pipeline. Uses LightRAG with
file-based storage (NetworkX graph + NanoVectorDB) so no additional
database services are required.
"""

from airweave.adapters.knowledge_graph.lightrag_adapter import KnowledgeGraphService

__all__ = ["KnowledgeGraphService"]
