"""
Search tools for the agentic RAG system.

Implements:
- search_text: Keyword-based search with LIKE queries
- search_semantic: Vector similarity search
- get_document_metadata: Retrieve document metadata

These tools are called by the agent during the evidence collection loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..extraction.schemas import ExtractionSchema, EXTRACTION_SCHEMAS
from ..extraction.persistence import MetadataStore, query_chunks_with_metadata_filter

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    total_found: int = 0


async def search_text(
    bucket: str,
    query: str,
    document_store: Any,
    settings: Any,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    context_chars: int = 400,
    doc_id: Optional[str] = None,
) -> ToolResult:
    """
    Keyword/phrase search within a bucket using LIKE queries.
    
    Args:
        bucket: Target bucket (L1 category ID, e.g., "financial_accounting")
        query: Keyword search string
        document_store: Database access object
        filters: Optional metadata filters (for extracted metadata)
        top_k: Number of results to return
        context_chars: Characters to include in snippet
        doc_id: Optional - scope search to specific document
    
    Returns:
        ToolResult with matching chunks
    """
    try:
        schema = _match_schema_for_bucket(bucket)
        matching_doc_hashes, doc_info_map = await _collect_bucket_documents(
            document_store=document_store,
            bucket=bucket,
            schema=schema,
            doc_id=doc_id,
        )
        
        matching_doc_hashes_set = set(matching_doc_hashes)
        
        if not matching_doc_hashes:
            return ToolResult(
                tool_name="search_text",
                success=True,
                results=[],
                total_found=0,
            )
        
        # If we have metadata filters and a schema, leverage the extraction tables directly
        if schema and filters:
            metadata_rows = await query_chunks_with_metadata_filter(
                db_path=str(settings.doc_store_path),
                schema=schema,
                metadata_filters=filters,
                text_search=query or None,
                limit=max(top_k, 50),
            )
            
            if not metadata_rows:
                return ToolResult(
                    tool_name="search_text",
                    success=True,
                    results=[],
                    total_found=0,
                )
            
            results: List[Dict[str, Any]] = []
            for row in metadata_rows:
                doc_hash = row.get("doc_hash")
                if doc_hash and doc_hash not in matching_doc_hashes_set:
                    continue
                text_preview = (row.get("text") or "")[:context_chars]
                results.append({
                    "doc_hash": doc_hash,
                    "chunk_id": row.get("chunk_id"),
                    "order_index": row.get("order_index", 0),
                    "text": text_preview,
                    "document_name": row.get("original_name") or row.get("document_name") or "Unknown",
                    "score": 1.0,
                    "match_type": "keyword+metadata" if query else "metadata",
                })
                if len(results) >= top_k:
                    break
            
            return ToolResult(
                tool_name="search_text",
                success=True,
                results=results,
                total_found=len(results),
            )
        
        # Search chunks with LIKE query
        results = []
        search_terms = query.lower().split()
        
        for doc_hash in matching_doc_hashes:
            chunks = await document_store.fetch_chunks(doc_hash=doc_hash)
            doc_info = doc_info_map.get(doc_hash) or await document_store.get_document(doc_hash)
            
            for chunk in chunks:
                chunk_text = chunk.get("text", "").lower()
                
                # Simple relevance scoring: count matching terms
                match_count = sum(1 for term in search_terms if term in chunk_text)
                
                if match_count > 0:
                    results.append({
                        "doc_hash": doc_hash,
                        "chunk_id": chunk.get("chunk_id"),
                        "order_index": chunk.get("order_index", 0),
                        "text": chunk.get("text", "")[:context_chars],
                        "document_name": doc_info.get("original_name", "Unknown") if doc_info else "Unknown",
                        "score": match_count / len(search_terms) if search_terms else 0,
                        "match_type": "keyword",
                    })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        return ToolResult(
            tool_name="search_text",
            success=True,
            results=results,
            total_found=len(results),
        )
        
    except Exception as e:
        logger.exception(f"search_text failed: {e}")
        return ToolResult(
            tool_name="search_text",
            success=False,
            error=str(e),
        )


async def search_semantic(
    bucket: str,
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    settings: Any,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    context_chars: int = 500,
    doc_id: Optional[str] = None,
) -> ToolResult:
    """
    Semantic/vector search within a bucket.
    
    Args:
        bucket: Target bucket (L1 category ID)
        query: Natural language query for embedding
        document_store: Database access object
        embedding_client: Client for generating embeddings
        embedding_cache: Cache with precomputed embeddings
        filters: Optional metadata filters
        top_k: Number of results to return
        context_chars: Characters to include in snippet
        doc_id: Optional - scope search to specific document
    
    Returns:
        ToolResult with semantically similar chunks
    """
    try:
        # Embed the query
        query_vectors = await embedding_client.embed_batch([query])
        if not query_vectors or query_vectors[0] is None:
            return ToolResult(
                tool_name="search_semantic",
                success=False,
                error="Failed to embed query",
            )
        
        query_vec = np.asarray(query_vectors[0], dtype=np.float32)
        
        schema = _match_schema_for_bucket(bucket)
        matching_doc_hashes, doc_info_map = await _collect_bucket_documents(
            document_store=document_store,
            bucket=bucket,
            schema=schema,
            doc_id=doc_id,
        )
        matching_doc_hashes_set: Set[str] = set(matching_doc_hashes)
        
        if schema and filters and matching_doc_hashes_set:
            matching_doc_hashes_set = await _filter_doc_hashes_by_metadata(
                matching_doc_hashes_set,
                schema,
                filters,
                settings,
            )
        
        if not matching_doc_hashes_set:
            return ToolResult(
                tool_name="search_semantic",
                success=True,
                results=[],
                total_found=0,
            )
        
        # Get embedding snapshot and filter by bucket
        snapshot = embedding_cache.snapshot()
        if snapshot.total == 0:
            return ToolResult(
                tool_name="search_semantic",
                success=True,
                results=[],
                total_found=0,
            )
        
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        # Compute similarities
        scores = snapshot.matrix @ query_vec
        
        # Get top candidates
        candidate_count = min(snapshot.total, top_k * 5)  # Fetch extra for filtering
        if candidate_count > 0:
            top_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = []
        
        # Filter by bucket and build results
        results = []
        for idx in top_indices:
            chunk_id = snapshot.chunk_ids[idx]
            score = float(scores[idx])
            
            # Get chunk info
            chunks = await document_store.fetch_chunks_by_ids([chunk_id])
            if not chunks:
                continue
            
            chunk = chunks[0]
            doc_hash = chunk.get("doc_hash")
            
            # Filter by bucket/metadata
            if doc_hash not in matching_doc_hashes_set:
                continue
            
            doc_info = doc_info_map.get(doc_hash) or await document_store.get_document(doc_hash)
            
            results.append({
                "doc_hash": doc_hash,
                "chunk_id": chunk_id,
                "order_index": chunk.get("order_index", 0),
                "text": chunk.get("text", "")[:context_chars],
                "document_name": doc_info.get("original_name", "Unknown") if doc_info else "Unknown",
                "score": score,
                "match_type": "semantic",
            })
            
            if len(results) >= top_k:
                break
        
        return ToolResult(
            tool_name="search_semantic",
            success=True,
            results=results,
            total_found=len(results),
        )
        
    except Exception as e:
        logger.exception(f"search_semantic failed: {e}")
        return ToolResult(
            tool_name="search_semantic",
            success=False,
            error=str(e),
        )


async def get_document_metadata(
    doc_id: str,
    document_store: Any,
    settings: Any,
) -> ToolResult:
    """
    Get full metadata for a document.
    
    Args:
        doc_id: Document hash
        document_store: Database access object
        settings: Application settings
    
    Returns:
        ToolResult with document metadata
    """
    try:
        # Get document info
        doc = await document_store.get_document(doc_id)
        if not doc:
            return ToolResult(
                tool_name="get_document_metadata",
                success=False,
                error=f"Document not found: {doc_id}",
            )
        
        # Get classification
        classification = await document_store.get_classification(doc_id)
        
        # Get extracted metadata if available
        from ..extraction.schemas import get_schema_for_category
        from ..extraction.persistence import MetadataStore
        
        extracted_metadata = None
        if classification:
            l1_id = classification.get("l1_id")
            l2_id = classification.get("l2_id")
            schema = get_schema_for_category(l1_id, l2_id)
            
            if schema:
                meta_store = MetadataStore(str(settings.doc_store_path))
                extracted_metadata = await meta_store.get_metadata(schema, doc_id)
        
        result = {
            "doc_hash": doc_id,
            "original_name": doc.get("original_name"),
            "status": doc.get("status"),
            "size": doc.get("size"),
            "created_at": doc.get("created_at"),
            "classification": {
                "l1_id": classification.get("l1_id") if classification else None,
                "l1_name": classification.get("l1_name") if classification else None,
                "l2_id": classification.get("l2_id") if classification else None,
                "l2_name": classification.get("l2_name") if classification else None,
            } if classification else None,
            "extracted_metadata": extracted_metadata,
        }
        
        return ToolResult(
            tool_name="get_document_metadata",
            success=True,
            results=[result],
            total_found=1,
        )
        
    except Exception as e:
        logger.exception(f"get_document_metadata failed: {e}")
        return ToolResult(
            tool_name="get_document_metadata",
            success=False,
            error=str(e),
        )


def _normalize(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _match_schema_for_bucket(bucket: str) -> Optional[ExtractionSchema]:
    bucket_norm = _normalize(bucket)
    if not bucket_norm:
        return None
    for schema in EXTRACTION_SCHEMAS.values():
        candidates = [
            schema.schema_id,
            schema.display_name,
            schema.target_l2_category,
        ]
        for candidate in candidates:
            if _normalize(candidate) == bucket_norm:
                return schema
    return None


async def _collect_bucket_documents(
    document_store: Any,
    bucket: str,
    schema: Optional[ExtractionSchema],
    doc_id: Optional[str] = None,
) -> tuple[List[str], Dict[str, Any]]:
    """Return matching doc hashes and a map of doc info for the requested bucket."""
    docs = await document_store.list_documents()
    matches: List[str] = []
    doc_info_map: Dict[str, Any] = {}
    bucket_norm = _normalize(bucket)
    schema_l1 = _normalize(schema.target_l1_category) if schema else ""
    schema_l2 = _normalize(schema.target_l2_category) if schema and schema.target_l2_category else ""
    
    for doc in docs:
        if doc.get("status") != "processed":
            continue
        if doc_id and doc["doc_hash"] != doc_id:
            continue
        
        classification = await document_store.get_classification(doc["doc_hash"])
        if not classification:
            continue
        
        l1_id = _normalize(classification.get("l1_id"))
        l1_name = _normalize(classification.get("l1_name"))
        l2_id = _normalize(classification.get("l2_id"))
        l2_name = _normalize(classification.get("l2_name"))
        
        if schema:
            if l1_id != schema_l1:
                continue
            if schema_l2 and l2_id != schema_l2:
                continue
        elif bucket_norm:
            if bucket_norm not in {l1_id, l1_name, l2_id, l2_name}:
                continue
        
        matches.append(doc["doc_hash"])
        doc_info_map[doc["doc_hash"]] = doc
    
    return matches, doc_info_map


async def _filter_doc_hashes_by_metadata(
    doc_hashes: Set[str],
    schema: ExtractionSchema,
    filters: Optional[Dict[str, Any]],
    settings: Any,
) -> Set[str]:
    """Filter doc hashes using metadata filters."""
    if not filters:
        return doc_hashes
    
    if not doc_hashes:
        return set()
    
    meta_store = MetadataStore(str(settings.doc_store_path))
    rows = await meta_store.query_metadata(
        schema=schema,
        filters=filters,
        limit=max(len(doc_hashes), 100),
    )
    allowed = {row.get("doc_hash") for row in rows if row.get("doc_hash")}
    if not allowed:
        return set()
    return {doc_hash for doc_hash in doc_hashes if doc_hash in allowed}


async def execute_tool(
    tool_name: str,
    args: Dict[str, Any],
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    settings: Any,
) -> ToolResult:
    """
    Execute a tool by name with given arguments.
    
    This is the main dispatcher called by the orchestrator.
    """
    if tool_name == "search_text":
        return await search_text(
            bucket=args.get("bucket", ""),
            query=args.get("query", ""),
            document_store=document_store,
            settings=settings,
            filters=args.get("filters"),
            top_k=args.get("top_k", 10),
            context_chars=args.get("context_chars", 400),
            doc_id=args.get("doc_id"),
        )
    
    elif tool_name == "search_semantic":
        return await search_semantic(
            bucket=args.get("bucket", ""),
            query=args.get("query", ""),
            document_store=document_store,
            embedding_client=embedding_client,
            embedding_cache=embedding_cache,
            settings=settings,
            filters=args.get("filters"),
            top_k=args.get("top_k", 10),
            context_chars=args.get("context_chars", 500),
            doc_id=args.get("doc_id"),
        )
    
    elif tool_name == "get_document_metadata":
        return await get_document_metadata(
            doc_id=args.get("doc_id", ""),
            document_store=document_store,
            settings=settings,
        )
    
    else:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            error=f"Unknown tool: {tool_name}",
        )
