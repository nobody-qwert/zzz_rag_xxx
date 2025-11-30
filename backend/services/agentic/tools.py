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
from typing import Any, Dict, List, Optional

import numpy as np

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
        # Get all processed documents
        all_docs = await document_store.list_documents()
        
        # Filter by bucket (classification L1) and optional doc_id
        matching_doc_hashes = []
        for doc in all_docs:
            if doc.get("status") != "processed":
                continue
            if doc_id and doc["doc_hash"] != doc_id:
                continue
            
            # Check classification
            classification = await document_store.get_classification(doc["doc_hash"])
            if classification and classification.get("l1_id") == bucket:
                matching_doc_hashes.append(doc["doc_hash"])
        
        if not matching_doc_hashes:
            return ToolResult(
                tool_name="search_text",
                success=True,
                results=[],
                total_found=0,
            )
        
        # Search chunks with LIKE query
        results = []
        search_terms = query.lower().split()
        
        for doc_hash in matching_doc_hashes:
            chunks = await document_store.fetch_chunks(doc_hash=doc_hash)
            doc_info = await document_store.get_document(doc_hash)
            
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
        
        # Get documents matching the bucket
        all_docs = await document_store.list_documents()
        matching_doc_hashes = set()
        
        for doc in all_docs:
            if doc.get("status") != "processed":
                continue
            if doc_id and doc["doc_hash"] != doc_id:
                continue
            
            classification = await document_store.get_classification(doc["doc_hash"])
            if classification and classification.get("l1_id") == bucket:
                matching_doc_hashes.add(doc["doc_hash"])
        
        if not matching_doc_hashes:
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
            
            # Filter by bucket
            if doc_hash not in matching_doc_hashes:
                continue
            
            doc_info = await document_store.get_document(doc_hash)
            
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
