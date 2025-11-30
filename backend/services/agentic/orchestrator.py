"""
Agentic RAG Orchestrator.

Main entry point for the agentic RAG system. Implements:
- The main agentic_answer() function
- The streaming version stream_agentic_answer()
- Evidence collection loop with tool calls
- Context pruning to manage token budget
- Citation verification
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from .modes import (
    decompose_query,
    plan_search,
    review_evidence,
    compose_answer,
    verify_citations,
    build_clarification_response,
    PlanResult,
)
from .tools import execute_tool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class AgenticStep:
    """Record of a single step in the agentic loop."""
    step_number: int
    name: str
    kind: str  # "decompose", "plan", "review", "tool", "compose"
    duration_seconds: float = 0.0
    state: str = "done"  # "started", "done", "error"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class AgenticResult:
    """Result from the agentic RAG pipeline."""
    answer: str
    success: bool
    sources: List[Dict[str, Any]] = field(default_factory=list)
    conversation_id: Optional[str] = None
    needs_clarification: bool = False
    clarification_message: Optional[str] = None
    steps: List[AgenticStep] = field(default_factory=list)
    total_tool_calls: int = 0
    evidence_count: int = 0
    finish_reason: Optional[str] = None


async def agentic_answer(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    llm_client: Any,
    settings: Any,
    conversation_id: Optional[str] = None,
    max_tool_calls: int = 5,
) -> AgenticResult:
    """
    Main agentic RAG function.
    
    Orchestrates the full pipeline:
    1. Decompose query
    2. Plan search
    3. Iterative search loop (review → tool call → review ...)
    4. Compose answer
    5. Verify citations
    
    Args:
        query: User question
        document_store: Database access
        embedding_client: For generating embeddings
        embedding_cache: Precomputed embeddings
        llm_client: OpenAI-compatible async client
        settings: Application settings
        conversation_id: Optional conversation ID for context
        max_tool_calls: Maximum tool calls allowed
    
    Returns:
        AgenticResult with answer and metadata
    """
    steps: List[AgenticStep] = []
    evidence: List[Dict[str, Any]] = []
    total_tool_calls = 0
    
    model = settings.llm_model
    
    # Step 0: Decompose query
    decomp_start = time.perf_counter()
    decomp_result = await decompose_query(query, llm_client, model)
    decomp_duration = time.perf_counter() - decomp_start
    
    steps.append(AgenticStep(
        step_number=0,
        name="Decompose Query",
        kind="decompose",
        duration_seconds=decomp_duration,
        details={"decomposition": decomp_result.data} if decomp_result.success else None,
        error=decomp_result.error if not decomp_result.success else None,
    ))
    
    if not decomp_result.success:
        # Fall back to simple decomposition
        decomposition = {
            "intent": "qa",
            "primary_buckets": [],
            "topic_terms": [query],
        }
    else:
        decomposition = decomp_result.data or {}
    
    # Step 1: Plan search
    plan_start = time.perf_counter()
    plan_result = await plan_search(query, decomposition, llm_client, model)
    plan_duration = time.perf_counter() - plan_start
    
    steps.append(AgenticStep(
        step_number=1,
        name="Plan Search",
        kind="plan",
        duration_seconds=plan_duration,
        details={
            "target_buckets": plan_result.target_buckets,
            "strategy": plan_result.strategy,
            "initial_queries": plan_result.initial_queries,
        } if plan_result.success else None,
        error=plan_result.error if not plan_result.success else None,
    ))
    
    if not plan_result.success:
        # Create default plan
        plan_result = PlanResult(
            success=True,
            target_buckets=decomposition.get("primary_buckets", []),
            strategy="hybrid",
            initial_queries=[query],
            max_tool_calls=max_tool_calls,
        )
    
    # Ensure we have some buckets to search
    if not plan_result.target_buckets:
        # Default to searching all buckets if none specified
        plan_result.target_buckets = ["financial_accounting", "legal_compliance", "technical_engineering"]
    
    effective_max_calls = min(max_tool_calls, plan_result.max_tool_calls or 5)
    
    # Step 2: Initial searches based on plan
    for bucket in plan_result.target_buckets[:2]:  # Limit initial buckets
        for search_query in plan_result.initial_queries[:2]:  # Limit initial queries
            if total_tool_calls >= effective_max_calls:
                break
            
            # Execute search based on strategy
            tool_start = time.perf_counter()
            
            if plan_result.strategy in ("keyword", "hybrid"):
                text_result = await execute_tool(
                    tool_name="search_text",
                    args={"bucket": bucket, "query": search_query, "top_k": 5},
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                steps.append(AgenticStep(
                    step_number=len(steps),
                    name=f"search_text({bucket})",
                    kind="tool",
                    duration_seconds=tool_duration,
                    details={
                        "bucket": bucket,
                        "query": search_query,
                        "results_count": text_result.total_found,
                    },
                    error=text_result.error if not text_result.success else None,
                ))
                
                if text_result.success:
                    evidence.extend(text_result.results)
            
            if plan_result.strategy in ("semantic", "hybrid") and total_tool_calls < effective_max_calls:
                tool_start = time.perf_counter()
                semantic_result = await execute_tool(
                    tool_name="search_semantic",
                    args={"bucket": bucket, "query": search_query, "top_k": 5},
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                steps.append(AgenticStep(
                    step_number=len(steps),
                    name=f"search_semantic({bucket})",
                    kind="tool",
                    duration_seconds=tool_duration,
                    details={
                        "bucket": bucket,
                        "query": search_query,
                        "results_count": semantic_result.total_found,
                    },
                    error=semantic_result.error if not semantic_result.success else None,
                ))
                
                if semantic_result.success:
                    evidence.extend(semantic_result.results)
    
    # Deduplicate evidence by chunk_id
    evidence = _deduplicate_evidence(evidence)
    
    # Step 3: Review and refine loop
    for iteration in range(effective_max_calls - total_tool_calls):
        review_start = time.perf_counter()
        review_result = await review_evidence(query, plan_result, evidence, llm_client, model)
        review_duration = time.perf_counter() - review_start
        
        steps.append(AgenticStep(
            step_number=len(steps),
            name="Review Evidence",
            kind="review",
            duration_seconds=review_duration,
            details={
                "status": review_result.status,
                "reason": review_result.reason,
                "evidence_count": len(evidence),
            },
            error=review_result.error if not review_result.success else None,
        ))
        
        if review_result.status == "enough":
            break
        
        if review_result.status == "clarify":
            # Need user clarification
            clarification_msg = build_clarification_response(
                review_result.clarification_details or {}
            )
            return AgenticResult(
                answer=clarification_msg,
                success=True,
                sources=_build_sources(evidence),
                conversation_id=conversation_id,
                needs_clarification=True,
                clarification_message=clarification_msg,
                steps=[_step_to_dict(s) for s in steps],
                total_tool_calls=total_tool_calls,
                evidence_count=len(evidence),
                finish_reason="clarify",
            )
        
        # status == "more" - execute next tool call
        if review_result.next_tool_call and total_tool_calls < effective_max_calls:
            tool_call = review_result.next_tool_call
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            
            tool_start = time.perf_counter()
            tool_result = await execute_tool(
                tool_name=tool_name,
                args=tool_args,
                document_store=document_store,
                embedding_client=embedding_client,
                embedding_cache=embedding_cache,
                settings=settings,
            )
            total_tool_calls += 1
            tool_duration = time.perf_counter() - tool_start
            
            steps.append(AgenticStep(
                step_number=len(steps),
                name=f"{tool_name}",
                kind="tool",
                duration_seconds=tool_duration,
                details={
                    "args": tool_args,
                    "results_count": tool_result.total_found,
                },
                error=tool_result.error if not tool_result.success else None,
            ))
            
            if tool_result.success:
                evidence.extend(tool_result.results)
                evidence = _deduplicate_evidence(evidence)
        else:
            break
    
    # Step 4: Prune evidence to top items
    evidence = _prune_evidence(evidence, max_items=15)
    
    # Step 5: Compose final answer
    compose_start = time.perf_counter()
    answer = await compose_answer(query, evidence, llm_client, model, stream=False)
    compose_duration = time.perf_counter() - compose_start
    
    steps.append(AgenticStep(
        step_number=len(steps),
        name="Compose Answer",
        kind="compose",
        duration_seconds=compose_duration,
    ))
    
    # Step 6: Verify citations
    verified_answer = verify_citations(answer, evidence)
    
    return AgenticResult(
        answer=verified_answer,
        success=True,
        sources=_build_sources(evidence),
        conversation_id=conversation_id,
        needs_clarification=False,
        steps=[_step_to_dict(s) for s in steps],
        total_tool_calls=total_tool_calls,
        evidence_count=len(evidence),
        finish_reason="complete",
    )


async def stream_agentic_answer(
    query: str,
    document_store: Any,
    embedding_client: Any,
    embedding_cache: Any,
    llm_client: Any,
    settings: Any,
    conversation_id: Optional[str] = None,
    max_tool_calls: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Streaming version of agentic RAG.
    
    Yields JSON-lines with:
    - {"type": "step", "step": {...}} - Step progress
    - {"type": "token", "content": "..."} - Answer tokens
    - {"type": "final", ...} - Final result metadata
    """
    steps: List[AgenticStep] = []
    evidence: List[Dict[str, Any]] = []
    total_tool_calls = 0
    
    model = settings.llm_model
    
    def _emit_step(step: AgenticStep, state: str = "done") -> str:
        step_dict = _step_to_dict(step)
        step_dict["state"] = state
        return json.dumps({"type": "step", "step": step_dict}) + "\n"
    
    # Step 0: Decompose query
    decomp_start = time.perf_counter()
    yield _emit_step(AgenticStep(0, "Decompose Query", "decompose"), "started")
    
    decomp_result = await decompose_query(query, llm_client, model)
    decomp_duration = time.perf_counter() - decomp_start
    
    step = AgenticStep(
        step_number=0,
        name="Decompose Query",
        kind="decompose",
        duration_seconds=decomp_duration,
        details={"decomposition": decomp_result.data} if decomp_result.success else None,
    )
    steps.append(step)
    yield _emit_step(step)
    
    if not decomp_result.success:
        decomposition = {"intent": "qa", "primary_buckets": [], "topic_terms": [query]}
    else:
        decomposition = decomp_result.data or {}
    
    # Step 1: Plan search
    plan_start = time.perf_counter()
    yield _emit_step(AgenticStep(1, "Plan Search", "plan"), "started")
    
    plan_result = await plan_search(query, decomposition, llm_client, model)
    plan_duration = time.perf_counter() - plan_start
    
    step = AgenticStep(
        step_number=1,
        name="Plan Search",
        kind="plan",
        duration_seconds=plan_duration,
        details={
            "target_buckets": plan_result.target_buckets,
            "strategy": plan_result.strategy,
        } if plan_result.success else None,
    )
    steps.append(step)
    yield _emit_step(step)
    
    if not plan_result.success:
        plan_result = PlanResult(
            success=True,
            target_buckets=decomposition.get("primary_buckets", []),
            strategy="hybrid",
            initial_queries=[query],
            max_tool_calls=max_tool_calls,
        )
    
    if not plan_result.target_buckets:
        plan_result.target_buckets = ["financial_accounting", "legal_compliance", "technical_engineering"]
    
    effective_max_calls = min(max_tool_calls, plan_result.max_tool_calls or 5)
    
    # Step 2: Initial searches
    step_num = len(steps)
    for bucket in plan_result.target_buckets[:2]:
        for search_query in plan_result.initial_queries[:2]:
            if total_tool_calls >= effective_max_calls:
                break
            
            if plan_result.strategy in ("keyword", "hybrid"):
                yield _emit_step(AgenticStep(step_num, f"search_text({bucket})", "tool"), "started")
                tool_start = time.perf_counter()
                
                text_result = await execute_tool(
                    tool_name="search_text",
                    args={"bucket": bucket, "query": search_query, "top_k": 5},
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                step = AgenticStep(
                    step_number=step_num,
                    name=f"search_text({bucket})",
                    kind="tool",
                    duration_seconds=tool_duration,
                    details={"results_count": text_result.total_found},
                )
                steps.append(step)
                yield _emit_step(step)
                step_num += 1
                
                if text_result.success:
                    evidence.extend(text_result.results)
            
            if plan_result.strategy in ("semantic", "hybrid") and total_tool_calls < effective_max_calls:
                yield _emit_step(AgenticStep(step_num, f"search_semantic({bucket})", "tool"), "started")
                tool_start = time.perf_counter()
                
                semantic_result = await execute_tool(
                    tool_name="search_semantic",
                    args={"bucket": bucket, "query": search_query, "top_k": 5},
                    document_store=document_store,
                    embedding_client=embedding_client,
                    embedding_cache=embedding_cache,
                    settings=settings,
                )
                total_tool_calls += 1
                tool_duration = time.perf_counter() - tool_start
                
                step = AgenticStep(
                    step_number=step_num,
                    name=f"search_semantic({bucket})",
                    kind="tool",
                    duration_seconds=tool_duration,
                    details={"results_count": semantic_result.total_found},
                )
                steps.append(step)
                yield _emit_step(step)
                step_num += 1
                
                if semantic_result.success:
                    evidence.extend(semantic_result.results)
    
    evidence = _deduplicate_evidence(evidence)
    
    # Step 3: Review loop (simplified for streaming)
    for iteration in range(min(2, effective_max_calls - total_tool_calls)):
        yield _emit_step(AgenticStep(step_num, "Review Evidence", "review"), "started")
        review_start = time.perf_counter()
        
        review_result = await review_evidence(query, plan_result, evidence, llm_client, model)
        review_duration = time.perf_counter() - review_start
        
        step = AgenticStep(
            step_number=step_num,
            name="Review Evidence",
            kind="review",
            duration_seconds=review_duration,
            details={"status": review_result.status, "evidence_count": len(evidence)},
        )
        steps.append(step)
        yield _emit_step(step)
        step_num += 1
        
        if review_result.status == "enough":
            break
        
        if review_result.status == "clarify":
            clarification_msg = build_clarification_response(review_result.clarification_details or {})
            yield json.dumps({"type": "token", "content": clarification_msg}) + "\n"
            yield json.dumps({
                "type": "final",
                "answer": clarification_msg,
                "needs_clarification": True,
                "sources": _build_sources(evidence),
                "steps": [_step_to_dict(s) for s in steps],
                "total_tool_calls": total_tool_calls,
                "evidence_count": len(evidence),
                "finish_reason": "clarify",
            }) + "\n"
            return
        
        if review_result.next_tool_call and total_tool_calls < effective_max_calls:
            tool_call = review_result.next_tool_call
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            
            yield _emit_step(AgenticStep(step_num, tool_name, "tool"), "started")
            tool_start = time.perf_counter()
            
            tool_result = await execute_tool(
                tool_name=tool_name,
                args=tool_args,
                document_store=document_store,
                embedding_client=embedding_client,
                embedding_cache=embedding_cache,
                settings=settings,
            )
            total_tool_calls += 1
            tool_duration = time.perf_counter() - tool_start
            
            step = AgenticStep(
                step_number=step_num,
                name=tool_name,
                kind="tool",
                duration_seconds=tool_duration,
                details={"results_count": tool_result.total_found},
            )
            steps.append(step)
            yield _emit_step(step)
            step_num += 1
            
            if tool_result.success:
                evidence.extend(tool_result.results)
                evidence = _deduplicate_evidence(evidence)
    
    # Prune evidence
    evidence = _prune_evidence(evidence, max_items=15)
    
    # Step 4: Compose answer with streaming
    yield _emit_step(AgenticStep(step_num, "Compose Answer", "compose"), "started")
    compose_start = time.perf_counter()
    
    answer_parts: List[str] = []
    answer_generator = await compose_answer(query, evidence, llm_client, model, stream=True)
    
    async for token in answer_generator:
        answer_parts.append(token)
        yield json.dumps({"type": "token", "content": token}) + "\n"
    
    full_answer = "".join(answer_parts)
    verified_answer = verify_citations(full_answer, evidence)
    
    compose_duration = time.perf_counter() - compose_start
    step = AgenticStep(
        step_number=step_num,
        name="Compose Answer",
        kind="compose",
        duration_seconds=compose_duration,
    )
    steps.append(step)
    yield _emit_step(step)
    
    # Final result
    yield json.dumps({
        "type": "final",
        "answer": verified_answer,
        "needs_clarification": False,
        "sources": _build_sources(evidence),
        "conversation_id": conversation_id,
        "steps": [_step_to_dict(s) for s in steps],
        "total_tool_calls": total_tool_calls,
        "evidence_count": len(evidence),
        "finish_reason": "complete",
    }) + "\n"


def _deduplicate_evidence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate evidence items by chunk_id."""
    seen = set()
    result = []
    for item in evidence:
        chunk_id = item.get("chunk_id")
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            result.append(item)
        elif not chunk_id:
            result.append(item)
    return result


def _prune_evidence(
    evidence: List[Dict[str, Any]],
    max_items: int = 15,
) -> List[Dict[str, Any]]:
    """Prune evidence to top items by score."""
    # Sort by score descending
    sorted_evidence = sorted(
        evidence,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )
    return sorted_evidence[:max_items]


def _build_sources(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build source list for response."""
    sources = []
    for item in evidence:
        sources.append({
            "doc_hash": item.get("doc_hash"),
            "chunk_id": item.get("chunk_id"),
            "document_name": item.get("document_name", "Unknown"),
            "score": item.get("score", 0),
            "text_preview": item.get("text", "")[:200],
            "match_type": item.get("match_type", "unknown"),
        })
    return sources


def _step_to_dict(step: AgenticStep) -> Dict[str, Any]:
    """Convert AgenticStep to dict."""
    result = {
        "name": step.name,
        "kind": step.kind,
        "order": step.step_number,
        "duration_seconds": step.duration_seconds,
        "state": step.state,
    }
    if step.details:
        result["details"] = step.details
    if step.error:
        result["error"] = step.error
    return result
