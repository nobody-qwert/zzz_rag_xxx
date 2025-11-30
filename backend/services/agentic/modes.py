"""
LLM modes for the agentic RAG system.

Implements the three conceptual modes:
- decompose_query: Parse user query into structured plan
- plan_search: Mode 1 - Decide search strategy
- review_evidence: Mode 2 - Decide if evidence is sufficient
- compose_answer: Mode 3 - Generate final answer with citations
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .prompts import (
    DECOMPOSER_SYSTEM_PROMPT,
    DECOMPOSER_USER_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    COMPOSER_SYSTEM_PROMPT,
    COMPOSER_USER_TEMPLATE,
    format_evidence_for_review,
    format_evidence_for_composer,
)

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Result from query decomposition."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class PlanResult:
    """Result from search planning."""
    success: bool
    target_buckets: List[str] = None
    strategy: str = "hybrid"
    initial_queries: List[str] = None
    filters_hint: Optional[Dict[str, Any]] = None
    max_tool_calls: int = 4
    error: Optional[str] = None
    raw_response: Optional[str] = None


@dataclass
class ReviewResult:
    """Result from evidence review."""
    success: bool
    status: str = "more"  # "enough", "more", "clarify"
    reason: str = ""
    next_tool_call: Optional[Dict[str, Any]] = None
    clarification_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response text."""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try to parse the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


async def decompose_query(
    query: str,
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> DecompositionResult:
    """
    Decompose a user query into a structured search plan.
    
    This is the first step that parses the natural language query
    into entities, constraints, buckets, and subqueries.
    """
    try:
        user_prompt = DECOMPOSER_USER_TEMPLATE.format(query=query)
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        
        if not response.choices:
            return DecompositionResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            return DecompositionResult(
                success=False,
                error="Could not parse JSON from response",
                raw_response=raw_response,
            )
        
        return DecompositionResult(
            success=True,
            data=data,
            raw_response=raw_response,
        )
        
    except Exception as e:
        logger.exception(f"decompose_query failed: {e}")
        return DecompositionResult(
            success=False,
            error=str(e),
        )


async def plan_search(
    query: str,
    decomposition: Dict[str, Any],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> PlanResult:
    """
    Generate a search plan based on query decomposition.
    
    Mode 1: Decides which buckets to search, what strategy to use,
    and constructs initial queries.
    """
    try:
        decomp_str = json.dumps(decomposition, indent=2)
        user_prompt = PLANNER_USER_TEMPLATE.format(
            query=query,
            decomposition=decomp_str,
        )
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        
        if not response.choices:
            return PlanResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            # Fall back to decomposition data
            return PlanResult(
                success=True,
                target_buckets=decomposition.get("primary_buckets", []),
                strategy="hybrid",
                initial_queries=decomposition.get("topic_terms", [query]),
                filters_hint=None,
                max_tool_calls=4,
                raw_response=raw_response,
            )
        
        return PlanResult(
            success=True,
            target_buckets=data.get("target_buckets", []),
            strategy=data.get("strategy", "hybrid"),
            initial_queries=data.get("initial_queries", [query]),
            filters_hint=data.get("filters_hint"),
            max_tool_calls=data.get("max_tool_calls", 4),
            raw_response=raw_response,
        )
        
    except Exception as e:
        logger.exception(f"plan_search failed: {e}")
        return PlanResult(
            success=False,
            error=str(e),
        )


async def review_evidence(
    query: str,
    plan: PlanResult,
    evidence: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    temperature: float = 0.1,
) -> ReviewResult:
    """
    Review collected evidence and decide next step.
    
    Mode 2: Decides if we have enough evidence, need more searches,
    or need to ask for clarification.
    """
    try:
        plan_str = json.dumps({
            "target_buckets": plan.target_buckets,
            "strategy": plan.strategy,
            "initial_queries": plan.initial_queries,
            "filters_hint": plan.filters_hint,
        }, indent=2)
        
        evidence_summary = format_evidence_for_review(evidence)
        
        user_prompt = REVIEWER_USER_TEMPLATE.format(
            query=query,
            plan=plan_str,
            evidence_count=len(evidence),
            evidence_summary=evidence_summary,
        )
        
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )
        
        if not response.choices:
            return ReviewResult(
                success=False,
                error="LLM returned no choices",
            )
        
        raw_response = response.choices[0].message.content or ""
        data = _extract_json(raw_response)
        
        if not data:
            # Default to "enough" if we have evidence, otherwise "more"
            return ReviewResult(
                success=True,
                status="enough" if evidence else "more",
                reason="Could not parse reviewer response",
                raw_response=raw_response,
            )
        
        return ReviewResult(
            success=True,
            status=data.get("status", "enough"),
            reason=data.get("reason", ""),
            next_tool_call=data.get("next_tool_call"),
            clarification_details=data.get("clarification_details"),
            raw_response=raw_response,
        )
        
    except Exception as e:
        logger.exception(f"review_evidence failed: {e}")
        return ReviewResult(
            success=False,
            error=str(e),
        )


async def compose_answer(
    query: str,
    evidence: List[Dict[str, Any]],
    llm_client: Any,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    stream: bool = False,
):
    """
    Compose final answer from evidence.
    
    Mode 3: Generates the final answer with proper citations.
    
    Args:
        query: User query
        evidence: List of evidence items
        llm_client: OpenAI-compatible client
        model: Model name
        temperature: LLM temperature
        max_tokens: Max response tokens
        stream: If True, returns async generator for streaming
    
    Returns:
        If stream=False: str (the answer)
        If stream=True: async generator yielding chunks
    """
    evidence_str = format_evidence_for_composer(evidence)
    
    user_prompt = COMPOSER_USER_TEMPLATE.format(
        query=query,
        evidence=evidence_str,
    )
    
    messages = [
        {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    if stream:
        async def stream_generator():
            try:
                response = await llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.exception(f"compose_answer streaming failed: {e}")
                yield f"\n\n[Error generating answer: {e}]"
        
        return stream_generator()
    
    else:
        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if not response.choices:
                return "I was unable to generate an answer."
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            logger.exception(f"compose_answer failed: {e}")
            return f"Error generating answer: {e}"


def verify_citations(answer: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Verify that all citations in the answer exist in the evidence.
    
    Removes or marks invalid citations to prevent hallucination.
    """
    # Extract all doc_ids from evidence
    valid_ids = set()
    for item in evidence:
        doc_id = item.get("doc_hash", item.get("doc_id"))
        if doc_id:
            valid_ids.add(doc_id)
            # Also add partial matches (first 8 chars)
            valid_ids.add(doc_id[:8])
    
    # Find all citations in answer
    citation_pattern = r'\[([a-zA-Z0-9_-]+)\]'
    
    def check_citation(match):
        cited_id = match.group(1)
        # Check if citation is valid
        if cited_id in valid_ids:
            return match.group(0)
        # Check partial match
        for valid_id in valid_ids:
            if valid_id.startswith(cited_id) or cited_id.startswith(valid_id):
                return match.group(0)
        # Invalid citation - mark it
        logger.warning(f"Invalid citation removed: [{cited_id}]")
        return f"[citation needed]"
    
    verified_answer = re.sub(citation_pattern, check_citation, answer)
    return verified_answer


def build_clarification_response(details: Dict[str, Any]) -> str:
    """Build a user-friendly clarification request."""
    clarify_type = details.get("type", "unknown")
    missing_info = details.get("missing_info", "")
    
    if clarify_type == "no_results":
        return (
            f"I couldn't find any documents matching your query. {missing_info}\n\n"
            "Could you please:\n"
            "- Check if the search terms are correct\n"
            "- Try broader search criteria\n"
            "- Specify a different document category"
        )
    
    elif clarify_type == "overload":
        return (
            f"I found too many results to process effectively. {missing_info}\n\n"
            "Could you please narrow your search by:\n"
            "- Adding specific dates or date ranges\n"
            "- Specifying particular companies or entities\n"
            "- Adding amount constraints (e.g., 'over $1000')\n"
            "- Being more specific about what you're looking for"
        )
    
    else:
        return (
            f"I need more information to answer your question. {missing_info}\n\n"
            "Please provide additional details or rephrase your query."
        )
