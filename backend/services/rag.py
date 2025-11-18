from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from fastapi import HTTPException

try:
    from ..embeddings import EmbeddingClient
    from ..schemas import AskRequest, AskResponse
    from ..token_utils import estimate_messages_tokens, estimate_tokens, truncate_text_to_tokens
    from ..dependencies import document_store, settings
    from ..utils.conversation import (
        build_distilled_query,
        render_context,
        summarize_history,
        trim_history_for_budget,
    )
    from ..utils.vectors import cosine_similarity
except ImportError:  # pragma: no cover
    from embeddings import EmbeddingClient  # type: ignore
    from schemas import AskRequest, AskResponse  # type: ignore
    from token_utils import estimate_messages_tokens, estimate_tokens, truncate_text_to_tokens  # type: ignore
    from dependencies import document_store, settings  # type: ignore
    from utils.conversation import (  # type: ignore
        build_distilled_query,
        render_context,
        summarize_history,
        trim_history_for_budget,
    )
    from utils.vectors import cosine_similarity  # type: ignore


logger = logging.getLogger(__name__)


async def ask_question(req: AskRequest) -> AskResponse:
    is_continuation = bool(req.continue_last)
    raw_query = (req.query or "").strip()
    if not is_continuation and not raw_query:
        raise HTTPException(status_code=400, detail="Query must not be empty")
    if is_continuation and not (req.conversation_id or "").strip():
        raise HTTPException(status_code=400, detail="conversation_id is required to continue a response")

    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")

    conversation_id = (req.conversation_id or "").strip() or str(uuid4())
    existing_conversation = await document_store.get_conversation(conversation_id)
    if not existing_conversation:
        await document_store.create_conversation(conversation_id)

    history_rows = await document_store.fetch_conversation_messages(conversation_id)
    history_messages = [{"role": row["role"], "content": row["content"]} for row in history_rows]

    if is_continuation:
        last_user_message: Optional[Dict[str, str]] = None
        for entry in reversed(history_messages):
            if (entry.get("role") or "").lower() == "user":
                last_user_message = entry
                break
        if not last_user_message:
            raise HTTPException(status_code=400, detail="Cannot continue because no prior user question was found")
        question = (last_user_message.get("content") or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Last user question is empty; cannot continue")
    else:
        question = raw_query

    history_summary = summarize_history(history_messages)
    distilled_query = build_distilled_query(history_summary, question)

    emb_client = EmbeddingClient()
    distilled_vec = await _embed_query(distilled_query, emb_client)
    if distilled_vec is None:
        raise HTTPException(status_code=500, detail="Failed to embed query")

    query_vec = distilled_vec
    try:
        hyde_text = await _generate_hyde_answer(question, history_summary)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("HyDE generation raised an unexpected error: %s", exc)
        hyde_text = None

    if hyde_text:
        hyde_vec = await _embed_query(hyde_text, emb_client)
        if hyde_vec is not None:
            try:
                query_vec = (np.asarray(distilled_vec, dtype=np.float32) + np.asarray(hyde_vec, dtype=np.float32)) / 2.0
            except Exception as exc:  # pragma: no cover - fallback
                logger.warning("HyDE vector blend failed: %s", exc)
                query_vec = distilled_vec

    parser_specs: List[Dict[str, str]] = [
        {"config_id": settings.chunk_config_small_id, "scale": "small"}
    ]
    if (
        settings.chunk_config_large_id
        and settings.chunk_config_large_id != settings.chunk_config_small_id
    ):
        parser_specs.append({"config_id": settings.chunk_config_large_id, "scale": "large"})

    chunks: List[Dict[str, Any]] = []
    for spec in parser_specs:
        config_id = spec["config_id"]
        scale = spec["scale"]
        parser_chunks = await document_store.fetch_chunks(chunk_config_id=config_id)
        for chunk in parser_chunks:
            chunk_copy = dict(chunk)
            chunk_copy["chunk_config_id"] = config_id
            chunk_copy["scale"] = scale
            chunks.append(chunk_copy)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    emb_rows = await document_store.fetch_embeddings_for_chunks(chunk_ids)
    emb_by_id = {row.chunk_id: row for row in emb_rows}

    docs = await document_store.list_documents()
    docs_by_hash = {doc["doc_hash"]: doc for doc in docs}

    chunks_per_doc: Dict[str, int] = {}
    for chunk in chunks:
        doc_hash = chunk["doc_hash"]
        chunks_per_doc[doc_hash] = chunks_per_doc.get(doc_hash, 0) + 1

    scored: List[Dict[str, Any]] = []
    for chunk in chunks:
        row = emb_by_id.get(chunk["chunk_id"])  # type: ignore[arg-type]
        if not row:
            continue
        score = cosine_similarity(row.vector, query_vec)
        scored.append({"chunk": chunk, "score": score})
    scored.sort(key=lambda entry: entry["score"], reverse=True)

    top_k = max(1, min(req.top_k, 20))
    filtered_sections = [entry for entry in scored if entry["score"] >= settings.min_context_similarity]
    context_sections = filtered_sections[:top_k]

    if not settings.llm_base_url or not settings.llm_api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_BASE_URL and LLM_API_KEY must be set",
        )

    prompt_token_limit = min(
        max(512, settings.chat_context_window - settings.chat_completion_reserve),
        settings.chat_context_window - 16,
    )
    if prompt_token_limit <= 0:
        raise HTTPException(status_code=500, detail="Context window too small for configured reserve")

    trimmed_history, history_truncated, _ = trim_history_for_budget(
        history_messages,
        model=settings.llm_model,
        token_limit=prompt_token_limit,
        system_prompt=settings.system_prompt,
    )

    history_for_prompt = list(trimmed_history)
    user_message_for_llm = (
        f"{settings.continue_prompt}\n\nOriginal question: {question}"
        if is_continuation
        else question
    )
    context_truncated = False

    while True:
        context_text = render_context(context_sections)
        context_content = (
            "Use only the following retrieved document snippets as context. "
            "Cite them as [source N].\n"
            + (context_text if context_text else "(No retrieved snippets available.)")
        )
        combined_user_message = f"{context_content}\n\nQuestion:\n{user_message_for_llm}"
        messages = [
            {"role": "system", "content": settings.system_prompt},
            *history_for_prompt,
            {"role": "user", "content": combined_user_message},
        ]
        prompt_tokens = _estimate_prompt_tokens(messages)
        if prompt_tokens <= prompt_token_limit:
            break
        if history_for_prompt:
            history_for_prompt.pop(0)
            history_truncated = True
            continue
        if context_sections:
            context_sections.pop()
            context_truncated = True
            continue
        without_user = messages[:-1]
        remaining = prompt_token_limit - _estimate_prompt_tokens(without_user)
        if remaining <= 0:
            raise HTTPException(status_code=500, detail="Prompt exceeds available context even after trimming")
        truncated = truncate_text_to_tokens(question, remaining, model=settings.llm_model)
        if not truncated:
            truncated = question[:200]
        if truncated == user_message_for_llm:
            raise HTTPException(status_code=500, detail="Unable to fit prompt within context window")
        user_message_for_llm = truncated
        context_truncated = True

    context_tokens_used = prompt_tokens
    context_usage_ratio = min(1.0, context_tokens_used / float(settings.chat_context_window))

    final_context_text = render_context(context_sections)
    final_context_content = (
        "Use only the following retrieved document snippets as context. "
        "Cite them as [source N].\n"
        + (final_context_text if final_context_text else "(No retrieved snippets available.)")
    )
    messages = [
        {"role": "system", "content": settings.system_prompt},
        *history_for_prompt,
        {"role": "user", "content": f"{final_context_content}\n\nQuestion:\n{user_message_for_llm}"},
    ]

    answer = ""
    finish_reason: Optional[str] = None
    if not context_sections:
        answer = settings.no_context_response
        finish_reason = "no_context"
    else:
        try:
            from openai import AsyncOpenAI  # type: ignore

            client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=0.2,
                max_tokens=settings.chat_completion_max_tokens,
                stream=False,
            )
            if not response.choices:
                raise HTTPException(status_code=500, detail="LLM returned no choices")
            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")

    sources: List[Dict[str, Any]] = []
    for entry in context_sections:
        chunk = entry["chunk"]
        doc_hash = chunk["doc_hash"]
        doc = docs_by_hash.get(doc_hash, {})
        total_chunks = chunks_per_doc.get(doc_hash, 0)
        sources.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": entry["score"],
                "doc_hash": doc_hash,
                "order_index": chunk["order_index"],
                "document_name": doc.get("original_name", "unknown"),
                "total_chunks": total_chunks,
                "chunk_text": chunk.get("text", ""),
                "chunk_text_preview": chunk.get("text", "")[:200],
                "parser": chunk.get("parser", settings.ocr_parser_key),
                "scale": chunk.get("scale"),
            }
        )

    if not is_continuation:
        user_token_count = estimate_tokens(question, model=settings.llm_model)
        await document_store.append_conversation_message(
            conversation_id,
            role="user",
            content=question,
            token_count=user_token_count,
        )
    assistant_token_count = estimate_tokens(answer, model=settings.llm_model)
    await document_store.append_conversation_message(
        conversation_id,
        role="assistant",
        content=answer,
        token_count=assistant_token_count,
    )

    updated_history = list(history_messages)
    if not is_continuation:
        updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": answer})
    summary_text = summarize_history(updated_history)
    await document_store.update_conversation_summary(conversation_id, summary_text)

    needs_follow_up = finish_reason not in (None, "stop")

    return AskResponse(
        answer=answer,
        sources=sources,
        conversation_id=conversation_id,
        context_tokens_used=context_tokens_used,
        context_window_limit=settings.chat_context_window,
        context_usage=context_usage_ratio,
        context_truncated=(history_truncated or context_truncated),
        finish_reason=finish_reason,
        needs_follow_up=needs_follow_up,
    )


async def warmup_llm() -> Dict[str, Any]:
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        return {"warmup_complete": False, "error": "LLM env missing"}
    try:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "Warmup"},
                {"role": "user", "content": "Say ready."},
            ],
            max_tokens=4,
            temperature=0,
        )
        return {"warmup_complete": True, "status": "ready"}
    except Exception as exc:
        return {"warmup_complete": False, "error": str(exc)}


async def _embed_query(text: str, client: EmbeddingClient) -> Any:
    vectors = await client.embed_batch([text])
    return vectors[0] if vectors else None


def _estimate_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    return estimate_messages_tokens(messages, model=settings.llm_model)


async def _generate_hyde_answer(question: str, history_summary: str) -> Optional[str]:
    """Produce a hypothetical ideal answer used for HyDE-style retrieval."""
    question_clean = (question or "").strip()
    if not question_clean:
        return None
    if not settings.llm_base_url or not settings.llm_api_key or not settings.llm_model:
        return None
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return None

    prompt_header = (
        "You draft concise, factual answers solely to improve document retrieval. "
        "Rely only on broad world knowledge and avoid speculation."
    )
    summary_text = history_summary.strip() if history_summary else "(no prior conversation)"
    user_prompt = (
        f"Conversation summary: {summary_text}\n"
        f"Question: {question_clean}\n"
        "Provide a single, information-dense paragraph that would ideally answer the question."
    )
    max_tokens = max(64, min(256, settings.chat_completion_max_tokens))

    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": prompt_header},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=False,
        )
    except Exception as exc:  # pragma: no cover - passthrough logging
        logger.debug("HyDE LLM call failed: %s", exc)
        return None

    if not response or not getattr(response, "choices", None):
        return None
    choice = response.choices[0]
    content = getattr(choice, "message", None)
    answer_text = (getattr(content, "content", "") or "").strip()
    return answer_text or None
