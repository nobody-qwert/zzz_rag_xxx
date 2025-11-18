# Streaming Pipeline Step Signals – Plan & Diagram

This document sketches how to stream pipeline steps so each box appears as the work starts/finishes. It shows where to emit signals in `backend/services/rag.py` and how the frontend can react.

## Mermaid sequence (high level)

```mermaid
sequenceDiagram
    participant B as Backend (_event_stream)
    participant FE as Frontend (ChatPage handler)

    Note over B: Validate request / fetch history (non-streamed)
    B->>FE: step started: Embed distilled query
    B->>B: run embedding
    B->>FE: step done: Embed distilled query (duration)

    B->>FE: step started: HyDE LLM
    B->>B: run HyDE LLM
    B->>FE: step done: HyDE LLM (duration)

    B->>FE: step started: Embed HyDE answer
    B->>B: run embedding (if hyde_text exists)
    B->>FE: step done: Embed HyDE answer (duration)

    B->>FE: step started: Vector search
    B->>B: cosine similarity + sorting
    B->>FE: step done: Vector search (duration, retrieved count)

    B->>B: build prompt/context (can emit optional Build prompt step)

    B->>FE: step started: Main LLM
    B-)FE: token events (type: token)
    B->>FE: step done: Main LLM (duration, TTFT, TPS)

    B->>FE: final payload (answer, sources, steps with state: done)
```

## Where to emit signals in `stream_question`

- Move expensive steps into `_event_stream()` (embed distilled, HyDE LLM, embed HyDE, vector search, prompt build, main LLM).
- Emit `{"type":"step","step":{name,kind,order,state:"started"}}` **before** each operation.
- Emit `{"type":"step","step":{name,kind,order,state:"done",duration_seconds,...}}` **after** each operation.
- Keep appending the “done” step dicts to `steps` so the final payload includes them.
- Stream tokens as you already do (`type: "token"`).

Suggested order keys:
1. Embed distilled query
2. HyDE LLM
3. Embed HyDE answer (if applicable)
4. Vector search
5. Build prompt (optional)
6. Main LLM

## Frontend handling (ChatPage.jsx)

- On `step` events, upsert bubbles keyed by `order` (fallback to name/kind).
- `state:"started"` → show “in progress…” (spinner/badge).
- `state:"done"` → replace content with formatted line (duration, TTFT/TPS when present).
- Final payload `steps` should just ensure all steps end in `done` state (no extra duplicates if you upsert by key).
