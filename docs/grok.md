# Agentic RAG Plan for Large Document Corpora (Revised)

*Revision Date: November 29, 2025*  
*This revised plan addresses key issues from the prior review: (1) Adds structured annotation tools (`annotations_search`, `annotations_aggregate`) for handling numeric/metadata constraints; (2) Fully integrates query decomposition as a foundational step (Mode 0), updating all signatures, pseudo-code, and prompts; (3) Wires route selection into the orchestrator and single-loop patterns with explicit fallbacks; (4) Extends search tools with `filters` for structured constraints. Inconsistencies fixed: Standardized `context_chars` to 400; refined plan to build on decomposition (no redundancy); added "clarify" handling; enhanced single-loop with JSON decisions/routes; merged diagram nodes; added route hints to decomposition prompt; enforced hard max calls; inlined key external references (e.g., from `docs/architecture/rag_workflow.md`) for standalone usability.*

This document describes an **agentic Retrieval-Augmented Generation (RAG)** pattern for searching across large, bucketed document collections (e.g., invoices, datasheets, contracts, certificates) using an LLM that can call search tools.

It’s designed for corpora with **100k–1M+ pages**, where:
- Documents are roughly grouped into *buckets* (e.g., `invoices`, `datasheets`, `contracts`, `certificates`, `generic`).
- You already have parsed text stored in a database / index, including structured annotations (e.g., key-value properties like `vceo_v` or `termination_notice_days`).
- The LLM can call tools like `search_text` or `search_semantic`, which return snippets around hits.

---

## 1. High-Level Overview

### Goals

- Let the **LLM act as an agent**: it plans how to search (including route selection), refines queries, inspects results, and then composes an answer.
- Keep the **backend simple**: tools expose basic search functions; indexes do the heavy lifting.
- Make the approach **domain-agnostic**: same logic works for invoices, contracts, datasheets, certificates, etc.

### Core Components

1. **Indexes / Storage**
   - Pre-chunked text for each document (e.g., 200-token chunks).
   - Structured annotations (e.g., extracted properties/tables as key-value pairs).
   - Indexed by `bucket`, `doc_id`, `chunk_id`, etc.
   - Full-text, vector, and structured search (e.g., SQL-like filters on annotations).

2. **Search Tools (API-level)**
   - `search_text(bucket, query, top_k, context_chars, doc_id?, filters?)`
   - `search_semantic(bucket, query, top_k, context_chars, doc_id?, filters?)`
   - `annotations_search(bucket, predicates, top_k, doc_id?)` (for structured filters)
   - `annotations_aggregate(bucket, group_by, predicates?)` (for summaries/computations)
   - `get_document_metadata(doc_id)` (optional)

3. **LLM “Agent”**
   - Starts with query decomposition to extract intent, buckets, constraints, etc.
   - Decides route (structured/hybrid/long-text), then which bucket(s) to search.
   - Chooses search strategy (keyword vs semantic vs annotations).
   - Calls tools repeatedly, inspects snippets/rows, and stops when it has enough evidence.
   - Composes the final answer from collected snippets/annotations.

4. **Orchestrator (your code)**
   - Wraps calls to the LLM and tools.
   - Enforces max tool calls / limits (hard cap from plan).
   - Maintains an `evidence` list of all tool results.
   - Logs routes, decisions, and fallbacks for observability.

---

## 2. Tools Exposed to the LLM

These tools are **simple and generic**. The intelligent behavior is in the LLM agent, not in the tools. All text-based tools use a standardized `context_chars: 400` (default) for surrounding context.

### 2.1. `search_text`

- **Purpose**: keyword / phrase search with surrounding context.
- **Example signature**:

```jsonc
{
  "name": "search_text",
  "description": "Keyword/phrase search within a specific bucket of documents.",
  "parameters": {
    "type": "object",
    "properties": {
      "bucket": { "type": "string", "description": "Document bucket (e.g., invoices, contracts, datasheets)." },
      "query": { "type": "string", "description": "Keyword or phrase to search for." },
      "top_k": { "type": "integer", "description": "Max number of snippets to return.", "default": 20 },
      "context_chars": { "type": "integer", "description": "Number of characters of context around each hit.", "default": 400 },
      "doc_id": { "type": "string", "description": "Optional filter to search only within a single document.", "nullable": true },
      "filters": { "type": "object", "description": "Optional structured filters (e.g., {year: {op: '=', value: 2023}}).", "nullable": true }
    },
    "required": ["bucket", "query"]
  }
}
```

- **Return shape (example)**: Same as original (array of `{doc_id, chunk_id, score, snippet, metadata}`).

### 2.2. `search_semantic`

- **Purpose**: semantic search using vector embeddings.
- Signature identical to `search_text` (including `filters`).

### 2.3. `annotations_search`

- **Purpose**: Search structured annotations (e.g., properties tables) with predicates/filters.
- **Example signature** (inlined from prior architecture doc for structured routes):

```jsonc
{
  "name": "annotations_search",
  "description": "Search structured annotations (e.g., key-value properties) with predicates.",
  "parameters": {
    "type": "object",
    "properties": {
      "bucket": { "type": "string" },
      "predicates": { "type": "object", "description": "Filters like {properties.item_desc: {op: '~', value: 'tire'}, year: {op: '=', value: 2023}} (op: =, >, <, ~ for like)." },
      "top_k": { "type": "integer", "default": 20 },
      "doc_id": { "type": "string", "nullable": true }
    },
    "required": ["bucket", "predicates"]
  }
}
```

- **Return shape**: Array of `{doc_id, annotation_id, score, row: {field: value}, metadata}` (e.g., rows from tables).

### 2.4. `annotations_aggregate`

- **Purpose**: Aggregate annotations (e.g., count, sum, group_by for lists/computations).
- **Example signature**:

```jsonc
{
  "name": "annotations_aggregate",
  "description": "Aggregate structured annotations (e.g., count matches, group by field).",
  "parameters": {
    "type": "object",
    "properties": {
      "bucket": { "type": "string" },
      "group_by": { "type": "string", "description": "Field to group by (e.g., 'party')." },
      "predicates": { "type": "object", "nullable": true },
      "aggregate": { "type": "string", "description": "e.g., 'count', 'sum(quantity)'." }
    },
    "required": ["bucket", "group_by"]
  }
}
```

- **Return shape**: `{groups: [{key: value, count: N, sum: X}], total: M}`.

### 2.5. `get_document_metadata` (optional but useful)

Unchanged.

---

## 3. Agentic RAG: Four Conceptual Modes

The LLM operates in **four conceptual modes**, starting with decomposition:

1. `decompose_query(query)` – *Decomposer* (new foundational mode)
2. `select_route(decomp)` – *Router* (selects structured/hybrid/long-text)
3. `plan_search(query, decomp, route)` – *Planner*
4. `review_evidence(query, decomp, route, plan, evidence)` – *Controller / Critic*
5. `compose_answer(query, decomp, evidence)` – *Answerer / Summarizer*

Implement as separate LLM calls (**orchestrator pattern**) or single-loop.

### 3.1. Mode 0 – `decompose_query(query)`

**Purpose**: Normalize query into structured JSON for downstream use.

**System prompt**: Unchanged, but add to user message: "Consider routes: structured for numerics/lists (favor annotations), hybrid for mixed, long-text for summaries/QA."

**Example output**: Unchanged.

### 3.2. Mode R – `select_route(decomp)`

**Purpose**: Score and select route based on decomposition signals.

**System prompt sketch** (inlined from prior architecture doc):

> You select retrieval routes:  
> - **Structured**: For list/compute with numeric/metadata constraints (e.g., "list tires 2023"). Bias: annotations_search/aggregate first.  
> - **Hybrid**: Mixed constraints/partial annotations (e.g., "contracts with ACME under 60 days"). Bias: annotations + search_text/semantic.  
> - **Long-text**: Summarize/QA no structured fields (e.g., "explain maintenance"). Bias: semantic on summaries, then chunks.  
> Output JSON: { "preferred_route": "structured|hybrid|long-text", "fallback_order": ["hybrid", "long-text"], "cutoffs": {min_hits: 1, max_hits: 100} }.  
> Base on decomp's intent/constraints.

**Example output**:

```json
{
  "preferred_route": "structured",
  "fallback_order": ["hybrid", "long-text"],
  "cutoffs": { "min_hits": 1, "max_hits": 100 }
}
```

### 3.3. Mode 1 – `plan_search(query, decomp, route)`

**Purpose**: Refine decomp into search plan (builds on primary_buckets/subqueries; no redundancy).

**System prompt sketch**: Update to "Refine decomp's primary_buckets, constraints, subqueries, and route into: target_buckets (from primary), strategy (per route bias), initial_queries (from subqueries/topic_terms), max_tool_calls (3-5, hard cap)."

**Example output**: Unchanged, but e.g., target_buckets from decomp.

### 3.4. Mode 2 – `review_evidence(query, decomp, route, plan, evidence)`

**Purpose**: Decide next action, with route fallbacks.

**Inputs**: Add `decomp`, `route`.

**System prompt sketch**: Update to include decomp/route: "Use decomp constraints for filters; follow route bias (e.g., annotations first). Output JSON: status ('enough'|'more'|'clarify'), reason, next_tool_call (null or {tool, args with filters from decomp}). For clarify: {type: 'no_low'|'overload', attempts: [...], reason: '...'}."

**Example outputs**: Unchanged; add clarify example:

```json
{
  "status": "clarify",
  "reason": "Zero hits after annotations fallback.",
  "clarification": { "type": "no_low", "attempts": ["annotations_search(year=2023)"], "reason": "No 2023 docs" }
}
```

### 3.5. Mode 3 – `compose_answer(query, decomp, evidence)`

**Purpose**: Answer using evidence, per decomp's intent/output_preferences.

**System prompt sketch**: Update: "Use decomp.intent (e.g., table for list) and output_preferences; cite via doc_id/annotation_id."

### 3.6. Route Selection Cheat Sheet

| Decomposition signals | Preferred route | Tool bias | Notes |
| --- | --- | --- | --- |
| `intent` = `list` or `compute` + explicit numeric/metadata constraints | Structured / annotations-first | `annotations_search`, `annotations_aggregate`, then fall back to `search_text` if empty | Mirrors “receipts containing tires” workflow: filter rows via `properties.*` before long-text. |
| Mixed constraints, partial annotation coverage, or contradictory signals | Hybrid | Run annotations for coverage, then `search_text`/`search_semantic` in same bucket | Fallback rules: structured → hybrid → long-text on low hits. |
| `intent` = `summarize` or `qa` with no structured fields, manuals-heavy corpora | Long-text | `search_semantic` scoped to summaries, then drill into segments | Start at summary vectors before fine-grained snippets. |

Persist router choice/fallbacks for observability (log success/failure, cutoffs).

---

## 4. Orchestrator Loop (Multi-Step Pattern)

Updated pseudo-code with decomposition, routing, clarify handling, and hard caps:

```python
def agentic_answer(user_query):
    # 0) Decompose
    decomp = llm_decompose_query(user_query)  # Mode 0

    # R) Route
    route = llm_select_route(decomp)  # Mode R

    # 1) Plan
    plan = llm_plan_search(user_query, decomp, route)  # Mode 1
    evidence = []

    # 2) Iterative search
    for step in range(plan["max_tool_calls"]):  # Hard cap
        decision = llm_review_evidence(user_query, decomp, route, plan, evidence)  # Mode 2

        if decision["status"] == "enough":
            break
        elif decision["status"] == "clarify":
            issue = decision["clarification"]
            emit_clarification_prompt(issue["type"], issue["attempts"], issue["reason"], user_query)
            return  # Or loop back after user input
        elif decision["status"] != "more":
            break  # Safety

        next_call = decision["next_tool_call"]
        if not next_call:
            # Fallback: Advance route (e.g., structured -> hybrid)
            route = route["fallback_order"][0]
            continue

        tool = next_call["tool"]
        args = next_call["args"]  # Includes filters from decomp

        result = execute_tool(tool, **args)
        evidence.append({
            "tool": tool,
            "args": args,
            "results": result
        })
        # Check cutoffs (e.g., max_hits from route)
        if len(evidence) > route["cutoffs"]["max_hits"]:
            emit_clarification_prompt("overload", [...], "Too many results", user_query)
            return

    # 3) Compose
    final_answer = llm_compose_answer(user_query, decomp, evidence)  # Mode 3
    return final_answer
```

### 4.1. Comprehensive Workflow Diagram

Consolidated Mermaid (merges prior diagrams; adds decomp/router/clarify):

```mermaid
flowchart TD
    U[User Query] --> D[LLM: decompose_query Mode 0]
    D --> R[LLM: select_route Mode R<br/>structured/hybrid/long-text]
    R --> P[LLM: plan_search Mode 1<br/>buckets/queries from decomp]
    P --> F[Apply filters from decomp<br/>e.g., metadata predicates]
    F --> O["Initialize evidence = []"]
    O --> L[LLM: review_evidence Mode 2<br/>next tool/fallback?]
    L -->|status=more| T[Execute tool<br/>annotations_search etc. per route]
    T --> E[Append to evidence<br/>check cutoffs]
    E --> L
    L -->|fallback route| R
    L -->|clarify: No/Low| CLAR_LOW["Emit No/Low Prompt P3<br/>attempts + hints"]
    L -->|clarify: Overload| CLAR_OVER["Emit Overload Prompt P4<br/>filters/export nudge"]
    CLAR_LOW --> U  %% Loop back
    CLAR_OVER --> U
    L -->|enough or max reached| C[LLM: compose_answer Mode 3 P5<br/>per intent/format]
    C --> POST["Post-process: log route/evidence<br/>Update conv summary"]
    POST --> NEXT["Next turn"]

    style D fill:#fef9c3,stroke:#d97706
    style R fill:#dcfce7,stroke:#15803d,stroke-width:2px
    style P fill:#fef9c3,stroke:#d97706
    style F fill:#fef9c3,stroke:#d97706
    style L fill:#dcfce7,stroke:#15803d,stroke-width:2px
    style T fill:#fef9c3,stroke:#d97706
    style CLAR_LOW fill:#ffcccb,stroke:#dc2626
    style CLAR_OVER fill:#ffcccb,stroke:#dc2626
    style C fill:#dcfce7,stroke:#15803d,stroke-width:2px
```

---

## 5. Single-Loop Tool-Calling Agent Pattern

Enhanced for decomposition/routes/JSON decisions.

### 5.1. Sketch of the system prompt

> You are an AI assistant with document search tools.  
> First, internally decompose the query into JSON (intent, buckets, constraints, subqueries, output_preferences) as per schema. Then select route (structured/hybrid/long-text).  
> Answer by:  
> 1. Choosing buckets (from decomp).  
> 2. Calling tools per route bias (e.g., annotations first for structured).  
> 3. Iteratively refining (use decomp constraints for filters).  
> 4. After each result, output JSON decision: {status: 'enough'|'more'|'clarify', reason: '...', next_tool_call: {... or null}}.  
> 5. When enough, stop tools and answer per decomp.intent (e.g., table for list).  
> Rules: At most N=5 tool calls (hard). Prefer narrow searches. No inventions; cite evidence.

### 5.2. Single-loop pseudo-code

Unchanged, but parse LLM response for JSON decisions before/after tool calls (e.g., if status=="enough", break).

---

## 6. Summary

- **Agentic RAG** = LLM drives retrieval, starting with decomposition for structure.
- Expose **enhanced tools** (incl. annotations, filters) for structured handling.
- **Orchestrator** (preferred) separates modes for control; **single-loop** for simplicity.
- Domain-agnostic; scales via buckets/filters.

---

## 7. Query Decomposition Step (Now Mode 0)

Moved earlier logically; unchanged except prompt update (route hints). Examples unchanged.

---

## 8. Example Execution Step Counts

Updated: Enforce hard cap at `max_tool_calls`; examples assume 3–5.

| Example | Decomposition LLM calls | Route LLM calls | Plan LLM calls | Review/Control LLM calls* | Tool calls | Compose LLM calls | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Contracts summary (7.4.1) | 1 | 1 | 1 | 3 | 2 | 1 | Route: hybrid; fallback not needed. |
| Transistor list (7.4.2) | 1 | 1 | 1 | 4 | 3 | 1 | Route: structured; aggregate for table. |
| Chemical solvents (7.4.3) | 1 | 1 | 1 | 5 | 4 | 1 | Route: hybrid; multi-bucket, cap at 5. |

*Includes final "enough".

---

## 9. Representative Retrieval Use Cases & Clarification Patterns

Inlined key scenarios/clarifications from prior doc.

### 9.1. Structured enumeration (annotations-first)
- **Query**: “List every tire purchase receipt.”
- **Flow**: Decomp → route=structured → annotations_search(predicates={properties.item_desc: {op:'~', value:'tire'}}) → aggregate if needed.
- **Clarify Overload** (if >100 hits): "Too many (1,042 receipts); filter by vendor/date? Or export CSV?"

### 9.2. Numeric lookup (key-value annotations)
- **Query**: “What is the minimum panel width for AlphaSolar models?”
- **Flow**: Route=structured → annotations_search({field:'width', op:'>=', value:0}) → min aggregate.
- **Clarify No/Low**: "Tried width annotations + semantic fallback (0 hits); try 'dimension' or upload date filter?"

### 9.3. Conceptual/manual deep dive (long-text)
- **Query**: “Explain the maintenance steps for battery cabinet B-120.”
- **Flow**: Route=long-text → semantic on summaries → chunk drill-down.
- **Clarify**: "Summaries off-topic; constrain to manual section/timeframe?"

### 9.4. Clarification Prompts
- **No/Low (P3)**: "Can't proceed ({{reason}}). Tried: {{attempts with counts}}. Hints: alternate terms (e.g., 'tyre'), ranges, or 'search all'."
- **Overload (P4)**: "Too many matches ({{reason}}). Tried: {{attempts with counts}}. Filter: vendor/2024? Stored for export."

Emit via templates when status="clarify".

---

*To download: Copy this content into a file named `agentic_rag_plan_revised.md` and save as Markdown.*