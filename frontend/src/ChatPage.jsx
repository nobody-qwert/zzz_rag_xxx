import React, { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) { try { return await res.json(); } catch {} }
  const raw = await res.text();
  return { nonJson: true, raw };
}

function prettyBytes(n) { if (n == null || isNaN(n)) return "-"; const u=["B","KB","MB","GB","TB"]; let i=0,v=n; while(v>=1024&&i<u.length-1){v/=1024;i++;} return `${v.toFixed(v<10&&i>0?1:0)} ${u[i]}`; }
function shortHash(h){ return (h||"").slice(0,8); }
const createMessageId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
function mergeSources(existing = [], incoming = []) {
  const base = Array.isArray(existing) ? existing : [];
  const extra = Array.isArray(incoming) ? incoming : [];
  const combined = [...base, ...extra];
  const deduped = new Map();
  combined.forEach((item, idx) => {
    if (!item) return;
    const key = item.chunk_id || `${item.doc_hash || "na"}-${item.order_index ?? idx}`;
    if (!deduped.has(key)) deduped.set(key, item);
  });
  return Array.from(deduped.values());
}

const styles = {
  page: { display: "flex", flexWrap: "wrap", gap: 18, alignItems: "stretch" },
  chatCard: { flex: "2 1 540px", border: "1px solid rgba(148, 163, 184, 0.18)", borderRadius: 0, padding: "16px 18px", background: "rgba(13, 16, 24, 0.92)", boxShadow: "0 16px 28px rgba(2, 6, 23, 0.32)", display: "grid", gap: 12, alignContent: "start" },
  sideCard: { flex: "1 1 320px", border: "1px solid rgba(148, 163, 184, 0.18)", borderRadius: 0, padding: "16px 16px", background: "rgba(13, 16, 24, 0.88)", boxShadow: "0 12px 24px rgba(2, 6, 23, 0.3)", display: "grid", gap: 10, alignContent: "start", maxHeight: "calc(100vh - 220px)", overflow: "hidden" },
  sectionHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" },
  sectionTitle: { margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 0.25 },
  badge: { padding: "6px 12px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.35)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", whiteSpace: "nowrap" },
  button: { font: "inherit", fontSize: 14, padding: "6px 12px", borderRadius: 0, border: "1px solid rgba(84, 105, 255, 0.45)", background: "linear-gradient(135deg, rgba(84, 105, 255, 0.18), rgba(84, 105, 255, 0.05))", color: "#c7d7ff", cursor: "pointer" },
  subtleButton: { font: "inherit", fontSize: 13, padding: "4px 10px", borderRadius: 0, border: "1px solid rgba(148, 163, 184, 0.35)", background: "rgba(15, 17, 23, 0.6)", color: "rgba(226, 232, 240, 0.9)", cursor: "pointer" },
  input: { font: "inherit", padding: "8px 12px", borderRadius: 0, border: "1px solid rgba(148, 163, 184, 0.35)", background: "rgba(9, 11, 18, 0.82)", color: "inherit", flex: 1, minWidth: 0 },
  muted: { fontSize: 13, color: "rgba(148, 163, 184, 0.78)" },
  messages: { minHeight: 240, maxHeight: "48vh", overflow: "auto", border: "1px solid rgba(148, 163, 184, 0.12)", borderRadius: 0, padding: 12, background: "rgba(9, 11, 18, 0.78)", whiteSpace: "pre-wrap", display: "flex", flexDirection: "column", gap: 8 },
  messageList: { display: "flex", flexDirection: "column", gap: 8 },
  userBubble: { alignSelf: "flex-end", background: "rgba(59, 130, 246, 0.18)", border: "1px solid rgba(59, 130, 246, 0.35)", borderRadius: 0, padding: 10, maxWidth: "85%" },
  assistantBubble: { alignSelf: "flex-start", background: "rgba(84, 105, 255, 0.12)", border: "1px solid rgba(148, 163, 184, 0.25)", borderRadius: 0, padding: 10, maxWidth: "95%", lineHeight: 1.5 },
  errorBubble: { alignSelf: "flex-start", background: "rgba(239, 68, 68, 0.12)", border: "1px solid rgba(239, 68, 68, 0.4)", borderRadius: 0, padding: 10, maxWidth: "95%" },
  messageRole: { fontSize: 12, textTransform: "uppercase", letterSpacing: 0.8, color: "rgba(148, 163, 184, 0.75)", marginBottom: 4 },
  sourcesBlock: { fontSize: 12, color: "rgba(148, 163, 184, 0.78)", marginTop: 10 },
  docs: { overflow: "auto", border: "1px solid rgba(148, 163, 184, 0.12)", borderRadius: 0, padding: 12, background: "rgba(9, 11, 18, 0.72)" },
  listItem: { padding: "10px 8px", borderRadius: 0, marginBottom: 6, background: "rgba(23, 25, 35, 0.7)", border: "1px solid rgba(148, 163, 184, 0.08)" },
  contextBadge: { padding: "8px 14px", borderRadius: 14, border: "1px solid rgba(148, 163, 184, 0.35)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", background: "rgba(12, 14, 22, 0.85)" },
  contextLabel: { fontSize: 12, color: "rgba(148, 163, 184, 0.8)" },
  kbd: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace' },
};

export default function ChatPage({ onAskingChange, warmupApi, llmReady, documents = [] }) {
  const defaultContextStats = useMemo(() => ({ used: 0, limit: 10000, truncated: false, ratio: 0 }), []);
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [warmingUp, setWarmingUp] = useState(false);
  const [warmedUp, setWarmedUp] = useState(false);
  const [contextStats, setContextStats] = useState(() => ({ ...defaultContextStats }));
  const [pendingFollowUp, setPendingFollowUp] = useState(null);
  const [continuing, setContinuing] = useState(false);
  const warmupAttemptRef = useRef(false);
  const messagesBodyRef = useRef(null);
  const navigate = useNavigate();

  const systemDocs = useMemo(() => (Array.isArray(documents) ? documents : []), [documents]);
  const displayDocs = docs.length ? docs : systemDocs;

  const api = { docs: "/api/documents", ask: "/api/ask" };

  const refreshDocs = async () => {
    setDocsLoading(true);
    try { const res = await fetch(api.docs); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`); setDocs(Array.isArray(data) ? data : []); }
    catch (e) { setDocs([]); }
    finally { setDocsLoading(false); }
  };

  useEffect(() => { void refreshDocs(); }, []);
  useEffect(() => { if (systemDocs.length > 0) { setDocs((existing) => { if (existing.length === 0) return systemDocs; if (existing.length !== systemDocs.length) return systemDocs; return existing; }); } }, [systemDocs]);

  useEffect(() => { if (onAskingChange) onAskingChange(asking || warmingUp || continuing); }, [asking, warmingUp, continuing, onAskingChange]);
  useEffect(() => {
    const el = messagesBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const performWarmup = useCallback(async () => {
    if (!warmupApi || warmedUp || llmReady) return; if (warmupAttemptRef.current) return; warmupAttemptRef.current = true; setWarmingUp(true);
    try { const res = await fetch(warmupApi, { method: "POST" }); const data = await readJsonSafe(res); if (res.ok && data.warmup_complete) { setWarmedUp(true); } else { warmupAttemptRef.current = false; } }
    catch (e) { warmupAttemptRef.current = false; }
    finally { setWarmingUp(false); }
  }, [warmupApi, warmedUp, llmReady]);

  useEffect(() => { if (!warmedUp && !llmReady && warmupApi) { void performWarmup(); } }, [warmedUp, llmReady, warmupApi, performWarmup]);
  useEffect(() => { if (llmReady) { setWarmedUp(true); setWarmingUp(false); } }, [llmReady]);

  const handleResetConversation = () => {
    setConversationId(null);
    setMessages([]);
    setContextStats({ ...defaultContextStats });
    setPendingFollowUp(null);
    setContinuing(false);
  };

  const handleAsk = async () => {
    const trimmed = query.trim();
    if (!trimmed || (warmingUp && !warmedUp) || pendingFollowUp || continuing) return;
    setAsking(true);
    const userId = createMessageId();
    setMessages((prev) => [...prev, { id: userId, role: "user", content: trimmed }]);
    setQuery("");
    try {
      const payload = { query: trimmed };
      if (conversationId) payload.conversation_id = conversationId;
      const res = await fetch(api.ask, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const nextConversationId = data?.conversation_id || conversationId;
      if (nextConversationId) setConversationId(nextConversationId);
      const needsFollowUp = !!data?.needs_follow_up;
      const assistantId = createMessageId();
      const assistantMessage = {
        id: assistantId,
        role: "assistant",
        content: data?.answer || "",
        sources: Array.isArray(data?.sources) ? data.sources : [],
        hideSources: needsFollowUp,
        pendingFollowUp: needsFollowUp,
        finishReason: data?.finish_reason || null,
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setContextStats({
        used: typeof data?.context_tokens_used === "number" ? data.context_tokens_used : 0,
        limit: typeof data?.context_window_limit === "number" ? data.context_window_limit : defaultContextStats.limit,
        truncated: !!data?.context_truncated,
        ratio: typeof data?.context_usage === "number" ? data.context_usage : 0,
      });
      setPendingFollowUp(needsFollowUp ? { conversationId: nextConversationId, messageId: assistantId } : null);
    } catch (e) {
      setMessages((prev) => [...prev, { id: createMessageId(), role: "assistant", content: `Error: ${e.message || String(e)}`, error: true }]);
    } finally {
      setAsking(false);
    }
  };

  const handleContinueResponse = async () => {
    if (!pendingFollowUp || continuing) return;
    const activeConversationId = pendingFollowUp.conversationId || conversationId;
    if (!activeConversationId) return;
    setContinuing(true);
    try {
      const payload = { continue_last: true, conversation_id: activeConversationId };
      const res = await fetch(api.ask, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      const nextConversationId = data?.conversation_id || activeConversationId;
      if (nextConversationId) setConversationId(nextConversationId);
      const addition = data?.answer || "";
      const needsFollowUp = !!data?.needs_follow_up;
      setMessages((prev) => prev.map((msg) => {
        if (msg.id !== pendingFollowUp.messageId) return msg;
        const mergedSources = mergeSources(msg.sources, data?.sources);
        const needsJoiner = Boolean(msg.content && addition && !msg.content.endsWith("\n") && !addition.startsWith("\n"));
        return {
          ...msg,
          content: msg.content + (addition ? `${needsJoiner ? "\n" : ""}${addition}` : ""),
          sources: mergedSources,
          hideSources: needsFollowUp,
          pendingFollowUp: needsFollowUp,
          finishReason: data?.finish_reason || null,
        };
      }));
      setContextStats({
        used: typeof data?.context_tokens_used === "number" ? data.context_tokens_used : 0,
        limit: typeof data?.context_window_limit === "number" ? data.context_window_limit : defaultContextStats.limit,
        truncated: !!data?.context_truncated,
        ratio: typeof data?.context_usage === "number" ? data.context_usage : 0,
      });
      setPendingFollowUp(needsFollowUp ? { conversationId: nextConversationId, messageId: pendingFollowUp.messageId } : null);
    } catch (e) {
      setMessages((prev) => [...prev, { id: createMessageId(), role: "assistant", content: `Error continuing response: ${e.message || String(e)}`, error: true }]);
    } finally {
      setContinuing(false);
    }
  };

  const handleAbortContinuation = () => {
    if (!pendingFollowUp || continuing) return;
    setMessages((prev) => prev.map((msg) => (msg.id === pendingFollowUp.messageId ? { ...msg, pendingFollowUp: false, hideSources: false, aborted: true } : msg)));
    setPendingFollowUp(null);
  };

  return (
    <div style={styles.page}>
      <section style={styles.chatCard}>
        <div style={styles.sectionHeader}>
          <h2 style={styles.sectionTitle}>Chat Workspace</h2>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <button onClick={() => navigate("/ingest")} style={{ ...styles.subtleButton, padding: "8px 16px" }}>Back to Ingestion</button>
            <button onClick={handleResetConversation} disabled={!messages.length && !conversationId} style={{ ...styles.subtleButton, padding: "8px 16px", opacity: (!messages.length && !conversationId) ? 0.5 : 1 }}>Reset Chat</button>
            <ContextIndicator stats={contextStats} defaultLimit={defaultContextStats.limit} />
          </div>
        </div>

        {warmingUp && !llmReady && (
          <div style={{ ...styles.muted, background: "rgba(250, 204, 21, 0.12)", border: "1px solid rgba(250, 204, 21, 0.24)", padding: 12, borderRadius: 0 }}>
            🔥 Warming up LM Studio model... Please wait.
          </div>
        )}

        <div style={styles.messages} ref={messagesBodyRef}>
          {warmingUp && !warmedUp ? (
            <div style={styles.muted}>🔥 Initializing LM Studio model for first use. This may take a moment...</div>
          ) : messages.length === 0 ? (
            <div style={styles.muted}>{warmedUp ? "✅ Ready! Ask a question to get started." : "Ask a question to get started."}</div>
          ) : (
            <div style={styles.messageList}>
              {messages.map((m, i) => (
                <div key={m.id || `${m.role}-${i}-${Math.abs(m.content?.length || 0)}`} style={m.error ? styles.errorBubble : m.role === "user" ? styles.userBubble : styles.assistantBubble}>
                  <div style={styles.messageRole}>{m.role === "user" ? "You" : "Assistant"}</div>
                  <div>{m.content}</div>
                  {m.role === "assistant" && m.pendingFollowUp && !m.error && (
                    <div style={{ ...styles.muted, marginTop: 8 }}>
                      Response paused because it reached the token limit. Continue or abort to proceed.
                    </div>
                  )}
                  {m.role === "assistant" && m.aborted && !m.error && (
                    <div style={{ ...styles.muted, marginTop: 8 }}>Generation aborted. You can ask another question.</div>
                  )}
                  {m.role === "assistant" && Array.isArray(m.sources) && m.sources.length > 0 && !m.hideSources && (
                    <div style={styles.sourcesBlock}>
                      <div style={{ fontWeight: 600, marginBottom: 4 }}>Sources</div>
                      <ol style={{ margin: 0, paddingLeft: 18 }}>
                        {m.sources.map((s, idx) => (
                          <li key={`${s.chunk_id || s.doc_hash || idx}-${idx}`} style={{ marginBottom: 6 }}>
                            <div>
                              <strong style={{ color: "rgba(226, 232, 240, 0.95)" }}>{s.document_name || "unknown"}</strong>
                              {s.total_chunks > 0 && (
                                <span style={{ marginLeft: 8, fontSize: 12 }}>
                                  chunk {s.order_index + 1}/{s.total_chunks}
                                </span>
                              )}
                              {typeof s.score === "number" && (
                                <span style={{ marginLeft: 8, fontSize: 12, color: "rgba(148, 163, 184, 0.7)" }}>
                                  similarity: {(s.score * 100).toFixed(1)}%
                                </span>
                              )}
                            </div>
                            {s.chunk_text_preview && (
                              <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.6)", fontStyle: "italic", marginTop: 2 }}>
                                "{s.chunk_text_preview}..."
                              </div>
                            )}
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {pendingFollowUp && (
          <div style={{ border: "1px solid rgba(248, 250, 252, 0.12)", borderRadius: 0, padding: 12, background: "rgba(24, 27, 40, 0.7)", display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ ...styles.muted, fontSize: 13 }}>
              The assistant stopped early (finish reason: {messages.find((m) => m.id === pendingFollowUp.messageId)?.finishReason || "unknown"}).
              Choose <strong>Continue</strong> to keep generating or <strong>Abort</strong> to accept the current response.
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button onClick={handleAbortContinuation} disabled={continuing} style={{ ...styles.subtleButton, padding: "10px 18px", opacity: continuing ? 0.6 : 1 }}>
                Abort
              </button>
              <button onClick={handleContinueResponse} disabled={continuing} style={{ ...styles.button, opacity: continuing ? 0.6 : 1 }}>
                {continuing ? "Continuing..." : "Continue"}
              </button>
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <input type="text" placeholder={warmingUp && !warmedUp ? "Warming up model..." : "Ask a question about your docs..."} value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !(warmingUp && !warmedUp)) handleAsk(); }} disabled={(warmingUp && !warmedUp) || pendingFollowUp || continuing} style={{ ...styles.input, opacity: (warmingUp && !warmedUp) || pendingFollowUp || continuing ? 0.6 : 1 }} />
          <button onClick={handleAsk} disabled={asking || !query.trim() || (warmingUp && !warmedUp) || pendingFollowUp || continuing} style={{ ...styles.button, minWidth: 70, letterSpacing: 0.3, opacity: (asking || !query.trim() || (warmingUp && !warmedUp) || pendingFollowUp || continuing) ? 0.6 : 1 }}>
            {asking ? "Asking..." : warmingUp && !warmedUp ? "Warming up..." : "Ask"}
          </button>
        </div>
      </section>

      <section style={styles.sideCard}>
        <div style={styles.sectionHeader}>
          <h2 style={{ ...styles.sectionTitle, fontSize: 18 }}>Ingested Documents</h2>
          <span style={styles.badge}>{displayDocs.length} file{displayDocs.length === 1 ? "" : "s"}</span>
        </div>
        {docsLoading && <span style={{ ...styles.muted, fontSize: 11 }}>Refreshing...</span>}
        <div style={{ ...styles.docs, ...(displayDocs.length ? {} : styles.muted) }}>
          {displayDocs.length === 0 ? (
            <div>No documents yet. Head back to ingestion to add some.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {displayDocs.map((d) => (
                <li key={d.hash || d.stored_name || d.name} style={styles.listItem}>
                  <div title={d.path}>
                    <strong>{d.name}</strong>{" "}
                    <span style={styles.muted}>{`${prettyBytes(d.size)}${d.hash ? ` · ${shortHash(d.hash)}` : ""}`}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}

function ContextIndicator({ stats, defaultLimit }) {
  const limit = stats?.limit || defaultLimit || 10000;
  const used = stats?.used || 0;
  const percent = limit ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  const accent = percent >= 85 ? "rgba(248, 113, 113, 0.95)" : percent >= 60 ? "rgba(251, 191, 36, 0.95)" : "rgba(16, 185, 129, 0.95)";
  const badgeStyle = {
    ...styles.contextBadge,
    borderColor: accent,
    color: accent,
    background: stats?.truncated ? "rgba(127, 29, 29, 0.25)" : styles.contextBadge.background,
    boxShadow: stats?.truncated ? "0 0 18px rgba(248, 113, 113, 0.55)" : "none",
  };
  return (
    <span style={badgeStyle} title="Prompt tokens used / max window">
      Context {used}/{limit} ({percent}%)
    </span>
  );
}
