import React, { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useNavigate } from "react-router-dom";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try {
      return await res.json();
    } catch {
      // fall through to text
    }
  }
  const raw = await res.text();
  return { nonJson: true, raw };
}

function prettyBytes(n) {
  if (n == null || isNaN(n)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0;
  let val = n;
  while (val >= 1024 && idx < units.length - 1) {
    val /= 1024;
    idx++;
  }
  return `${val.toFixed(val < 10 && idx > 0 ? 1 : 0)} ${units[idx]}`;
}

function shortHash(hash) {
  if (!hash || typeof hash !== "string") return "";
  return hash.slice(0, 8);
}

const styles = {
  page: {
    display: "flex",
    flexWrap: "wrap",
    gap: 24,
    alignItems: "stretch",
  },
  chatCard: {
    flex: "2 1 540px",
    border: "1px solid rgba(148, 163, 184, 0.18)",
    borderRadius: 20,
    padding: "28px 32px",
    background: "rgba(13, 16, 24, 0.92)",
    boxShadow: "0 22px 45px rgba(2, 6, 23, 0.35)",
    display: "grid",
    gap: 20,
    alignContent: "start",
  },
  sideCard: {
    flex: "1 1 320px",
    border: "1px solid rgba(148, 163, 184, 0.18)",
    borderRadius: 20,
    padding: "28px 24px",
    background: "rgba(13, 16, 24, 0.88)",
    boxShadow: "0 18px 38px rgba(2, 6, 23, 0.32)",
    display: "grid",
    gap: 16,
    alignContent: "start",
    maxHeight: "calc(100vh - 220px)",
    overflow: "hidden",
  },
  sectionHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 16,
  },
  sectionTitle: {
    margin: 0,
    fontSize: 20,
    fontWeight: 600,
    letterSpacing: 0.25,
  },
  badge: {
    padding: "6px 12px",
    borderRadius: 999,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    fontSize: 12,
    color: "rgba(148, 163, 184, 0.85)",
    whiteSpace: "nowrap",
  },
  button: {
    font: "inherit",
    padding: "10px 18px",
    borderRadius: 14,
    border: "1px solid rgba(84, 105, 255, 0.45)",
    background: "linear-gradient(135deg, rgba(84, 105, 255, 0.18), rgba(84, 105, 255, 0.05))",
    color: "#c7d7ff",
    cursor: "pointer",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
  },
  subtleButton: {
    font: "inherit",
    padding: "8px 14px",
    borderRadius: 12,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(15, 17, 23, 0.6)",
    color: "rgba(226, 232, 240, 0.9)",
    cursor: "pointer",
  },
  input: {
    font: "inherit",
    padding: "12px 16px",
    borderRadius: 14,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(9, 11, 18, 0.82)",
    color: "inherit",
    flex: 1,
    minWidth: 0,
  },
  muted: {
    fontSize: 13,
    color: "rgba(148, 163, 184, 0.78)",
  },
  messages: {
    minHeight: 280,
    maxHeight: "55vh",
    overflow: "auto",
    border: "1px solid rgba(148, 163, 184, 0.12)",
    borderRadius: 16,
    padding: 18,
    background: "rgba(9, 11, 18, 0.78)",
    whiteSpace: "pre-wrap",
  },
  answer: {
    padding: 16,
    borderRadius: 14,
    background: "rgba(84, 105, 255, 0.18)",
    lineHeight: 1.55,
  },
  docs: {
    overflow: "auto",
    border: "1px solid rgba(148, 163, 184, 0.12)",
    borderRadius: 14,
    padding: 16,
    background: "rgba(9, 11, 18, 0.72)",
  },
  listItem: {
    padding: "10px 8px",
    borderRadius: 10,
    marginBottom: 6,
    background: "rgba(23, 25, 35, 0.7)",
    border: "1px solid rgba(148, 163, 184, 0.08)",
  },
  kbd: {
    fontFamily:
      'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
};

export default function ChatPage({ onAskingChange, warmupApi, llmReady, documents = [] }) {
  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [warmingUp, setWarmingUp] = useState(false);
  const [warmedUp, setWarmedUp] = useState(false);
  const warmupAttemptRef = useRef(false);
  const navigate = useNavigate();

  const systemDocs = useMemo(() => (Array.isArray(documents) ? documents : []), [documents]);
  const displayDocs = docs.length ? docs : systemDocs;

  const api = {
    docs: "/api/docs",
    ask: "/api/ask",
  };

  // Load document list (only called when component mounts and documents are ready)
  const refreshDocs = async () => {
    setDocsLoading(true);
    try {
      const res = await fetch(api.docs);
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`);
      setDocs(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error(e);
      setDocs([]);
    } finally {
      setDocsLoading(false);
    }
  };

  useEffect(() => {
    void refreshDocs();
  }, []);

  useEffect(() => {
    if (systemDocs.length > 0) {
      setDocs((existing) => {
        if (existing.length === 0) return systemDocs;
        if (existing.length !== systemDocs.length) return systemDocs;
        return existing;
      });
    }
  }, [systemDocs]);

  // Communicate asking status to parent
  useEffect(() => {
    if (onAskingChange) {
      onAskingChange(asking || warmingUp);
    }
  }, [asking, warmingUp, onAskingChange]);

  const performWarmup = useCallback(async () => {
    if (!warmupApi || warmedUp || llmReady) return;
    if (warmupAttemptRef.current) return;

    warmupAttemptRef.current = true;
    setWarmingUp(true);
    try {
      const res = await fetch(warmupApi, { method: "POST" });
      const data = await readJsonSafe(res);
      if (res.ok && data.warmup_complete) {
        setWarmedUp(true);
      } else {
        console.warn("Warmup failed:", data.error || "Unknown error");
        warmupAttemptRef.current = false;
      }
    } catch (e) {
      console.error("Warmup error:", e);
      warmupAttemptRef.current = false;
    } finally {
      setWarmingUp(false);
    }
  }, [warmupApi, warmedUp, llmReady]);

  // Separate effect for warmup to avoid infinite loops
  useEffect(() => {
    if (!warmedUp && !llmReady && warmupApi) {
      void performWarmup();
    }
  }, [warmedUp, llmReady, warmupApi, performWarmup]);

  useEffect(() => {
    if (llmReady) {
      setWarmedUp(true);
      setWarmingUp(false);
    }
  }, [llmReady]);

  const handleAsk = async () => {
    const q = query.trim();
    if (!q) return;
    setAsking(true);
    setAnswer("");
    setSources([]);
    try {
      const res = await fetch(api.ask, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      setAnswer(data?.answer || "");
      setSources(Array.isArray(data?.sources) ? data.sources : []);
    } catch (e) {
      console.error(e);
      setAnswer(`Error: ${e.message || String(e)}`);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div style={styles.page}>
      <section style={styles.chatCard}>
        <div style={styles.sectionHeader}>
          <h2 style={styles.sectionTitle}>Chat Workspace</h2>
          <button
            onClick={() => navigate("/ingest")}
            style={{ ...styles.subtleButton, padding: "8px 16px" }}
          >
            Back to Ingestion
          </button>
        </div>

        {warmingUp && !llmReady && (
          <div style={{
            ...styles.muted,
            background: "rgba(250, 204, 21, 0.12)",
            border: "1px solid rgba(250, 204, 21, 0.24)",
            padding: 12,
            borderRadius: 12,
          }}>
            🔥 Warming up LM Studio model... Please wait.
          </div>
        )}

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <input
            type="text"
            placeholder={warmingUp && !warmedUp ? "Warming up model..." : "Ask a question about your docs..."}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !(warmingUp && !warmedUp)) handleAsk();
            }}
            disabled={warmingUp && !warmedUp}
            style={{
              ...styles.input,
              opacity: warmingUp && !warmedUp ? 0.6 : 1,
            }}
          />
          <button
            onClick={handleAsk}
            disabled={asking || !query.trim() || (warmingUp && !warmedUp)}
            style={{
              ...styles.button,
              opacity: (asking || !query.trim() || (warmingUp && !warmedUp)) ? 0.6 : 1,
              cursor: (asking || !query.trim() || (warmingUp && !warmedUp)) ? "not-allowed" : "pointer",
            }}
          >
            {asking ? "Asking..." : warmingUp && !warmedUp ? "Warming up..." : "Ask"}
          </button>
        </div>

        <div style={styles.messages}>
          {warmingUp && !warmedUp ? (
            <div style={styles.muted}>
              🔥 Initializing LM Studio model for first use. This may take a moment...
            </div>
          ) : !answer ? (
            <div style={styles.muted}>
              {warmedUp ? "✅ Ready! Ask a question to get started." : "Ask a question to get started."}
            </div>
          ) : (
            <div>
              <div style={styles.answer}>{answer}</div>
              {!!sources?.length && (
                <div style={{ ...styles.muted, marginTop: 12 }}>
                  <div style={{ fontWeight: 600, marginBottom: 6 }}>Sources</div>
                  <ol style={{ margin: 0, paddingLeft: 18 }}>
                    {sources.map((s, i) => (
                      <li key={i} style={{ marginBottom: 6 }}>
                        <code style={styles.kbd}>{summarizeSource(s)}</code>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      <section style={styles.sideCard}>
        <div style={styles.sectionHeader}>
          <h2 style={{ ...styles.sectionTitle, fontSize: 18 }}>Ingested Documents</h2>
          <span style={styles.badge}>
            {displayDocs.length} file{displayDocs.length === 1 ? "" : "s"}
          </span>
        </div>
        <div style={{ ...styles.docs, ...(displayDocs.length ? {} : styles.muted) }}>
          {displayDocs.length === 0 ? (
            <div>No documents yet. Head back to ingestion to add some.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {displayDocs.map((d) => (
                <li key={d.hash || d.stored_name || d.name} style={styles.listItem}>
                  <div title={d.path}>
                    <strong>{d.name}</strong>{" "}
                    <span style={styles.muted}>
                      {`${prettyBytes(d.size)}${d.hash ? ` · ${shortHash(d.hash)}` : ""}`}
                    </span>
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

function summarizeSource(s) {
  try {
    if (!s) return "unknown";
    if (typeof s === "string") return s;
    if (s.path) return s.path;
    if (s.name) return s.name;
    if (s.file) return s.file;
    return JSON.stringify(s);
  } catch {
    return "unknown";
  }
}
