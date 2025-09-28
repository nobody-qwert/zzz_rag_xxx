import React, { useEffect, useState, useCallback, useRef } from "react";
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
  card: {
    border: "1px solid #4443",
    borderRadius: 8,
    padding: 12,
    background: "transparent",
  },
  row: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" },
  button: {
    font: "inherit",
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid #4443",
    background: "transparent",
    cursor: "pointer",
  },
  input: {
    font: "inherit",
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid #4443",
    background: "transparent",
  },
  docs: {
    maxHeight: "40vh",
    overflow: "auto",
    border: "1px dashed #4443",
    borderRadius: 6,
    padding: 8,
  },
  muted: { opacity: 0.7, fontSize: 12 },
  messages: {
    minHeight: 240,
    maxHeight: "60vh",
    overflow: "auto",
    border: "1px dashed #4443",
    borderRadius: 6,
    padding: 8,
    whiteSpace: "pre-wrap",
  },
  answer: { padding: 8, borderRadius: 6, background: "#4a90e22a" },
  kbd: {
    fontFamily:
      'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
};

export default function ChatPage({ onAskingChange, warmupApi, llmReady }) {
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
    <div style={styles.card}>
      <section>
        <div style={{ ...styles.row, justifyContent: "space-between", marginBottom: 8 }}>
          <h3 style={{ margin: 0, fontSize: 16 }}>Chat</h3>
          <button onClick={() => navigate("/ingest")} style={styles.button}>
            Back to Ingestion
          </button>
        </div>
        {warmingUp && !llmReady && (
          <div style={{ ...styles.muted, marginBottom: 8, padding: 8, background: "#fff3cd", borderRadius: 4 }}>
            🔥 Warming up LM Studio model... Please wait.
          </div>
        )}

        <div style={styles.row}>
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
              flex: 1,
              opacity: warmingUp && !warmedUp ? 0.6 : 1
            }}
          />
          <button 
            onClick={handleAsk} 
            disabled={asking || !query.trim() || (warmingUp && !warmedUp)} 
            style={{
              ...styles.button,
              opacity: (asking || !query.trim() || (warmingUp && !warmedUp)) ? 0.6 : 1
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
                <div style={{ ...styles.muted, marginTop: 8 }}>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>Sources</div>
                  <ol style={{ margin: 0, paddingLeft: 16 }}>
                    {sources.map((s, i) => (
                      <li key={i} style={{ marginBottom: 4 }}>
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

      <section style={{ marginTop: 16 }}>
        <h3 style={{ margin: "0 0 8px 0", fontSize: 16 }}>Ingested Documents</h3>
        <div style={{ ...styles.docs, ...(docs.length ? {} : styles.muted) }}>
          {docs.length === 0 ? (
            <div>No documents yet.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {docs.map((d) => (
                <li key={d.hash || d.stored_name || d.name}>
                  <span title={d.path}>
                    {d.name}{" "}
                    <span style={styles.muted}>
                      {`${prettyBytes(d.size)}${d.hash ? ` · ${shortHash(d.hash)}` : ""}`}
                    </span>
                  </span>
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
