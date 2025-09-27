import React, { useEffect, useMemo, useRef, useState } from "react";

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

function Section({ title, right, children }) {
  return (
    <section style={styles.card}>
      <div style={{ ...styles.row, justifyContent: "space-between", marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 16 }}>{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [file, setFile] = useState(null);

  const [query, setQuery] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const messagesRef = useRef(null);

  const api = useMemo(() => {
    // In local dev, Vite proxy forwards /api -> http://localhost:8000
    // In Docker, nginx inside frontend container proxies /api -> http://rag-backend:8000
    return {
      ingest: "/api/ingest",
      docs: "/api/docs",
      ask: "/api/ask",
    };
  }, []);

  useEffect(() => {
    void refreshDocs();
  }, []);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [answer, sources]);

  async function refreshDocs() {
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
  }

  async function handleUpload() {
    if (!file) {
      setUploadStatus("Select a file first.");
      return;
    }
    setUploading(true);
    setUploadStatus("Uploading...");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(api.ingest, {
        method: "POST",
        body: form,
      });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      if (data?.indexed) {
        setUploadStatus(`Indexed: ${data.file || file.name}`);
      } else {
        setUploadStatus(`Uploaded (indexing may have failed): ${data.file || file.name}`);
        if (data?.error) console.warn("Ingest error:", data.error);
      }
      setFile(null);
      await refreshDocs();
    } catch (e) {
      console.error(e);
      setUploadStatus(`Upload failed: ${e.message || String(e)}`);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(""), 4000);
    }
  }

  async function handleAsk() {
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
  }

  return (
    <div>
      <header style={styles.header}>
        <h1 style={{ margin: 0, fontSize: 16 }}>RAG-Anything Demo</h1>
        <span style={styles.muted}>Upload documents, then ask questions</span>
      </header>

      <main style={styles.main}>
        <div style={styles.stack}>
          <Section
            title="Upload / Ingest"
            right={
              <button onClick={handleUpload} disabled={uploading || !file} style={styles.button}>
                {uploading ? "Ingesting..." : "Ingest"}
              </button>
            }
          >
            <div style={styles.row}>
              <input
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                style={styles.input}
              />
            </div>
            <div style={styles.muted}>{uploadStatus}</div>
          </Section>

          <Section
            title="Documents"
            right={
              <button onClick={refreshDocs} disabled={docsLoading} style={styles.button}>
                {docsLoading ? "Refreshing..." : "Refresh"}
              </button>
            }
          >
            <div style={{ ...styles.docs, ...(docs.length ? {} : styles.muted) }}>
              {docs.length === 0 ? (
                <div>No documents yet.</div>
              ) : (
                <ul style={{ margin: 0, paddingLeft: 16 }}>
                  {docs.map((d) => (
                    <li key={d.name}>
                      <span title={d.path}>
                        {d.name} <span style={styles.muted}>({prettyBytes(d.size)})</span>
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </Section>
        </div>

        <Section title="Chat">
          <div style={styles.row}>
            <input
              type="text"
              placeholder="Ask a question about your docs..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAsk();
              }}
              style={{ ...styles.input, flex: 1 }}
            />
            <button onClick={handleAsk} disabled={asking || !query.trim()} style={styles.button}>
              {asking ? "Asking..." : "Ask"}
            </button>
          </div>
          <div ref={messagesRef} style={styles.messages}>
            {!answer ? (
              <div style={styles.muted}>No messages yet.</div>
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
        </Section>
      </main>
    </div>
  );
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

const styles = {
  header: {
    padding: "12px 16px",
    borderBottom: "1px solid #4443",
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  main: {
    display: "grid",
    gap: 16,
    gridTemplateColumns: "320px 1fr",
    padding: 16,
  },
  card: {
    border: "1px solid #4443",
    borderRadius: 8,
    padding: 12,
    background: "transparent",
  },
  stack: { display: "grid", gap: 8 },
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
  messages: {
    minHeight: 240,
    maxHeight: "60vh",
    overflow: "auto",
    border: "1px dashed #4443",
    borderRadius: 6,
    padding: 8,
    whiteSpace: "pre-wrap",
  },
  muted: { opacity: 0.7, fontSize: 12 },
  answer: { padding: 8, borderRadius: 6, background: "#4a90e22a" },
  kbd: {
    fontFamily:
      'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
};
