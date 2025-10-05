import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try { return await res.json(); } catch {}
  }
  const raw = await res.text();
  return { nonJson: true, raw };
}

function prettyBytes(n) {
  if (n == null || isNaN(n)) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let idx = 0; let val = n;
  while (val >= 1024 && idx < units.length - 1) { val /= 1024; idx++; }
  return `${val.toFixed(val < 10 && idx > 0 ? 1 : 0)} ${units[idx]}`;
}

function shortHash(h) { return (h || "").slice(0, 8); }
function formatDate(v) { try { const d = new Date(v); return isNaN(d) ? "" : d.toLocaleString(); } catch { return ""; } }

const styles = {
  page: { display: "grid", gap: 24, gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", alignItems: "start" },
  leftColumn: { display: "grid", gap: 24 },
  card: { border: "1px solid rgba(148, 163, 184, 0.18)", borderRadius: 18, padding: "24px 28px", background: "rgba(13, 16, 24, 0.9)", boxShadow: "0 22px 45px rgba(2, 6, 23, 0.35)" },
  sectionHeader: { display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, marginBottom: 16 },
  sectionTitle: { margin: 0, fontSize: 18, fontWeight: 600, letterSpacing: 0.2 },
  badge: { padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.4)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", whiteSpace: "nowrap" },
  row: { display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" },
  button: { font: "inherit", padding: "10px 16px", borderRadius: 12, border: "1px solid rgba(84, 105, 255, 0.45)", background: "linear-gradient(135deg, rgba(84, 105, 255, 0.18), rgba(84, 105, 255, 0.05))", color: "#c7d7ff", cursor: "pointer" },
  subtleButton: { font: "inherit", padding: "6px 10px", borderRadius: 10, border: "1px solid rgba(148, 163, 184, 0.35)", background: "rgba(15, 17, 23, 0.6)", color: "rgba(226, 232, 240, 0.88)", cursor: "pointer" },
  input: { font: "inherit", padding: "10px 14px", borderRadius: 12, border: "1px solid rgba(148, 163, 184, 0.35)", background: "rgba(9, 11, 18, 0.8)", color: "inherit", minWidth: 0 },
  feedback: { marginTop: 10, fontSize: 13, color: "rgba(148, 163, 184, 0.85)" },
  docs: { maxHeight: "55vh", overflow: "auto", border: "1px solid rgba(148, 163, 184, 0.12)", borderRadius: 14, padding: 16, background: "rgba(9, 11, 18, 0.75)" },
  listItem: { padding: "12px 10px", borderRadius: 10, background: "rgba(23, 25, 35, 0.75)", border: "1px solid rgba(148, 163, 184, 0.08)", marginBottom: 8 },
  muted: { opacity: 0.75, fontSize: 13, color: "rgba(148, 163, 184, 0.8)" },
  error: { fontSize: 13, color: "#ff8f8f", marginTop: 6 },
};

export default function IngestPage({ systemStatus = {} }) {
  const api = useMemo(() => ({
    ingest: "/api/ingest",
    docs: "/api/documents",
    status: (jobId) => `/api/status/${jobId}`,
    retry: (hash) => `/api/ingest/${hash}/retry`,
    previewText: (hash, maxChars = 2000, parser = "mineru") => `/api/debug/parsed_text/${hash}?parser=${parser}&max_chars=${maxChars}`,
  }), []);

  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [jobId, setJobId] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [retryingHash, setRetryingHash] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [preview, setPreview] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [previewInfo, setPreviewInfo] = useState(null);
  const [previewMaxChars, setPreviewMaxChars] = useState(2000);
  const [parser, setParser] = useState("mineru");

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try { const res = await fetch(api.docs); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`); setDocs(Array.isArray(data) ? data : []); }
    catch { setDocs([]); }
    finally { setDocsLoading(false); }
  }, [api.docs]);

  useEffect(() => { void refreshDocs(); }, [refreshDocs]);

  const systemDocs = useMemo(() => (Array.isArray(systemStatus.documents) ? systemStatus.documents : []), [systemStatus.documents]);
  const displayDocs = docs.length ? docs : systemDocs;

  const handleUpload = async () => {
    if (!file) { setUploadStatus("Select a file first."); return; }
    setUploading(true); setUploadStatus("Uploading...");
    try {
      const form = new FormData(); form.append("file", file);
      const res = await fetch(api.ingest, { method: "POST", body: form });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      if (data.status === "skipped") { setUploadStatus(`Already ingested (hash ${shortHash(data.hash)})`); setProcessing(false); setJobId(null); return; }
      if (!data.job_id) throw new Error("Upload did not return a job identifier");
      setJobId(data.job_id); setProcessing(true); setUploadStatus(`Queued (job ${data.job_id})`);
    } catch (e) { setUploadStatus(`Upload failed: ${e.message || String(e)}`); }
    finally { setUploading(false); setFile(null); }
  };

  const handleRetry = useCallback(async (hash) => {
    if (!hash) return; setRetryingHash(hash); setUploadStatus(`Retrying ingestion for ${shortHash(hash)}...`);
    try { const res = await fetch(api.retry(hash), { method: "POST" }); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText); setJobId(data.job_id || null); setProcessing(true); setUploadStatus(`Re-queued (job ${data.job_id})`); void refreshDocs(); }
    catch (e) { setUploadStatus(`Retry failed: ${e.message || String(e)}`); }
    finally { setRetryingHash(null); }
  }, [api, refreshDocs]);

  const handlePreview = useCallback(async (doc) => {
    if (!doc || !doc.hash) return; setSelectedDoc(doc); setPreview(""); setPreviewError(""); setPreviewInfo(null); setPreviewLoading(true);
    try { const res = await fetch(api.previewText(doc.hash, previewMaxChars, parser)); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET preview ${res.status}`); setPreview(typeof data.preview === "string" ? data.preview : ""); setPreviewInfo({ document_name: data.document_name, file_size: data.file_size, extracted_chars: data.extracted_chars, total_tokens: data.total_tokens, chunk_count: data.chunk_count, preview_chars: data.preview_chars, truncated: !!data.truncated, parser }); }
    catch (e) { setPreviewError(e.message || String(e)); }
    finally { setPreviewLoading(false); }
  }, [api, previewMaxChars, parser]);

  useEffect(() => { if (!jobId) return; const id = setInterval(async () => { try { const res = await fetch(`/api/status/${jobId}`); const data = await readJsonSafe(res); if (res.ok && data.status) { const st = String(data.status).toLowerCase(); if (st === "done") { setProcessing(false); setJobId(null); setUploadStatus("Ingestion complete."); void refreshDocs(); clearInterval(id); } else if (st.startsWith("error")) { setProcessing(false); setJobId(null); setUploadStatus(`Ingestion error: ${data.status}`); clearInterval(id); } } } catch {} }, 1500); return () => clearInterval(id); }, [jobId, refreshDocs]);

  return (
    <div style={styles.page}>
      <div style={styles.leftColumn}>
        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Upload Documents</h3>
            <span style={styles.badge}>{processing ? "Processing" : uploading ? "Uploading" : "Ready"}</span>
          </div>
          <div style={styles.row}>
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} style={styles.input} />
            <button onClick={handleUpload} disabled={uploading || processing || !file} style={{ ...styles.button, opacity: uploading || processing || !file ? 0.6 : 1 }}>
              {uploading ? "Uploading…" : processing ? "Processing…" : "Ingest"}
            </button>
          </div>
          <div style={styles.feedback}>{uploadStatus || "PDFs supported. Large files allowed."}</div>
        </section>

        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Document Details & Preview</h3>
            {selectedDoc && <span style={styles.badge}>{parser}</span>}
          </div>
          
          {selectedDoc ? (
            <>
              {/* Document Metadata Section */}
              <div style={{ background: "rgba(23, 25, 35, 0.6)", border: "1px solid rgba(148, 163, 184, 0.12)", borderRadius: 12, padding: 16, marginBottom: 16 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <span style={{ fontSize: 18 }}>📄</span>
                  <strong style={{ fontSize: 15, color: "#e2e8f0" }}>{previewInfo?.document_name || selectedDoc.name}</strong>
                </div>
                
                {/* Statistics Grid */}
                {previewInfo && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: 12, marginBottom: 12 }}>
                    <div style={{ textAlign: "center", padding: "8px 12px", background: "rgba(84, 105, 255, 0.08)", borderRadius: 8, border: "1px solid rgba(84, 105, 255, 0.2)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{prettyBytes(previewInfo.file_size)}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>File Size</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "8px 12px", background: "rgba(84, 105, 255, 0.08)", borderRadius: 8, border: "1px solid rgba(84, 105, 255, 0.2)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{(previewInfo.extracted_chars || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Characters</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "8px 12px", background: "rgba(84, 105, 255, 0.08)", borderRadius: 8, border: "1px solid rgba(84, 105, 255, 0.2)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{(previewInfo.total_tokens || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Tokens</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "8px 12px", background: "rgba(84, 105, 255, 0.08)", borderRadius: 8, border: "1px solid rgba(84, 105, 255, 0.2)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{previewInfo.chunk_count || 0}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Embeddings</div>
                    </div>
                  </div>
                )}
                
                {/* Parser Controls */}
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <label style={{ ...styles.muted, fontSize: 12 }}>Parser</label>
                  <select value={parser} onChange={(e) => setParser(e.target.value)} style={{ ...styles.input, padding: "6px 10px", fontSize: 13 }}>
                    <option value="mineru">mineru</option>
                    <option value="pymupdf">pymupdf</option>
                  </select>
                  <label style={{ ...styles.muted, fontSize: 12, marginLeft: 8 }}>Max chars</label>
                  <input type="number" min={200} max={20000} step={100} value={previewMaxChars} onChange={(e) => setPreviewMaxChars(Number(e.target.value) || 2000)} style={{ ...styles.input, width: 100, padding: "6px 10px", fontSize: 13 }} />
                  <button style={{ ...styles.subtleButton, marginLeft: "auto" }} onClick={() => handlePreview(selectedDoc)} disabled={previewLoading}>
                    {previewLoading ? "Refreshing…" : "Refresh"}
                  </button>
                </div>
              </div>
              
              {/* Text Preview Section */}
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(148, 163, 184, 0.9)", marginBottom: 8 }}>Extracted Text</div>
                <div style={{ border: "1px solid rgba(148,163,184,0.12)", borderRadius: 12, background: "rgba(9, 11, 18, 0.72)", padding: 12, maxHeight: "40vh", overflow: "auto", whiteSpace: "pre-wrap", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", fontSize: 13, lineHeight: 1.5 }}>
                  {preview || (previewLoading ? "Loading…" : "No text extracted.")}
                </div>
                {previewInfo && (
                  <div style={{ ...styles.muted, marginTop: 8, fontSize: 12 }}>
                    {`Showing ${(previewInfo.preview_chars || 0).toLocaleString()} of ${(previewInfo.extracted_chars || 0).toLocaleString()} characters`}
                    {previewInfo.truncated ? " (truncated)" : ""}
                  </div>
                )}
                {previewError && (<div style={styles.error}>Error: {previewError}</div>)}
              </div>
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "40px 20px", color: "rgba(148, 163, 184, 0.7)" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>📄</div>
              <div>Select a document from the library below to view its details and preview the extracted text.</div>
            </div>
          )}
        </section>
      </div>

      <section style={styles.card}>
        <div style={styles.sectionHeader}>
          <div>
            <h3 style={styles.sectionTitle}>Document Library</h3>
            <span style={styles.muted}>{`${systemStatus.docs_count || 0} ready / ${systemStatus.total_docs || displayDocs.length}`}</span>
          </div>
          <button onClick={refreshDocs} disabled={docsLoading} style={{ ...styles.subtleButton, opacity: docsLoading ? 0.6 : 1 }}>{docsLoading ? "Refreshing…" : "Refresh"}</button>
        </div>
        <div style={{ ...styles.docs, ...(displayDocs.length ? {} : styles.muted) }}>
          {displayDocs.length === 0 ? (
            <div>No documents yet. Upload files to build your knowledge base.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {displayDocs.map((d) => (
                <li key={d.hash || d.stored_name || d.name} style={styles.listItem}>
                  <div title={d.path}>
                    <strong>{d.name}</strong>{" "}
                    <span style={styles.muted}>{`${prettyBytes(d.size)} · ${(d.status || "unknown").toLowerCase()}${d.hash ? ` · ${shortHash(d.hash)}` : ""}`}</span>
                    {d.hash && (<button style={{ ...styles.subtleButton, marginLeft: 8 }} onClick={() => handlePreview(d)} disabled={previewLoading && selectedDoc && selectedDoc.hash === d.hash} title="Preview extracted text">{previewLoading && selectedDoc && selectedDoc.hash === d.hash ? "Loading…" : "Preview"}</button>)}
                    {String(d.status || "").toLowerCase() === "error" && d.hash && (<button style={{ ...styles.subtleButton, marginLeft: 8 }} onClick={() => handleRetry(d.hash)} disabled={retryingHash === d.hash}>{retryingHash === d.hash ? "Retrying…" : "Retry"}</button>)}
                  </div>
                  {d.last_ingested_at && (<div style={styles.muted}>Last ingested {formatDate(d.last_ingested_at)}</div>)}
                  {d.error && (<div style={styles.error}>Error: {d.error}</div>)}
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
