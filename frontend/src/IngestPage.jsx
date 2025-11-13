import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

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
  page: { display: "flex", flexWrap: "wrap", gap: 18, alignItems: "flex-start" },
  leftColumn: { display: "grid", gap: 18, flex: "2 1 520px", minWidth: 0 },
  libraryCard: { flex: "1 1 360px", minWidth: 0 },
  card: { border: "none", borderRadius: 22, padding: "20px 22px", background: "rgba(13, 16, 24, 0.92)", boxShadow: "0 22px 44px rgba(2, 6, 23, 0.45)" },
  sectionHeader: { display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 12 },
  sectionTitle: { margin: 0, fontSize: 18, fontWeight: 600, letterSpacing: 0.2 },
  badge: { padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.4)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", whiteSpace: "nowrap" },
  row: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  button: { font: "inherit", fontSize: 14, padding: "8px 18px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(84, 105, 255, 0.35), rgba(84, 105, 255, 0.2))", color: "#c7d7ff", cursor: "pointer", boxShadow: "0 12px 24px rgba(2, 6, 23, 0.45)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  subtleButton: { font: "inherit", fontSize: 13, padding: "6px 14px", borderRadius: 999, border: "none", background: "rgba(29, 34, 49, 0.82)", color: "rgba(226, 232, 240, 0.88)", cursor: "pointer", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  input: { font: "inherit", padding: "10px 16px", borderRadius: 999, border: "none", background: "rgba(9, 11, 18, 0.9)", color: "inherit", minWidth: 0, boxShadow: "0 0 0 1px rgba(148, 163, 184, 0.12), inset 0 2px 10px rgba(2, 6, 23, 0.65)", outline: "none" },
  feedback: { marginTop: 10, fontSize: 13, color: "rgba(148, 163, 184, 0.85)" },
  docs: { maxHeight: "55vh", overflow: "auto", border: "none", borderRadius: 20, padding: 12, background: "rgba(9, 11, 18, 0.78)", boxShadow: "inset 0 0 0 1px rgba(15, 23, 42, 0.25)" },
  listItem: { padding: "8px 12px 8px", borderRadius: 16, background: "rgba(23, 25, 35, 0.78)", border: "none", marginBottom: 10, display: "flex", flexDirection: "column", gap: 2, boxShadow: "0 10px 20px rgba(2, 6, 23, 0.35)" },
  docTitleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 },
  docTitleActions: { display: "flex", alignItems: "center", gap: 8, flexShrink: 0 },
  docName: { fontSize: 15, fontWeight: 600, color: "rgba(226, 232, 240, 0.95)", margin: 0, wordBreak: "break-word", lineHeight: 1.2 },
  docStatusPill: { fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6, padding: "2px 8px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.22)", color: "rgba(148, 163, 184, 0.85)" },
  docMetaRow: { display: "flex", flexWrap: "wrap", gap: 2, fontSize: 12, color: "rgba(148, 163, 184, 0.9)", marginTop: 0, lineHeight: 1.25 },
  docMetaItem: { whiteSpace: "nowrap" },
  docActions: { display: "flex", justifyContent: "flex-end", gap: 1, marginTop: 2 },
  docPreviewButton: { font: "inherit", fontSize: 13, padding: "4px 14px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(84, 105, 255, 0.28), rgba(84, 105, 255, 0.12))", color: "rgba(226, 232, 240, 0.98)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)" },
  dangerIconButton: { font: "inherit", fontSize: 16, lineHeight: 1, width: 24, height: 24, borderRadius: 12, border: "none", background: "rgba(239, 68, 68, 0.18)", color: "rgba(248, 250, 252, 0.92)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", padding: 0, boxShadow: "0 8px 16px rgba(239, 68, 68, 0.25)" },
  docIngestRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, marginTop: 1, fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  docIngestInfo: { flex: 1, lineHeight: 1 },
  docPerfToggle: { font: "inherit", fontSize: 12, display: "inline-flex", alignItems: "center", gap: 2, padding: 0, border: "none", background: "transparent", color: "rgba(148, 163, 184, 0.75)", cursor: "pointer" },
  docPerfToggleActive: { color: "rgba(226, 232, 240, 0.95)" },
  docPerfDetails: { marginTop: 1, marginLeft: 12, fontSize: 11, color: "rgba(148, 163, 184, 0.85)", lineHeight: 1.35 },
  muted: { opacity: 0.75, fontSize: 13, color: "rgba(148, 163, 184, 0.8)" },
  error: { fontSize: 13, color: "#ff8f8f", marginTop: 6 },
};

const FALLBACK_PARSER = "mineru";
const FALLBACK_PARSER_OPTIONS = [FALLBACK_PARSER, "pymupdf"];
const IN_PROGRESS_STATUSES = new Set(["processing", "ingesting", "queued", "pending", "running", "parsing", "uploading"]);
const COMPLETED_STATUSES = new Set(["processed", "done", "completed", "ready"]);

export default function IngestPage({ systemStatus = {} }) {
  const navigate = useNavigate();
  const api = useMemo(() => ({
    ingest: "/api/ingest",
    docs: "/api/documents",
    status: (jobId) => `/api/status/${jobId}`,
    retry: (hash) => `/api/ingest/${hash}/retry`,
    previewText: (hash, maxChars = 2000, parser = FALLBACK_PARSER) => `/api/debug/parsed_text/${hash}?parser=${parser}&max_chars=${maxChars}`,
  }), []);

  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState([]);
  const [activeJobs, setActiveJobs] = useState(new Set());
  const [retryingHash, setRetryingHash] = useState(null);
  const [deletingHash, setDeletingHash] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [preview, setPreview] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [previewInfo, setPreviewInfo] = useState(null);
  const [previewMaxChars, setPreviewMaxChars] = useState(2000);
  const [parser, setParser] = useState(FALLBACK_PARSER);
  const [parserOptions, setParserOptions] = useState(FALLBACK_PARSER_OPTIONS);
  const [expandedPerf, setExpandedPerf] = useState(new Set());
  const selectedDocRef = useRef(null);
  const lastPreviewParamsRef = useRef({ parser, previewMaxChars });

  useEffect(() => {
    selectedDocRef.current = selectedDoc;
  }, [selectedDoc]);

  const readyDocCount = Number(systemStatus.docs_count || 0);
  const canOpenChat = systemStatus.ready && !systemStatus.has_running_jobs && readyDocCount > 0;
  const chatDisabledReason = systemStatus.has_running_jobs
    ? "Finish processing the current jobs before chatting."
    : !systemStatus.ready
      ? "System is still preparing the chat experience."
      : readyDocCount === 0
        ? "Ingest at least one document to unlock chat."
        : "";

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try { const res = await fetch(api.docs); const data = await readJsonSafe(res); if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`); setDocs(Array.isArray(data) ? data : []); }
    catch { setDocs([]); }
    finally { setDocsLoading(false); }
  }, [api.docs]);

  useEffect(() => { void refreshDocs(); }, [refreshDocs]);

  useEffect(() => {
    let cancelled = false;
    const fetchParsers = async () => {
      try {
        const res = await fetch("/api/parsers");
        const data = await readJsonSafe(res);
        if (!res.ok || cancelled) return;
        const available = Array.isArray(data?.options)
          ? data.options
          : Array.isArray(data?.available)
            ? data.available
            : [];
        const cleaned = Array.from(
          new Set(
            available
              .map((v) => (typeof v === "string" ? v.trim() : ""))
              .filter(Boolean)
          )
        );
        if (!cancelled && cleaned.length > 0) {
          setParserOptions(cleaned);
        }
        const nextDefault = typeof data?.default === "string"
          ? data.default.trim()
          : typeof data?.default_parser === "string"
            ? data.default_parser.trim()
            : typeof data?.ocr_parser === "string"
              ? data.ocr_parser.trim()
              : "";
        if (!cancelled && nextDefault) {
          setParser(prev => (prev === nextDefault ? prev : nextDefault));
        }
      } catch (err) {
        console.warn("Failed to load parser metadata", err);
      }
    };
    void fetchParsers();
    return () => { cancelled = true; };
  }, []);

  const systemDocs = useMemo(() => (Array.isArray(systemStatus.documents) ? systemStatus.documents : []), [systemStatus.documents]);
  const displayDocs = docs.length ? docs : systemDocs;

  const handleUploadAll = async () => {
    if (files.length === 0) { setUploadStatus("Select files first."); return; }
    
    setUploading(true);
    setUploadStatus(`Uploading ${files.length} file(s)...`);
    
    // Initialize progress tracking
    const progress = files.map((f, idx) => ({
      id: idx,
      name: f.name,
      status: "pending",
      jobId: null,
      error: null,
    }));
    setUploadProgress(progress);
    
    // Upload all files in parallel
    const uploadPromises = files.map(async (file, idx) => {
      try {
        // Client-side guard: only allow PDFs to be sent
        const isPdf = (file && (file.type === "application/pdf" || /\.pdf$/i.test(file.name)));
        if (!isPdf) {
          setUploadProgress(prev => prev.map((p, i) => 
            i === idx ? { ...p, status: "error", error: "Only PDF files are supported" } : p
          ));
          return;
        }
        // Update status to uploading
        setUploadProgress(prev => prev.map((p, i) => 
          i === idx ? { ...p, status: "uploading" } : p
        ));
        
        const form = new FormData();
        form.append("file", file);
        const res = await fetch(api.ingest, { method: "POST", body: form });
        const data = await readJsonSafe(res);
        
        if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
        
        if (data.status === "skipped") {
          setUploadProgress(prev => prev.map((p, i) => 
            i === idx ? { ...p, status: "skipped", error: "Already ingested" } : p
          ));
        } else if (data.job_id) {
          setUploadProgress(prev => prev.map((p, i) => 
            i === idx ? { ...p, status: "queued", jobId: data.job_id } : p
          ));
          setActiveJobs(prev => new Set([...prev, data.job_id]));
        }
      } catch (e) {
        setUploadProgress(prev => prev.map((p, i) => 
          i === idx ? { ...p, status: "error", error: e.message || String(e) } : p
        ));
      }
    });
    
    await Promise.all(uploadPromises);
    setUploading(false);
    setUploadStatus(`Uploaded ${files.length} file(s). Processing...`);
    void refreshDocs();
  };

  const handleRetry = useCallback(async (hash) => {
    if (!hash) return; 
    setRetryingHash(hash); 
    setUploadStatus(`Retrying ingestion for ${shortHash(hash)}...`);
    try { 
      const res = await fetch(api.retry(hash), { method: "POST" }); 
      const data = await readJsonSafe(res); 
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText); 
      if (data.job_id) {
        setActiveJobs(prev => new Set([...prev, data.job_id]));
        setUploadStatus(`Re-queued (job ${data.job_id})`);
      }
      void refreshDocs(); 
    }
    catch (e) { setUploadStatus(`Retry failed: ${e.message || String(e)}`); }
    finally { setRetryingHash(null); }
  }, [api, refreshDocs]);

  const handleDeleteDoc = useCallback(async (doc) => {
    if (!doc?.hash) return;
    const hash = doc.hash;
    const displayName = doc.name || doc.stored_name || shortHash(hash);
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(`Remove "${displayName}" and all derived data? This cannot be undone.`);
      if (!confirmed) return;
    }

    setDeletingHash(hash);
    setUploadStatus(`Removing ${displayName}...`);
    try {
      const res = await fetch(`${api.docs}/${hash}`, { method: "DELETE" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);
      setUploadStatus(`Removed ${displayName}.`);
      if (selectedDoc && selectedDoc.hash === hash) {
        setSelectedDoc(null);
        setPreview("");
        setPreviewInfo(null);
        setPreviewError("");
      }
      setExpandedPerf(prev => {
        const next = new Set(prev);
        next.delete(hash);
        return next;
      });
      await refreshDocs();
    } catch (err) {
      setUploadStatus(`Remove failed: ${err.message || String(err)}`);
    } finally {
      setDeletingHash(null);
    }
  }, [api.docs, refreshDocs, selectedDoc]);

  const handlePreview = useCallback(async (doc, { skipSelect = false } = {}) => {
    if (!doc || !doc.hash) return;

    if (!skipSelect) {
      setSelectedDoc(prev => (prev && prev.hash === doc.hash ? prev : doc));
    }

    setPreview("");
    setPreviewError("");
    setPreviewInfo(null);
    setPreviewLoading(true);

    try {
      const res = await fetch(api.previewText(doc.hash, previewMaxChars, parser));
      const data = await readJsonSafe(res);
      if (!res.ok) {
        throw new Error((data && (data.detail || data.error || data.raw)) || `GET preview ${res.status}`);
      }
      setPreview(typeof data.preview === "string" ? data.preview : "");
      setPreviewInfo({
        document_name: data.document_name,
        file_size: data.file_size,
        extracted_chars: data.extracted_chars,
        total_tokens: data.total_tokens,
        chunk_count: data.chunk_count,
        preview_chars: data.preview_chars,
        truncated: !!data.truncated,
        parser,
      });
    } catch (e) {
      setPreviewError(e.message || String(e));
    } finally {
      setPreviewLoading(false);
    }
  }, [api, parser, previewMaxChars]);

  useEffect(() => {
    const prev = lastPreviewParamsRef.current;
    const hasParserChanged = prev.parser !== parser;
    const hasMaxChanged = prev.previewMaxChars !== previewMaxChars;
    const doc = selectedDocRef.current;

    if (!doc?.hash) {
      lastPreviewParamsRef.current = { parser, previewMaxChars };
      return;
    }
    if (!hasParserChanged && !hasMaxChanged) {
      return;
    }

    lastPreviewParamsRef.current = { parser, previewMaxChars };
    void handlePreview(doc, { skipSelect: true });
  }, [parser, previewMaxChars, handlePreview]);

  // Poll active jobs
  useEffect(() => {
    if (activeJobs.size === 0) return;
    
    const id = setInterval(async () => {
      const jobsToCheck = Array.from(activeJobs);
      const results = await Promise.all(
        jobsToCheck.map(async (jobId) => {
          try {
            const res = await fetch(api.status(jobId));
            const data = await readJsonSafe(res);
            if (res.ok && data.status) {
              const st = String(data.status).toLowerCase();
              return { jobId, status: st };
            }
          } catch {}
          return { jobId, status: null };
        })
      );
      
      // Remove completed jobs
      const completedJobs = results.filter(r => r.status === "done" || r.status?.startsWith("error"));
      if (completedJobs.length > 0) {
        setActiveJobs(prev => {
          const newSet = new Set(prev);
          completedJobs.forEach(j => newSet.delete(j.jobId));
          return newSet;
        });
        void refreshDocs();
      }
    }, 2000);
    
    return () => clearInterval(id);
  }, [activeJobs, api, refreshDocs]);

  const handleOpenChat = useCallback(() => {
    if (!canOpenChat) return;
    navigate("/chat");
  }, [canOpenChat, navigate]);

  return (
    <div style={styles.page}>
      <div style={styles.leftColumn}>
        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Upload Documents</h3>
            <span style={styles.badge}>{uploading ? "Uploading" : activeJobs.size > 0 ? "Processing" : "Ready"}</span>
          </div>
          <div style={styles.row}>
            <input 
              type="file" 
              multiple 
              accept=".pdf,application/pdf"
              onChange={(e) => setFiles(Array.from(e.target.files || []))} 
              style={styles.input} 
            />
            <button 
              onClick={handleUploadAll} 
              disabled={uploading || files.length === 0} 
              style={{ ...styles.button, opacity: uploading || files.length === 0 ? 0.6 : 1 }}
            >
              {uploading ? "Uploading…" : `Ingest ${files.length > 0 ? `(${files.length})` : ""}`}
            </button>
          </div>
          <div style={styles.feedback}>
            {uploadStatus || (files.length > 0 ? `${files.length} file(s) selected` : "PDFs supported. Select multiple files.")}
          </div>
          
          {/* Upload Progress */}
          {uploadProgress.length > 0 && (
            <div style={{ marginTop: 12, maxHeight: "200px", overflow: "auto" }}>
              {uploadProgress.map((p) => (
                <div 
                  key={p.id} 
                  style={{ 
                    fontSize: 12, 
                    padding: "6px 10px", 
                    marginBottom: 4, 
                    background: "rgba(23, 25, 35, 0.55)", 
                    borderRadius: 12,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    boxShadow: "0 6px 14px rgba(2, 6, 23, 0.35)"
                  }}
                >
                  <span>
                    {p.status === "pending" && "⏸"}
                    {p.status === "uploading" && "⏳"}
                    {p.status === "queued" && "✓"}
                    {p.status === "skipped" && "⊘"}
                    {p.status === "error" && "✗"}
                  </span>
                  <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {p.name}
                  </span>
                  <span style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.7)" }}>
                    {p.status === "queued" && p.jobId ? `job ${shortHash(p.jobId)}` : p.status}
                  </span>
                  {p.error && (
                    <span style={{ fontSize: 11, color: "#ff8f8f" }} title={p.error}>
                      {p.error.length > 20 ? p.error.substring(0, 20) + "..." : p.error}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Document Details & Preview</h3>
            {selectedDoc && <span style={styles.badge}>{parser}</span>}
          </div>
          
          {selectedDoc ? (
            <>
              {/* Document Metadata Section */}
              <div style={{ background: "rgba(23, 25, 35, 0.68)", border: "none", borderRadius: 20, padding: 16, marginBottom: 12, boxShadow: "0 18px 32px rgba(2, 6, 23, 0.4)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 18 }}>📄</span>
                  <strong style={{ fontSize: 15, color: "#e2e8f0", flex: 1, minWidth: 0, wordBreak: "break-word" }}>
                    {previewInfo?.document_name || selectedDoc.name}
                  </strong>
                </div>
                
                {/* Statistics Grid */}
                {previewInfo && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: 10, marginBottom: 10 }}>
                    <div style={{ textAlign: "center", padding: "10px 12px", background: "rgba(84, 105, 255, 0.15)", borderRadius: 16, border: "none", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{prettyBytes(previewInfo.file_size)}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>File Size</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "10px 12px", background: "rgba(84, 105, 255, 0.15)", borderRadius: 16, border: "none", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{(previewInfo.extracted_chars || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Characters</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "10px 12px", background: "rgba(84, 105, 255, 0.15)", borderRadius: 16, border: "none", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{(previewInfo.total_tokens || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Tokens</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "10px 12px", background: "rgba(84, 105, 255, 0.15)", borderRadius: 16, border: "none", boxShadow: "0 10px 18px rgba(2, 6, 23, 0.35)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{previewInfo.chunk_count || 0}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Embeddings</div>
                    </div>
                  </div>
                )}
                
                {/* Parser Controls */}
                <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                  <label style={{ ...styles.muted, fontSize: 12 }}>Parser</label>
                  <select value={parser} onChange={(e) => setParser(e.target.value)} style={{ ...styles.input, padding: "6px 10px", fontSize: 13 }}>
                    {parserOptions.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                  <label style={{ ...styles.muted, fontSize: 12, marginLeft: 8 }}>Max chars</label>
                  <input type="number" min={200} max={20000} step={100} value={previewMaxChars} onChange={(e) => setPreviewMaxChars(Number(e.target.value) || 2000)} style={{ ...styles.input, width: 100, padding: "6px 10px", fontSize: 13 }} />
                </div>
              </div>
              
              {/* Text Preview Section */}
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: "rgba(148, 163, 184, 0.9)", marginBottom: 6 }}>Extracted Text</div>
                <div style={{ border: "none", borderRadius: 18, background: "rgba(9, 11, 18, 0.78)", padding: 14, maxHeight: "40vh", overflow: "auto", whiteSpace: "pre-wrap", wordBreak: "break-word", overflowWrap: "anywhere", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace", fontSize: 13, lineHeight: 1.5, boxShadow: "inset 0 0 0 1px rgba(15, 23, 42, 0.35)" }}>
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

      <section style={{ ...styles.card, ...styles.libraryCard }}>
        <div style={styles.sectionHeader}>
          <div>
            <h3 style={styles.sectionTitle}>Document Library</h3>
            <span style={styles.muted}>{`${systemStatus.docs_count || 0} ready / ${systemStatus.total_docs || displayDocs.length}`}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-end" }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
              <button
                onClick={refreshDocs}
                disabled={docsLoading}
                style={{ ...styles.subtleButton, opacity: docsLoading ? 0.6 : 1 }}
              >
                {docsLoading ? "Refreshing…" : "Refresh"}
              </button>
              <button
                onClick={handleOpenChat}
                disabled={!canOpenChat}
                style={{ ...styles.button, padding: "6px 14px", opacity: canOpenChat ? 1 : 0.5 }}
                title={canOpenChat ? "Jump to chat" : chatDisabledReason}
              >
                Open Chat
              </button>
            </div>
            {!canOpenChat && (
              <span style={{ ...styles.muted, fontSize: 11, textAlign: "right" }}>{chatDisabledReason}</span>
            )}
          </div>
        </div>
        <div style={{ ...styles.docs, ...(displayDocs.length ? {} : styles.muted) }}>
          {displayDocs.length === 0 ? (
            <div>No documents yet. Upload files to build your knowledge base.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {displayDocs.map((d) => {
                const perf = d.performance;
                const hasPerf = perf && perf.total_time_sec != null;
                const isExpanded = expandedPerf.has(d.hash);
                const statusLabel = String(d.status || "pending").trim().toLowerCase();
                const isErrored = statusLabel === "error";
                const isInProgress = IN_PROGRESS_STATUSES.has(statusLabel);
                const isCompleted = COMPLETED_STATUSES.has(statusLabel);
                const canPreviewInHeader = Boolean(d.hash && !isErrored && !isInProgress);
                const showStatusPill = !canPreviewInHeader;
                const showRetry = isErrored && Boolean(d.hash);
                const showPerf = hasPerf && isCompleted;
                const isDeleting = deletingHash === d.hash;
                
                return (
                  <li key={d.hash || d.stored_name || d.name} style={styles.listItem}>
                    <div style={styles.docTitleRow} title={d.path}>
                      <div style={{ flex: 1 }}>
                        <div style={styles.docName}>{d.name || d.stored_name || "Untitled document"}</div>
                      </div>
                      <div style={styles.docTitleActions}>
                        {canPreviewInHeader ? (
                          <button
                            style={{ ...styles.docPreviewButton, opacity: previewLoading && selectedDoc && selectedDoc.hash === d.hash ? 0.6 : 1 }}
                            onClick={() => handlePreview(d)}
                            disabled={previewLoading && selectedDoc && selectedDoc.hash === d.hash}
                            title="Preview extracted text"
                          >
                            {previewLoading && selectedDoc && selectedDoc.hash === d.hash ? "Loading…" : "Preview"}
                          </button>
                        ) : showStatusPill ? (
                          <span style={styles.docStatusPill}>{statusLabel.toUpperCase()}</span>
                        ) : null}
                        <button
                          type="button"
                          style={{ ...styles.dangerIconButton, opacity: isDeleting ? 0.5 : 1 }}
                          onClick={() => handleDeleteDoc(d)}
                          disabled={isDeleting}
                          title="Remove document"
                          aria-label={`Remove ${d.name || d.stored_name || "document"}`}
                        >
                          ×
                        </button>
                      </div>
                    </div>

                    <div style={styles.docMetaRow}>
                      <span style={styles.docMetaItem}>
                        <span style={{ opacity: 0.65 }}>Size:</span> {prettyBytes(d.size)}
                      </span>
                      {d.hash && (
                        <span style={styles.docMetaItem}>
                          <span style={{ opacity: 0.65 }}>Hash:</span> {shortHash(d.hash)}
                        </span>
                      )}
                    </div>

                    {(d.last_ingested_at || showPerf) && (
                      <>
                        <div style={styles.docIngestRow}>
                          <span style={styles.docIngestInfo}>
                            {d.last_ingested_at ? `Ingested ${formatDate(d.last_ingested_at)}` : "Ingestion details pending"}
                            {showPerf && (
                              <>
                                {" • "}
                                <span style={{ fontWeight: 600 }}>{perf.total_time_sec.toFixed(1)}s</span>
                              </>
                            )}
                          </span>
                          {showPerf && (
                            <button
                              type="button"
                              onClick={() => {
                                const newSet = new Set(expandedPerf);
                                if (isExpanded) newSet.delete(d.hash);
                                else newSet.add(d.hash);
                                setExpandedPerf(newSet);
                              }}
                              style={{
                                ...styles.docPerfToggle,
                                ...(isExpanded ? styles.docPerfToggleActive : {}),
                              }}
                              title={isExpanded ? "Hide breakdown" : "Show breakdown"}
                              aria-label={`${isExpanded ? "Hide" : "Show"} ingestion breakdown`}
                            >
                              {isExpanded ? "▲" : "▼"}
                            </button>
                          )}
                        </div>

                        {showPerf && isExpanded && (
                          <div style={styles.docPerfDetails}>
                            {perf.pymupdf_time_sec != null && (
                              <div>• PyMuPDF: {perf.pymupdf_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.mineru_time_sec != null && (
                              <div>• MinerU: {perf.mineru_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.mineru_time_sec == null && (
                              <div style={{ opacity: 0.6 }}>• MinerU: skipped</div>
                            )}
                            {perf.chunking_time_sec != null && (
                              <div>• Chunking: {perf.chunking_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.embedding_time_sec != null && (
                              <div>• Embeddings: {perf.embedding_time_sec.toFixed(2)}s</div>
                            )}
                          </div>
                        )}
                      </>
                    )}
                    {d.error && (<div style={styles.error}>Error: {d.error}</div>)}

                    {showRetry && (
                      <div style={styles.docActions}>
                        <button
                          style={styles.subtleButton}
                          onClick={() => handleRetry(d.hash)}
                          disabled={retryingHash === d.hash}
                        >
                          {retryingHash === d.hash ? "Retrying…" : "Retry"}
                        </button>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
