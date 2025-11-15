import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import "./IngestPage.css";

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

function clampPercent(v, fallback = 0) {
  const n = Number(v);
  if (Number.isFinite(n)) {
    return Math.min(100, Math.max(0, n));
  }
  return fallback;
}

function formatProgressDetails(progress) {
  if (!progress) return "";
  const phaseRaw = typeof progress.phase === "string" ? progress.phase : progress.stage;
  const stageRaw = typeof progress.stage === "string" ? progress.stage : "";
  const phase = (phaseRaw || "").replace(/[:_]/g, " ").trim();
  const stage = (stageRaw || "").replace(/[:_]/g, " ").trim();
  const percent = typeof progress.percent === "number" && !Number.isNaN(progress.percent)
    ? `${Math.round(progress.percent)}%`
    : "";
  const parts = [];
  if (phase) parts.push(phase.toUpperCase());
  if (percent) parts.push(percent);
  if (stage && stage.toLowerCase() !== phase.toLowerCase()) {
    parts.push(`· ${stage}`);
  }
  const current = progress.current ?? progress.processed;
  const total = progress.total ?? progress.total_items ?? progress.total_chunks;
  if (typeof current === "number" && typeof total === "number" && total > 0) {
    parts.push(`(${current}/${total})`);
  } else if (typeof progress.chunk_count === "number") {
    parts.push(`(${progress.chunk_count} chunk${progress.chunk_count === 1 ? "" : "s"})`);
  }
  return parts.join(" ").trim();
}

const styles = {
  page: {
    display: "grid",
    gridTemplateColumns: "var(--ingest-grid-columns, minmax(0, 1.65fr) minmax(320px, 0.95fr))",
    gap: 20,
    alignItems: "stretch",
    width: "100%",
    minHeight: "var(--ingest-left-min-height, calc(100vh - 64px))",
  },
  leftColumn: {
    display: "grid",
    gridTemplateRows: "var(--ingest-upload-height, 300px) minmax(0, 1fr)",
    gap: 18,
    minWidth: 0,
    minHeight: "var(--ingest-left-min-height, calc(100vh - 64px))",
    height: "var(--ingest-left-min-height, calc(100vh - 64px))",
  },
  uploadCard: {
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "hidden",
  },
  uploadProgressList: {
    marginTop: 12,
    flex: 1,
    minHeight: 0,
    maxHeight: "var(--ingest-upload-scroll-height, calc(5 * 58px))",
    overflow: "auto",
    paddingRight: 4,
  },
  previewCard: {
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
    overflow: "hidden",
  },
  previewBody: { flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" },
  previewContent: { flex: 1, display: "flex", flexDirection: "column", minHeight: 0 },
  previewHeaderRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 6 },
  previewTitle: { fontSize: 13, fontWeight: 600, color: "rgba(148, 163, 184, 0.9)" },
  previewLimiterControls: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  previewLimitToggle: { display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  previewLimitCheckbox: { width: 16, height: 16, accentColor: "#818cf8" },
  libraryCard: {
    minWidth: 0,
    minHeight: 0,
    width: "100%",
    display: "flex",
    flexDirection: "column",
    gap: 16,
    position: "var(--ingest-library-position, sticky)",
    top: "var(--ingest-library-top, 16px)",
    height: "var(--ingest-library-height, calc(100vh - 64px))",
    maxHeight: "var(--ingest-library-height, calc(100vh - 64px))",
    minHeight: "var(--ingest-library-height, calc(100vh - 64px))",
    overflow: "hidden",
  },
  card: { border: "none", borderRadius: 26, padding: "24px 26px", background: "linear-gradient(150deg, rgba(78, 89, 162, 0.98), rgba(26, 30, 64, 0.95))", boxShadow: "0 38px 76px rgba(5, 8, 25, 0.78)" },
  sectionHeader: { display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 12 },
  sectionTitle: { margin: 0, fontSize: 18, fontWeight: 600, letterSpacing: 0.2 },
  badge: { padding: "4px 10px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.4)", fontSize: 12, color: "rgba(148, 163, 184, 0.85)", whiteSpace: "nowrap" },
  row: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" },
  button: { font: "inherit", fontSize: 14, padding: "11px 24px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(167, 139, 250, 0.94), rgba(59, 130, 246, 0.82))", color: "#f8f9ff", cursor: "pointer", boxShadow: "0 24px 44px rgba(5, 9, 25, 0.65)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  subtleButton: { font: "inherit", fontSize: 13, padding: "8px 18px", borderRadius: 999, border: "none", background: "rgba(77, 88, 142, 0.96)", color: "rgba(248, 250, 255, 0.96)", cursor: "pointer", boxShadow: "0 18px 32px rgba(6, 9, 23, 0.62)", transition: "transform 0.15s ease, box-shadow 0.15s ease" },
  input: { font: "inherit", padding: "12px 20px", borderRadius: 999, border: "none", background: "rgba(23, 28, 60, 0.98)", color: "inherit", minWidth: 0, boxShadow: "0 0 0 2px rgba(59, 130, 246, 0.26), inset 0 2px 14px rgba(3, 6, 18, 0.7)", outline: "none" },
  feedback: { marginTop: 10, fontSize: 13, color: "rgba(148, 163, 184, 0.85)" },
  docs: {
    flex: 1,
    width: "100%",
    minHeight: 0,
    overflowY: "auto",
    overflowX: "hidden",
    maxHeight: "var(--ingest-library-scroll-max-height, calc(100vh - 180px))",
    border: "none",
    borderRadius: 26,
    padding: 16,
    background: "rgba(21, 26, 58, 0.97)",
    boxShadow: "0 24px 46px rgba(0, 0, 0, 0.6), inset 0 0 0 2px rgba(99, 102, 241, 0.11)",
  },
  listItem: { padding: "12px 18px 12px", borderRadius: 20, background: "rgba(50, 63, 128, 0.96)", border: "none", marginBottom: 14, display: "flex", flexDirection: "column", gap: 4, boxShadow: "0 18px 32px rgba(4, 7, 20, 0.65)" },
  docTitleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 },
  docTitleActions: { display: "flex", alignItems: "center", gap: 8, flexShrink: 0 },
  docName: { fontSize: 15, fontWeight: 600, color: "rgba(226, 232, 240, 0.95)", margin: 0, wordBreak: "break-word", lineHeight: 1.2 },
  docStatusPill: { fontSize: 11, textTransform: "uppercase", letterSpacing: 0.6, padding: "2px 8px", borderRadius: 999, border: "1px solid rgba(148, 163, 184, 0.22)", color: "rgba(148, 163, 184, 0.85)" },
  docMetaRow: { display: "flex", flexWrap: "wrap", gap: 2, fontSize: 12, color: "rgba(148, 163, 184, 0.9)", marginTop: 0, lineHeight: 1.25 },
  docMetaItem: { whiteSpace: "nowrap" },
  docActions: { display: "flex", justifyContent: "flex-end", gap: 1, marginTop: 2 },
  docPreviewButton: { font: "inherit", fontSize: 13, padding: "6px 18px", borderRadius: 999, border: "none", background: "linear-gradient(135deg, rgba(165, 180, 252, 0.55), rgba(99, 102, 241, 0.32))", color: "rgba(226, 232, 240, 0.98)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", boxShadow: "0 16px 28px rgba(3, 6, 19, 0.5)" },
  dangerIconButton: { font: "inherit", fontSize: 16, lineHeight: 1, width: 28, height: 28, borderRadius: 14, border: "none", background: "rgba(252, 165, 165, 0.28)", color: "rgba(255, 241, 242, 0.96)", cursor: "pointer", display: "inline-flex", alignItems: "center", justifyContent: "center", padding: 0, boxShadow: "0 14px 26px rgba(239, 68, 68, 0.38)" },
  docIngestRow: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, marginTop: 1, fontSize: 12, color: "rgba(148, 163, 184, 0.85)" },
  docIngestInfo: { flex: 1, lineHeight: 1 },
  docPerfToggle: { font: "inherit", fontSize: 12, display: "inline-flex", alignItems: "center", gap: 2, padding: 0, border: "none", background: "transparent", color: "rgba(148, 163, 184, 0.75)", cursor: "pointer" },
  docPerfToggleActive: { color: "rgba(226, 232, 240, 0.95)" },
  docPerfDetails: { marginTop: 1, marginLeft: 12, fontSize: 11, color: "rgba(148, 163, 184, 0.85)", lineHeight: 1.35 },
  muted: { opacity: 0.75, fontSize: 13, color: "rgba(148, 163, 184, 0.8)" },
  error: { fontSize: 13, color: "#ff8f8f", marginTop: 6 },
  progressTrack: { position: "relative", height: 8, background: "rgba(255, 255, 255, 0.12)", borderRadius: 999, overflow: "hidden" },
  progressFill: { position: "absolute", top: 0, bottom: 0, left: 0, background: "linear-gradient(90deg, rgba(167,139,250,0.9), rgba(59,130,246,0.9))", borderRadius: 999, transition: "width 0.4s ease" },
  progressLabel: { marginTop: 6, fontSize: 11, letterSpacing: 0.4, color: "rgba(226, 232, 240, 0.8)", textTransform: "uppercase" },
  miniProgress: { flex: 1, display: "flex", alignItems: "center", gap: 6, minWidth: 140 },
  miniProgressTrack: { flex: 1, height: 6, borderRadius: 999, background: "rgba(255, 255, 255, 0.12)", overflow: "hidden" },
  miniProgressFill: { height: "100%", background: "linear-gradient(90deg, rgba(167,139,250,0.95), rgba(59,130,246,0.85))", transition: "width 0.4s ease" },
  miniProgressLabel: { fontSize: 11, color: "rgba(226, 232, 240, 0.8)", minWidth: 46, textAlign: "right" },
  previewText: {
    border: "none",
    borderRadius: 22,
    background: "rgba(23, 28, 60, 0.97)",
    padding: 18,
    flex: 1,
    minHeight: 0,
    overflow: "auto",
    boxShadow: "0 22px 38px rgba(0, 0, 0, 0.5), inset 0 0 0 2px rgba(99, 102, 241, 0.14)",
  },
  previewEmpty: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "40px 20px",
    color: "rgba(148, 163, 184, 0.7)",
  },
  markdown: { fontSize: 14, lineHeight: 1.6, color: "#f8fbff", whiteSpace: "normal", wordBreak: "break-word" },
  markdownTable: { width: "100%", borderCollapse: "collapse", margin: "12px 0" },
  tableCell: { border: "1px solid rgba(148, 163, 184, 0.18)", padding: "8px 10px", textAlign: "left" },
  inlineCode: {
    background: "rgba(15, 23, 42, 0.6)",
    borderRadius: 8,
    padding: "2px 6px",
    fontSize: 13,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
  codeBlock: {
    background: "rgba(15, 23, 42, 0.75)",
    borderRadius: 16,
    padding: "14px 16px",
    margin: "12px 0",
    overflowX: "auto",
    fontSize: 13,
    border: "1px solid rgba(148, 163, 184, 0.25)",
  },
};

const markdownRemarkPlugins = [remarkGfm, remarkMath];
const markdownRehypePlugins = [rehypeRaw, rehypeKatex];
const markdownComponents = {
  table: (props) => <table style={styles.markdownTable} {...props} />,
  th: (props) => <th style={styles.tableCell} {...props} />,
  td: (props) => <td style={styles.tableCell} {...props} />,
  code: ({ inline, children = [], ...props }) =>
    inline ? (
      <code style={styles.inlineCode} {...props}>
        {children}
      </code>
    ) : (
      <pre style={styles.codeBlock}>
        <code {...props}>{String(children).replace(/\n$/, "")}</code>
      </pre>
    ),
};

const FALLBACK_PARSER = "mineru";
const IN_PROGRESS_STATUSES = new Set(["processing", "ingesting", "queued", "pending", "running", "parsing", "uploading"]);
const COMPLETED_STATUSES = new Set(["processed", "done", "completed", "ready"]);

export default function IngestPage({ systemStatus = {} }) {
  const navigate = useNavigate();
  const api = useMemo(() => ({
    ingest: "/api/ingest",
    docs: "/api/documents",
    status: (jobId) => `/api/status/${jobId}`,
    retry: (hash) => `/api/ingest/${hash}/retry`,
    previewText: (hash, maxChars = null, parser = FALLBACK_PARSER) => {
      const params = new URLSearchParams({ parser });
      if (typeof maxChars === "number" && maxChars > 0) {
        params.set("max_chars", String(maxChars));
      }
      return `/api/debug/parsed_text/${hash}?${params.toString()}`;
    },
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
  const [limitPreview, setLimitPreview] = useState(true);
  const parser = FALLBACK_PARSER;
  const [expandedPerf, setExpandedPerf] = useState(new Set());
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);
  const selectedDocRef = useRef(null);
  const lastPreviewParamsRef = useRef({ previewMaxChars, limitPreview });

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

  const systemDocs = useMemo(() => (Array.isArray(systemStatus.documents) ? systemStatus.documents : []), [systemStatus.documents]);
  const settingsGroups = useMemo(
    () => systemStatus.settings || systemStatus.settings_snapshot || null,
    [systemStatus.settings, systemStatus.settings_snapshot],
  );
  const displayDocs = docs.length ? docs : systemDocs;
  const jobInfoByHash = useMemo(() => {
    const map = new Map();
    if (Array.isArray(systemStatus.jobs)) {
      systemStatus.jobs.forEach((job) => {
        if (job && job.doc_hash) {
          map.set(job.doc_hash, job);
        }
      });
    }
    return map;
  }, [systemStatus.jobs]);

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
      jobProgress: null,
      jobHash: null,
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
            i === idx ? { ...p, status: "queued", jobId: data.job_id, jobHash: data.hash || p.jobHash } : p
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

  const handlePreview = useCallback(async (doc, { skipSelect = false, preserveContent = false } = {}) => {
    if (!doc || !doc.hash) return;

    if (!skipSelect) {
      setSelectedDoc(prev => (prev && prev.hash === doc.hash ? prev : doc));
    }

    if (!preserveContent) {
      setPreview("");
      setPreviewInfo(null);
    }
    setPreviewError("");
    setPreviewLoading(true);

    try {
      const maxCharsParam = limitPreview ? previewMaxChars : null;
      const res = await fetch(api.previewText(doc.hash, maxCharsParam, parser));
      const data = await readJsonSafe(res);
      if (!res.ok) {
        throw new Error((data && (data.detail || data.error || data.raw)) || `GET preview ${res.status}`);
      }
      const totalEmbeddings = Number(data.total_embeddings ?? data.chunk_count ?? 0) || 0;
      const hasSmall = typeof data.small_embeddings === "number";
      const hasLarge = typeof data.large_embeddings === "number";
      const smallEmbeddings = hasSmall ? Number(data.small_embeddings) : totalEmbeddings;
      const inferredLarge = Math.max(0, totalEmbeddings - smallEmbeddings);
      const largeEmbeddings = hasLarge ? Number(data.large_embeddings) : inferredLarge;
      setPreview(typeof data.preview === "string" ? data.preview : "");
      setPreviewInfo({
        document_name: data.document_name,
        file_size: data.file_size,
        extracted_chars: data.extracted_chars,
        total_tokens: typeof data.total_tokens === "number" ? data.total_tokens : 0,
        chunk_count: data.chunk_count,
        total_embeddings: totalEmbeddings,
        small_embeddings: smallEmbeddings,
        large_embeddings: largeEmbeddings,
        preview_chars: data.preview_chars,
        truncated: !!data.truncated,
        parser,
      });
    } catch (e) {
      setPreviewError(e.message || String(e));
    } finally {
      setPreviewLoading(false);
    }
  }, [api, previewMaxChars, limitPreview, parser]);

  useEffect(() => {
    const prev = lastPreviewParamsRef.current;
    const hasMaxChanged = prev.previewMaxChars !== previewMaxChars;
    const hasLimitChanged = prev.limitPreview !== limitPreview;
    const doc = selectedDocRef.current;

    if (!doc?.hash) {
      lastPreviewParamsRef.current = { previewMaxChars, limitPreview };
      return;
    }
    if (!hasMaxChanged && !hasLimitChanged) {
      return;
    }

    lastPreviewParamsRef.current = { previewMaxChars, limitPreview };
    void handlePreview(doc, { skipSelect: true, preserveContent: true });
  }, [previewMaxChars, limitPreview, handlePreview]);

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
              return { jobId, status: st, payload: data };
            }
          } catch {}
          return { jobId, status: null, payload: null };
        })
      );
      
      const progressMap = new Map();
      results.forEach((r) => {
        if (r && r.jobId && r.payload) {
          progressMap.set(r.jobId, r);
        }
      });
      if (progressMap.size > 0) {
        setUploadProgress(prev =>
          prev.map((entry) => {
            if (!entry.jobId || !progressMap.has(entry.jobId)) return entry;
            const info = progressMap.get(entry.jobId);
            const nextProgress = info.payload?.progress || entry.jobProgress;
            const nextHash = info.payload?.doc_hash || info.payload?.hash || entry.jobHash;
            return {
              ...entry,
              status: info.status || entry.status,
              jobProgress: nextProgress || entry.jobProgress,
              jobHash: nextHash || entry.jobHash,
            };
          })
        );
      }
      
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

  const toggleDiagnostics = useCallback(() => setDiagnosticsOpen((prev) => !prev), []);

  return (
    <div style={{ position: "relative" }}>
      <DiagnosticsPanel open={diagnosticsOpen} onToggle={toggleDiagnostics} groups={settingsGroups} />
      <div className="ingest-layout" style={styles.page}>
      <div style={styles.leftColumn}>
        <section style={{ ...styles.card, ...styles.uploadCard }}>
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
            <div style={styles.uploadProgressList}>
              {uploadProgress.map((p) => {
                const uploadWidth = clampPercent(p.jobProgress?.percent, 0);
                const uploadLabel = formatProgressDetails(p.jobProgress) || p.status;
                return (
                  <div 
                    key={p.id} 
                    style={{ 
                    fontSize: 12, 
                    padding: "8px 14px", 
                    marginBottom: 4, 
                    background: "rgba(57, 69, 130, 0.95)", 
                    borderRadius: 16,
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    boxShadow: "0 18px 30px rgba(4, 7, 20, 0.6)"
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
                  <div style={styles.miniProgress}>
                    <div style={styles.miniProgressTrack}>
                      <div
                        style={{
                          ...styles.miniProgressFill,
                          width: `${uploadWidth}%`,
                        }}
                      />
                    </div>
                    <span style={styles.miniProgressLabel}>
                      {uploadLabel}
                    </span>
                  </div>
                  {p.error && (
                    <span style={{ fontSize: 11, color: "#ff8f8f" }} title={p.error}>
                      {p.error.length > 20 ? p.error.substring(0, 20) + "..." : p.error}
                    </span>
                  )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section style={{ ...styles.card, ...styles.previewCard }}>
        {!selectedDoc && (
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Document Details & Preview</h3>
          </div>
        )}
        <div style={styles.previewBody}>
          {selectedDoc ? (
            <>
              {/* Document Metadata Section */}
              <div style={{ background: "rgba(33, 42, 88, 0.94)", border: "none", borderRadius: 24, padding: 20, marginBottom: 18, boxShadow: "0 26px 46px rgba(5, 9, 25, 0.6)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                  <span style={{ fontSize: 18 }}>📄</span>
                  <strong style={{ fontSize: 15, color: "#e2e8f0", flex: 1, minWidth: 0, wordBreak: "break-word" }}>
                    {previewInfo?.document_name || selectedDoc.name}
                  </strong>
                  <span style={styles.badge}>{parser}</span>
                </div>
                
                {/* Statistics Grid */}
                {previewInfo && (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))", gap: 10, marginBottom: 10 }}>
                    <div style={{ textAlign: "center", padding: "14px 16px", background: "rgba(147, 197, 253, 0.35)", borderRadius: 20, border: "none", boxShadow: "0 18px 34px rgba(3, 6, 20, 0.5)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{prettyBytes(previewInfo.file_size)}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>File Size</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "14px 16px", background: "rgba(147, 197, 253, 0.35)", borderRadius: 20, border: "none", boxShadow: "0 18px 34px rgba(3, 6, 20, 0.5)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{(previewInfo.extracted_chars || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Characters</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "14px 16px", background: "rgba(147, 197, 253, 0.35)", borderRadius: 20, border: "none", boxShadow: "0 18px 34px rgba(3, 6, 20, 0.5)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{Number(previewInfo.total_tokens || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Tokens</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "14px 16px", background: "rgba(147, 197, 253, 0.35)", borderRadius: 20, border: "none", boxShadow: "0 18px 34px rgba(3, 6, 20, 0.5)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{Number(previewInfo.small_embeddings || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Small Embeddings</div>
                    </div>
                    <div style={{ textAlign: "center", padding: "14px 16px", background: "rgba(147, 197, 253, 0.35)", borderRadius: 20, border: "none", boxShadow: "0 18px 34px rgba(3, 6, 20, 0.5)" }}>
                      <div style={{ fontSize: 16, fontWeight: 600, color: "#c7d7ff" }}>{Number(previewInfo.large_embeddings || 0).toLocaleString()}</div>
                      <div style={{ fontSize: 11, color: "rgba(148, 163, 184, 0.85)", marginTop: 2 }}>Large Embeddings</div>
                    </div>
                  </div>
                )}
                
              </div>
              
              {/* Text Preview Section */}
              <div style={styles.previewContent}>
                <div style={styles.previewHeaderRow}>
                  <div style={styles.previewTitle}>Extracted Text</div>
                  <div style={styles.previewLimiterControls}>
                    <label style={styles.previewLimitToggle}>
                      <input
                        type="checkbox"
                        checked={limitPreview}
                        onChange={(e) => setLimitPreview(e.target.checked)}
                        style={styles.previewLimitCheckbox}
                      />
                      <span>Limit preview</span>
                    </label>
                    <input
                      type="number"
                      min={200}
                      max={20000}
                      step={100}
                      value={previewMaxChars}
                      onChange={(e) => setPreviewMaxChars(Number(e.target.value) || 2000)}
                      disabled={!limitPreview}
                      style={{ ...styles.input, width: 100, padding: "6px 10px", fontSize: 13, opacity: limitPreview ? 1 : 0.45 }}
                    />
                  </div>
                </div>
                <div style={{ ...styles.previewText, wordBreak: "break-word", overflowWrap: "anywhere" }}>
                  {previewLoading ? (
                    <span style={styles.muted}>Loading…</span>
                  ) : (preview || "").trim().length > 0 ? (
                    <div style={styles.markdown}>
                      <ReactMarkdown
                        remarkPlugins={markdownRemarkPlugins}
                        rehypePlugins={markdownRehypePlugins}
                        components={markdownComponents}
                      >
                        {preview}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <span style={styles.muted}>No text extracted.</span>
                  )}
                </div>
                {previewError && (<div style={styles.error}>Error: {previewError}</div>)}
              </div>
            </>
          ) : (
            <div style={styles.previewEmpty}>
              <div>
                <div style={{ fontSize: 32, marginBottom: 12 }}>📄</div>
                <div>Select a document from the library below to view its details and preview the extracted text.</div>
              </div>
            </div>
          )}
        </div>
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
                const jobInfo = d.hash ? jobInfoByHash.get(d.hash) : null;
                const jobProgress = jobInfo?.progress;
                const showDocProgress = isInProgress && jobProgress && typeof jobProgress.percent === "number";
                const docProgressPercent = clampPercent(jobProgress?.percent, 0);
                const docProgressWidth = docProgressPercent;
                const docProgressLabel = formatProgressDetails(jobProgress) || statusLabel.toUpperCase();
                
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
                            {perf.ocr_time_sec != null && (
                              <div>• OCR: {perf.ocr_time_sec.toFixed(2)}s</div>
                            )}
                            {perf.ocr_time_sec == null && (
                              <div style={{ opacity: 0.6 }}>• OCR: unavailable</div>
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
                    {showDocProgress && (
                      <div style={{ marginTop: 8 }}>
                        <div style={styles.progressTrack} aria-label="Document progress">
                          <div
                            style={{
                              ...styles.progressFill,
                              width: `${docProgressWidth}%`,
                            }}
                          />
                        </div>
                        <div style={styles.progressLabel}>
                          {docProgressLabel}
                        </div>
                      </div>
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
  </div>
  );
}
