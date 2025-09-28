import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

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

function formatDate(value) {
  if (!value) return "";
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return "";
  return dt.toLocaleString();
}

const styles = {
  page: {
    display: "grid",
    gap: 24,
    gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
    alignItems: "start",
  },
  leftColumn: {
    display: "grid",
    gap: 24,
  },
  card: {
    border: "1px solid rgba(148, 163, 184, 0.18)",
    borderRadius: 18,
    padding: "24px 28px",
    background: "rgba(13, 16, 24, 0.9)",
    boxShadow: "0 22px 45px rgba(2, 6, 23, 0.35)",
  },
  sectionHeader: {
    display: "flex",
    alignItems: "baseline",
    justifyContent: "space-between",
    gap: 16,
    marginBottom: 16,
  },
  sectionTitle: {
    margin: 0,
    fontSize: 18,
    fontWeight: 600,
    letterSpacing: 0.2,
  },
  badge: {
    padding: "4px 10px",
    borderRadius: 999,
    border: "1px solid rgba(148, 163, 184, 0.4)",
    fontSize: 12,
    color: "rgba(148, 163, 184, 0.85)",
    whiteSpace: "nowrap",
  },
  row: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    flexWrap: "wrap",
  },
  button: {
    font: "inherit",
    padding: "10px 16px",
    borderRadius: 12,
    border: "1px solid rgba(84, 105, 255, 0.45)",
    background: "linear-gradient(135deg, rgba(84, 105, 255, 0.18), rgba(84, 105, 255, 0.05))",
    color: "#c7d7ff",
    cursor: "pointer",
    transition: "transform 0.15s ease, box-shadow 0.15s ease",
  },
  subtleButton: {
    font: "inherit",
    padding: "6px 10px",
    borderRadius: 10,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(15, 17, 23, 0.6)",
    color: "rgba(226, 232, 240, 0.88)",
    cursor: "pointer",
  },
  input: {
    font: "inherit",
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid rgba(148, 163, 184, 0.35)",
    background: "rgba(9, 11, 18, 0.8)",
    color: "inherit",
    minWidth: 0,
  },
  feedback: {
    marginTop: 10,
    fontSize: 13,
    color: "rgba(148, 163, 184, 0.85)",
  },
  docs: {
    maxHeight: "55vh",
    overflow: "auto",
    border: "1px solid rgba(148, 163, 184, 0.12)",
    borderRadius: 14,
    padding: 16,
    background: "rgba(9, 11, 18, 0.75)",
  },
  listItem: {
    padding: "12px 10px",
    borderRadius: 10,
    background: "rgba(23, 25, 35, 0.75)",
    border: "1px solid rgba(148, 163, 184, 0.08)",
    marginBottom: 8,
  },
  muted: {
    opacity: 0.75,
    fontSize: 13,
    color: "rgba(148, 163, 184, 0.8)",
  },
  error: {
    fontSize: 13,
    color: "#ff8f8f",
    marginTop: 6,
  },
};

export default function IngestPage({ systemStatus = {} }) {
  const api = useMemo(() => ({
    ingest: "/api/ingest",
    docs: "/api/docs",
    status: (jobId) => `/api/status/${jobId}`,
    retry: (hash) => `/api/ingest/${hash}/retry`,
  }), []);

  const [docs, setDocs] = useState([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [jobId, setJobId] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [retryingHash, setRetryingHash] = useState(null);

  const jobSummaries = Array.isArray(systemStatus.jobs) ? systemStatus.jobs : [];
  const runningJobs = Array.isArray(systemStatus.running_jobs) ? systemStatus.running_jobs : [];
  const activeJobCount = jobSummaries.filter((j) => {
    const status = String(j.status || "").toLowerCase();
    return status === "running" || status === "queued";
  }).length;

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res = await fetch(api.docs);
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || `GET /docs ${res.status}`);
      setDocs(Array.isArray(data) ? data : []);
    } catch {
      setDocs([]);
    } finally {
      setDocsLoading(false);
    }
  }, [api.docs]);

  useEffect(() => {
    void refreshDocs();
  }, [refreshDocs]);

  const systemDocs = useMemo(() => (Array.isArray(systemStatus.documents) ? systemStatus.documents : []), [systemStatus.documents]);

  const displayDocs = docs.length ? docs : systemDocs;

  useEffect(() => {
    if (systemDocs.length > 0) {
      setDocs((existing) => {
        if (existing.length === 0) return systemDocs;
        if (existing.length !== systemDocs.length) return systemDocs;
        return existing;
      });
    }
  }, [systemDocs]);

  const docsCountRef = useRef(systemStatus.docs_count || 0);

  useEffect(() => {
    const currentCount = systemStatus.docs_count || 0;
    if (currentCount !== docsCountRef.current) {
      docsCountRef.current = currentCount;
      void refreshDocs();
    }
  }, [systemStatus.docs_count, refreshDocs]);

  // Upload file and start background job
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
      const res = await fetch(api.ingest, { method: "POST", body: form });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);

      if (data.status === "skipped") {
        setUploadStatus(`Already ingested (hash ${shortHash(data.hash)})`);
        setProcessing(false);
        setJobId(null);
        return;
      }

      if (data.status === "already_processing") {
        setProcessing(true);
        setJobId(data.job_id || null);
        setUploadStatus(
          data.job_id
            ? `Ingestion already running (job ${data.job_id})`
            : "Ingestion already running"
        );
        return;
      }

      if (!data.job_id) {
        throw new Error("Upload did not return a job identifier");
      }

      // Backend returns job_id
      setJobId(data.job_id);
      setProcessing(true);
      setUploadStatus(`Queued (job ${data.job_id})`);
    } catch (e) {
      console.error(e);
      setUploadStatus(`Upload failed: ${e.message || String(e)}`);
    } finally {
      setUploading(false);
      setFile(null);
    }
  }

  const handleRetry = useCallback(async (hash) => {
    if (!hash) return;
    setRetryingHash(hash);
    setUploadStatus(`Retrying ingestion for ${shortHash(hash)}...`);
    try {
      const res = await fetch(api.retry(hash), { method: "POST" });
      const data = await readJsonSafe(res);
      if (!res.ok) throw new Error((data && (data.detail || data.error || data.raw)) || res.statusText);

      setJobId(data.job_id || null);
      setProcessing(true);
      setUploadStatus(`Re-queued (job ${data.job_id})`);
      void refreshDocs();
    } catch (e) {
      console.error(e);
      setUploadStatus(`Retry failed: ${e.message || String(e)}`);
    } finally {
      setRetryingHash(null);
    }
  }, [api, refreshDocs]);

  useEffect(() => {
    if (!jobId) return;

    const jobInfo = jobSummaries.find((j) => j.job_id === jobId);
    if (!jobInfo) return;

    const status = String(jobInfo.status || "").toLowerCase();

    if (status.startsWith("error")) {
      setProcessing(false);
      setJobId(null);
      setUploadStatus(`Ingestion error: ${jobInfo.status}`);
    } else if (status === "done") {
      setProcessing(false);
      setJobId(null);
      setUploadStatus("Ingestion complete.");
      void refreshDocs();
    } else if (status === "queued") {
      setProcessing(true);
      setUploadStatus(`Queued… (job ${jobInfo.job_id})`);
    } else if (status === "running") {
      setProcessing(true);
      setUploadStatus(`Processing… (job ${jobInfo.job_id})`);
    } else {
      setProcessing(true);
      setUploadStatus(`${jobInfo.status || "Processing"}… (job ${jobInfo.job_id})`);
    }
  }, [jobId, jobSummaries, refreshDocs]);

  return (
    <div style={styles.page}>
      <div style={styles.leftColumn}>
        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Ingestion Jobs</h3>
            <span style={styles.badge}>
              {systemStatus.has_running_jobs
                ? `Processing ${activeJobCount || runningJobs.length} job(s)`
                : jobSummaries.length
                ? `${jobSummaries.length} job${jobSummaries.length > 1 ? "s" : ""} tracked`
                : "Idle"}
            </span>
          </div>
          {jobSummaries.length === 0 ? (
            <div style={styles.muted}>No jobs yet. Upload a document to kick things off.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none" }}>
              {jobSummaries.map((job) => (
                <li key={job.job_id} style={styles.listItem}>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <div>
                      <strong>{job.file || job.job_id}</strong>
                      {job.doc_hash && (
                        <span style={styles.muted}> · {shortHash(job.doc_hash)}</span>
                      )}
                    </div>
                    {job.doc_hash && String(job.status || "").toLowerCase().startsWith("error") && (
                      <button
                        style={styles.subtleButton}
                        onClick={() => handleRetry(job.doc_hash)}
                        disabled={retryingHash === job.doc_hash}
                      >
                        {retryingHash === job.doc_hash ? "Retrying…" : "Retry"}
                      </button>
                    )}
                  </div>
                  <div style={styles.muted}>
                    {job.status || "unknown"}
                    {job.error ? ` – ${job.error}` : ""}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section style={styles.card}>
          <div style={styles.sectionHeader}>
            <h3 style={styles.sectionTitle}>Upload Documents</h3>
            <span style={styles.badge}>
              {processing ? "Processing" : uploading ? "Uploading" : "Ready"}
            </span>
          </div>
          <div style={styles.row}>
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} style={styles.input} />
            <button
              onClick={handleUpload}
              disabled={uploading || processing || !file}
              style={{
                ...styles.button,
                opacity: uploading || processing || !file ? 0.6 : 1,
                cursor: uploading || processing || !file ? "not-allowed" : "pointer",
              }}
            >
              {uploading ? "Uploading…" : processing ? "Processing…" : "Ingest"}
            </button>
          </div>
          <div style={styles.feedback}>{uploadStatus || "Supported formats include PDFs, text files, and markdown."}</div>
        </section>
      </div>

      <section style={styles.card}>
        <div style={styles.sectionHeader}>
          <div>
            <h3 style={styles.sectionTitle}>Document Library</h3>
            <span style={styles.muted}>
              {`${systemStatus.docs_count || 0} ready / ${systemStatus.total_docs || displayDocs.length}`}
            </span>
          </div>
          <button
            onClick={refreshDocs}
            disabled={docsLoading}
            style={{
              ...styles.subtleButton,
              opacity: docsLoading ? 0.6 : 1,
              cursor: docsLoading ? "wait" : "pointer",
            }}
          >
            {docsLoading ? "Refreshing…" : "Refresh"}
          </button>
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
                    <span style={styles.muted}>
                      {`${prettyBytes(d.size)} · ${(d.status || "unknown").toLowerCase()}`}
                      {d.hash ? ` · ${shortHash(d.hash)}` : ""}
                    </span>
                    {String(d.status || "").toLowerCase() === "error" && d.hash && (
                      <button
                        style={{ ...styles.subtleButton, marginLeft: 8 }}
                        onClick={() => handleRetry(d.hash)}
                        disabled={retryingHash === d.hash}
                      >
                        {retryingHash === d.hash ? "Retrying…" : "Retry"}
                      </button>
                    )}
                  </div>
                  {d.last_ingested_at && (
                    <div style={styles.muted}>
                      Last ingested {formatDate(d.last_ingested_at)}
                    </div>
                  )}
                  {d.error && (
                    <div style={styles.error}>Error: {d.error}</div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}
