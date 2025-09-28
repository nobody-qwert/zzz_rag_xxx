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
  error: { fontSize: 12, color: "#b00020" },
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
    <div style={styles.card}>
      {jobSummaries.length > 0 && (
        <section style={styles.card}>
          <div style={{ ...styles.row, justifyContent: "space-between", marginBottom: 8 }}>
            <h3 style={{ margin: 0, fontSize: 16 }}>Ingestion Jobs</h3>
            <span style={styles.muted}>
              {systemStatus.has_running_jobs
                ? `Processing ${activeJobCount || runningJobs.length} job(s)…`
                : "No active jobs"}
            </span>
          </div>
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            {jobSummaries.map((job) => (
              <li key={job.job_id} style={{ marginBottom: 4 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {job.file || job.job_id}
                  {job.doc_hash && (
                    <span style={styles.muted}> · {shortHash(job.doc_hash)}</span>
                  )}
                  {job.doc_hash && String(job.status || "").toLowerCase().startsWith("error") && (
                    <button
                      style={{ ...styles.button, padding: "4px 8px" }}
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
        </section>
      )}

      <section style={styles.card}>
        <div style={styles.row}>
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} style={styles.input} />
          <button
            onClick={handleUpload}
            disabled={uploading || processing || !file}
            style={styles.button}
          >
            {uploading ? "Uploading…" : processing ? "Processing…" : "Ingest"}
          </button>
        </div>
        <div className="muted" style={styles.muted}>{uploadStatus}</div>
      </section>

      <section style={styles.card}>
        <div style={{ ...styles.row, justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <h3 style={{ margin: 0, fontSize: 16 }}>Documents</h3>
            <span style={styles.muted}>
              {`${systemStatus.docs_count || 0} ready / ${systemStatus.total_docs || docs.length}`}
            </span>
          </div>
          <button onClick={refreshDocs} disabled={docsLoading} style={styles.button}>
            {docsLoading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
        <div style={{ ...styles.docs, ...(docs.length ? {} : styles.muted) }}>
          {docs.length === 0 ? (
            <div>No documents yet.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {docs.map((d) => (
                <li key={d.hash || d.stored_name || d.name} style={{ marginBottom: 6 }}>
                  <div title={d.path}>
                    {d.name}{" "}
                    <span style={styles.muted}>
                      {`${prettyBytes(d.size)} · ${(d.status || "unknown").toLowerCase()}`}
                      {d.hash ? ` · ${shortHash(d.hash)}` : ""}
                    </span>
                    {String(d.status || "").toLowerCase() === "error" && d.hash && (
                      <button
                        style={{ ...styles.button, marginLeft: 8, padding: "4px 8px" }}
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
