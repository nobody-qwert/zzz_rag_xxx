import React, { useEffect, useMemo, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";

async function readJsonSafe(res) {
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try { return await res.json(); } catch {}
  }
  const raw = await res.text();
  return { nonJson: true, raw };
}

import IngestPage from "./IngestPage";
import ChatPage from "./ChatPage";

function NavigationBar({ systemStatus, currentPath, onNavigate, statusLabel }) {
  const canNavigateToChat = systemStatus.ready && systemStatus.docs_count > 0 && !systemStatus.has_running_jobs;
  const isOnChat = currentPath === "/chat";
  const isOnIngest = currentPath === "/ingest";
  return (
    <header style={styles.headerShell}>
      <div style={styles.topRow}>
        <div style={styles.topLeft}>
          <div style={styles.navButtonGroup}>
            <button
              onClick={() => onNavigate("/ingest")}
              disabled={isOnChat && systemStatus.asking}
              style={{
                ...styles.navButton,
                background: isOnIngest ? "rgba(84,105,255,0.18)" : "transparent",
                opacity: (isOnChat && systemStatus.asking) ? 0.5 : 1,
              }}
            >
              Ingest Docs
            </button>
            {canNavigateToChat && (
              <button
                onClick={() => onNavigate("/chat")}
                style={{
                  ...styles.navButton,
                  background: isOnChat ? "rgba(84,105,255,0.18)" : "transparent",
                }}
              >
                Chat
              </button>
            )}
          </div>
        </div>
        <div style={styles.topCenter}>
          <h1 style={styles.appTitle}>RAG MinerU</h1>
          <span style={styles.tagline}>Upload documents, then ask questions</span>
        </div>
        <div style={styles.topRight}>
          <div style={styles.statusPill}>{statusLabel}</div>
        </div>
      </div>
    </header>
  );
}

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();
  const [systemStatus, setSystemStatus] = useState({ ready: false, has_running_jobs: false, running_jobs: [], total_jobs: 0, asking: false, docs_count: 0, total_docs: 0, jobs: [], llm_ready: false, documents: [] });

  const api = useMemo(() => ({ status: "/api/status", warmup: "/api/warmup" }), []);

  useEffect(() => {
    let interval;
    const checkStatus = async () => {
      try {
        // synthesize status from separate endpoints
        const [docsRes, readyRes] = await Promise.all([
          fetch("/api/documents"),
          fetch("/api/ready"),
        ]);
        const [docs, readyInfo] = await Promise.all([docsRes, readyRes].map(readJsonSafe));
        const docsArr = Array.isArray(docs) ? docs : [];
        const ready = !!(readyInfo && readyInfo.ready);

        // jobs listing is optional; try and ignore errors
        let jobs = [];
        try {
          const r = await fetch("/api/status");
          const j = await readJsonSafe(r);
          if (r.ok && j && Array.isArray(j.jobs)) jobs = j.jobs;
        } catch {}

        setSystemStatus(prev => ({
          ...prev,
          ready,
          has_running_jobs: jobs.some(x => x.status === "running" || x.status === "queued"),
          running_jobs: jobs.filter(x => x.status === "running" || x.status === "queued"),
          total_jobs: jobs.length,
          docs_count: docsArr.filter(d => String(d.status || "").toLowerCase() === "processed").length,
          total_docs: docsArr.length,
          jobs,
          documents: docsArr,
        }));
      } catch (e) {
        console.error("status error", e);
      }
    };
    checkStatus();
    interval = setInterval(checkStatus, 2000);
    return () => interval && clearInterval(interval);
  }, [api.status]);

  const handleNavigate = (path) => {
    if (path === "/chat" && (!systemStatus.ready || systemStatus.has_running_jobs)) return;
    if (location.pathname === "/chat" && systemStatus.asking) return;
    navigate(path);
  };

  useEffect(() => {
    if (location.pathname === "/chat" && (!systemStatus.ready || systemStatus.has_running_jobs || systemStatus.docs_count === 0)) {
      navigate("/ingest", { replace: true });
    }
  }, [location.pathname, systemStatus.ready, systemStatus.has_running_jobs, systemStatus.docs_count, navigate]);

  const updateAskingStatus = (asking) => setSystemStatus(prev => ({ ...prev, asking }));
  const statusLabel = systemStatus.has_running_jobs ? "Processing" : systemStatus.ready ? "Ready" : "Standby";

  return (
    <div style={styles.appShell}>
      <NavigationBar
        systemStatus={systemStatus}
        currentPath={location.pathname}
        onNavigate={handleNavigate}
        statusLabel={statusLabel}
      />

      <main style={styles.main}>
        <div style={styles.content}>
          <Routes>
            <Route path="/ingest" element={<IngestPage systemStatus={systemStatus} />} />
            <Route path="/chat" element={systemStatus.ready && systemStatus.docs_count > 0 && !systemStatus.has_running_jobs ? (
              <ChatPage onAskingChange={updateAskingStatus} warmupApi={api.warmup} llmReady={systemStatus.llm_ready} documents={systemStatus.documents} />
            ) : (<Navigate to="/ingest" replace />)} />
            <Route path="/" element={<Navigate to="/ingest" replace />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

const styles = {
  appShell: { minHeight: "100vh", display: "flex", flexDirection: "column", background: "radial-gradient(circle at top left, rgba(45,55,95,0.35), rgba(15,17,23,0.98) 55%)", color: "#f4f6fb" },
  headerShell: { borderBottom: "1px solid rgba(148,163,184,0.16)", background: "rgba(15,17,23,0.92)", backdropFilter: "blur(12px)" },
  topRow: { maxWidth: "1200px", margin: "0 auto", padding: "18px 32px", display: "grid", gridTemplateColumns: "auto 1fr auto", alignItems: "center", gap: 24 },
  topLeft: { display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" },
  navButtonGroup: { display: "flex", gap: 10, flexWrap: "wrap" },
  topCenter: { textAlign: "center" },
  topRight: { display: "flex", justifyContent: "flex-end", alignItems: "center" },
  appTitle: { margin: 0, fontSize: 22, fontWeight: 600, letterSpacing: 0.3 },
  tagline: { display: "block", fontSize: 14, color: "rgba(148,163,184,0.82)", marginTop: 4 },
  statusPill: { padding: "6px 12px", borderRadius: 999, border: "1px solid rgba(56,189,248,0.35)", color: "rgba(125,211,252,0.92)", fontSize: 12, letterSpacing: 0.5, textTransform: "uppercase" },
  main: { flex: 1, padding: "32px 24px 48px" },
  content: { maxWidth: "1200px", margin: "0 auto", width: "100%" },
  navButton: { font: "inherit", padding: "10px 18px", borderRadius: 14, border: "1px solid rgba(84,105,255,0.4)", background: "transparent", color: "#c7d7ff", textDecoration: "none", transition: "background 0.2s ease, border 0.2s ease" },
  muted: { fontSize: 13, color: "rgba(148,163,184,0.8)" },
};
