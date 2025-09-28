import React, { useEffect, useMemo, useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";

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

import IngestPage from "./IngestPage";
import ChatPage from "./ChatPage";

function NavigationBar({ systemStatus, currentPath, onNavigate }) {
  const canNavigateToChat = systemStatus.ready && systemStatus.docs_count > 0 && !systemStatus.has_running_jobs;
  const isOnChat = currentPath === "/chat";
  const isOnIngest = currentPath === "/ingest";

  return (
    <nav style={{ padding: "8px 16px", borderBottom: "1px solid #4443", display: "flex", gap: 16, alignItems: "center" }}>
      <button
        onClick={() => onNavigate("/ingest")}
        disabled={isOnChat && systemStatus.asking}
        style={{
          ...styles.navButton,
          background: isOnIngest ? "#4a90e22a" : "transparent",
          opacity: (isOnChat && systemStatus.asking) ? 0.5 : 1,
          cursor: (isOnChat && systemStatus.asking) ? "not-allowed" : "pointer"
        }}
      >
        Ingest Docs
      </button>
      
      {/* Only show chat button when documents are ready and no jobs are running */}
      {canNavigateToChat && (
        <button
          onClick={() => onNavigate("/chat")}
          style={{
            ...styles.navButton,
            background: isOnChat ? "#4a90e22a" : "transparent"
          }}
        >
          Chat
        </button>
      )}

      {systemStatus.has_running_jobs && (
        <span style={styles.muted}>
          Processing {systemStatus.running_jobs.length} job(s)...
        </span>
      )}
      
      {/* Show status message when chat is not available */}
      {(!systemStatus.ready || systemStatus.docs_count === 0) && !systemStatus.has_running_jobs && (
        <span style={styles.muted}>
          Upload documents to enable chat
        </span>
      )}
    </nav>
  );
}

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();
  
  const [systemStatus, setSystemStatus] = useState({
    ready: false,
    has_running_jobs: false,
    running_jobs: [],
    total_jobs: 0,
    asking: false,
    docs_count: 0,
    total_docs: 0,
    jobs: [],
    llm_ready: false,
  });

  const api = useMemo(() => ({
    status: "/api/status",
    warmup: "/api/warmup"
  }), []);

  // Poll system status
  useEffect(() => {
    let interval;
    
    const checkStatus = async () => {
      try {
        const res = await fetch(api.status);
        const data = await readJsonSafe(res);
        if (!res.ok) {
          throw new Error((data && (data.detail || data.error || data.raw)) || `GET /status ${res.status}`);
        }

        setSystemStatus(prev => ({
          ...prev,
          ready: !!data.ready,
          has_running_jobs: !!data.has_running_jobs,
          running_jobs: data.running_jobs || [],
          total_jobs: data.total_jobs || 0,
          docs_count: typeof data.docs_count === "number" ? data.docs_count : prev.docs_count,
          total_docs: typeof data.total_docs === "number" ? data.total_docs : prev.total_docs,
          jobs: Array.isArray(data.jobs) ? data.jobs : prev.jobs,
          llm_ready: !!data.llm_ready,
        }));
      } catch (e) {
        console.error("Failed to check system status:", e);
        setSystemStatus(prev => ({
          ...prev,
          ready: false
        }));
      }
    };

    // Initial check
    checkStatus();
    
    // Poll every 2 seconds
    interval = setInterval(checkStatus, 2000);
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [api.status]);

  const handleNavigate = (path) => {
    // Prevent navigation to chat if not ready or jobs running
    if (path === "/chat" && (!systemStatus.ready || systemStatus.has_running_jobs)) {
      console.log("Navigation to chat blocked: not ready or jobs running");
      return;
    }
    
    // Prevent navigation away from chat if asking
    if (location.pathname === "/chat" && systemStatus.asking) {
      console.log("Navigation blocked: currently asking");
      return;
    }
    
    navigate(path);
  };

  // Redirect to ingest if trying to access chat without ready docs
  useEffect(() => {
    if (
      location.pathname === "/chat" &&
      (!systemStatus.ready || systemStatus.has_running_jobs || systemStatus.docs_count === 0)
    ) {
      navigate("/ingest", { replace: true });
    }
  }, [
    location.pathname,
    systemStatus.ready,
    systemStatus.has_running_jobs,
    systemStatus.docs_count,
    navigate
  ]);

  const updateAskingStatus = (asking) => {
    setSystemStatus(prev => ({ ...prev, asking }));
  };

  return (
    <div>
      <header style={styles.header}>
        <h1 style={{ margin: 0, fontSize: 16 }}>RAG-Anything Demo</h1>
        <span style={styles.muted}>Upload documents, then ask questions</span>
      </header>

      <NavigationBar 
        systemStatus={systemStatus}
        currentPath={location.pathname}
        onNavigate={handleNavigate}
      />

      <main style={styles.main}>
        <Routes>
          <Route path="/ingest" element={<IngestPage systemStatus={systemStatus} />} />
          <Route
            path="/chat"
            element={
              systemStatus.ready && systemStatus.docs_count > 0 && !systemStatus.has_running_jobs ? (
                <ChatPage
                  onAskingChange={updateAskingStatus}
                  warmupApi={api.warmup}
                  llmReady={systemStatus.llm_ready}
                />
              ) : (
                <Navigate to="/ingest" replace />
              )
            }
          />
          <Route path="/" element={<Navigate to="/ingest" replace />} />
        </Routes>
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
  header: {
    padding: "12px 16px",
    borderBottom: "1px solid #4443",
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  main: {
    padding: 16,
  },
  navButton: {
    font: "inherit",
    padding: "8px 12px",
    borderRadius: 6,
    border: "1px solid #4443",
    background: "transparent",
    color: "#4a90e2",
    textDecoration: "none",
  },
  muted: { 
    opacity: 0.7, 
    fontSize: 12,
    color: "#666"
  },
};
