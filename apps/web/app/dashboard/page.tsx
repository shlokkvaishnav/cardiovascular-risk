"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { LogOut, FileDown, Eye } from "lucide-react";
import { SiteFooter, SiteHeader } from "../components/SiteShell";
import { fetchCurrentUser, isLoggedIn, logout, type CurrentUser } from "../lib/authApi";
import { listReports, downloadReportPdf, type ReportSummary } from "../lib/reportsApi";

function formatDate(input: string) {
  return new Date(input).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit" });
}

function riskChipClass(level: ReportSummary["risk_level"]) {
  if (level === "Low") return "chip chip-success";
  if (level === "High") return "chip chip-danger";
  return "chip chip-warning";
}

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isLoggedIn()) {
      router.push("/login");
      return;
    }
    Promise.all([fetchCurrentUser(), listReports()])
      .then(([currentUser, reportList]) => {
        if (!currentUser) {
          router.push("/login");
          return;
        }
        setUser(currentUser);
        setReports(reportList);
      })
      .finally(() => setLoading(false));
  }, [router]);

  const chart = useMemo(() => {
    if (reports.length < 2) return null;
    const ordered = [...reports].reverse();
    const width = 620;
    const height = 160;
    const pad = 24;
    const scores = ordered.map((r) => r.probability * 100);
    const max = Math.max(25, ...scores);
    const min = Math.min(0, ...scores);
    const points = ordered.map((r, i) => {
      const x = pad + (i * (width - pad * 2)) / Math.max(1, ordered.length - 1);
      const y = height - pad - ((r.probability * 100 - min) / Math.max(1, max - min)) * (height - pad * 2);
      return { x, y, r };
    });
    const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
    return { width, height, path, points };
  }, [reports]);

  if (loading) {
    return (
      <main>
        <SiteHeader />
        <section className="section">
          <div className="container">
            <div className="card-panel empty-state">
              <p className="lead">Loading dashboard...</p>
            </div>
          </div>
        </section>
        <SiteFooter />
      </main>
    );
  }

  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container stack">
          <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <div className="dashboard-head">
              <div>
                <div className="mini-badge">Account</div>
                <h1 className="type-h2">Welcome back{user ? `, ${user.email}` : ""}</h1>
                <p className="lead">{reports.length} saved assessment{reports.length === 1 ? "" : "s"}</p>
              </div>
              <button
                type="button"
                className="btn btn-subtle"
                onClick={() => {
                  logout();
                  router.push("/");
                }}
              >
                <LogOut size={16} /> Log out
              </button>
            </div>
          </motion.div>

          {chart && (
            <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
              <h2 className="type-h3">Risk Trend</h2>
              <svg viewBox={`0 0 ${chart.width} ${chart.height}`} className="timeline-chart" role="img" aria-label="Saved risk trend">
                <path d={chart.path} fill="none" stroke="#c0392b" strokeWidth="3" />
                {chart.points.map((p) => (
                  <g key={p.r.id}>
                    <circle cx={p.x} cy={p.y} r="4.2" fill="#922b21" />
                    <title>{`${formatDate(p.r.created_at)}: ${(p.r.probability * 100).toFixed(1)}%`}</title>
                  </g>
                ))}
              </svg>
            </motion.div>
          )}

          <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
            <h2 className="type-h3">Report History</h2>
            {reports.length === 0 ? (
              <p className="lead">
                No saved reports yet. Complete an assessment and choose &quot;Save this result&quot; on the results
                page to start building your history.
              </p>
            ) : (
              <div className="review-list">
                {reports.map((r) => (
                  <div className="review-row" key={r.id}>
                    <span>{formatDate(r.created_at)}</span>
                    <span className={riskChipClass(r.risk_level)}>{r.risk_level.toUpperCase()}</span>
                    <strong className="mono">{(r.probability * 100).toFixed(1)}%</strong>
                    <div style={{ display: "flex", gap: 8 }}>
                      <Link href={`/dashboard/reports/${r.id}`} className="btn btn-subtle">
                        <Eye size={14} /> View
                      </Link>
                      <button type="button" className="btn btn-subtle" onClick={() => downloadReportPdf(r.id)}>
                        <FileDown size={14} /> PDF
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}
