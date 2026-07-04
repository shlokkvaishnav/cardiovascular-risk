"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowLeft, FileDown } from "lucide-react";
import { SiteFooter, SiteHeader } from "../../../components/SiteShell";
import { isLoggedIn } from "../../../lib/authApi";
import { getReport, downloadReportPdf, type ReportDetail } from "../../../lib/reportsApi";
import { humanizeContributorLabel } from "../../../lib/predictApi";

export default function ReportDetailPage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const [report, setReport] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isLoggedIn()) {
      router.push("/login");
      return;
    }
    getReport(params.id)
      .then(setReport)
      .catch(() => setError("Report not found or you don't have access to it."));
  }, [params.id, router]);

  return (
    <main>
      <SiteHeader />
      <section className="section">
        <div className="container stack">
          <Link href="/dashboard" className="btn btn-subtle" style={{ width: "fit-content" }}>
            <ArrowLeft size={14} /> Back to dashboard
          </Link>

          {error && (
            <div className="card-panel empty-state">
              <p className="lead">{error}</p>
            </div>
          )}

          {report && (
            <>
              <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <div className="dashboard-head">
                  <div>
                    <h1 className="type-h2">Saved Assessment</h1>
                    <p className="lead">{new Date(report.created_at).toLocaleString()}</p>
                  </div>
                  <button type="button" className="btn btn-primary" onClick={() => downloadReportPdf(report.id)}>
                    <FileDown size={16} /> Download PDF
                  </button>
                </div>
                <p className="type-h3" style={{ marginTop: 12 }}>
                  {report.risk_level.toUpperCase()} RISK -- {(report.probability * 100).toFixed(1)}%
                </p>
              </motion.div>

              <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <h2 className="type-h3">Submitted Inputs</h2>
                <div className="review-list">
                  {Object.entries(report.inputs).map(([key, value]) => (
                    <div className="review-row" key={key}>
                      <span>{key}</span>
                      <strong>{String(value)}</strong>
                    </div>
                  ))}
                </div>
              </motion.div>

              {report.top_contributors && report.top_contributors.length > 0 && (
                <motion.div className="card-panel" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                  <h2 className="type-h3">Top Contributing Factors (SHAP)</h2>
                  <div className="driver-list">
                    {report.top_contributors.map((entry) => {
                      const [rawKey, value] = Object.entries(entry)[0] ?? ["", 0];
                      return (
                        <div className="driver-row" key={rawKey}>
                          <span>{humanizeContributorLabel(rawKey)}</span>
                          <strong className="mono">{value >= 0 ? "+" : ""}{value.toFixed(3)}</strong>
                        </div>
                      );
                    })}
                  </div>
                </motion.div>
              )}
            </>
          )}
        </div>
      </section>
      <SiteFooter />
    </main>
  );
}
