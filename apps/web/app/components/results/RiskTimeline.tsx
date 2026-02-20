"use client";

import type { AssessmentHistoryEntry } from "../../hooks/useAssessmentHistory";

type Props = {
  history: AssessmentHistoryEntry[];
  onExportCsv: () => void;
  onClearHistory: () => void;
};

function formatDate(input: string) {
  const date = new Date(input);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "2-digit" });
}

export function RiskTimeline(props: Props) {
  const { history, onExportCsv, onClearHistory } = props;

  if (history.length === 0) {
    return (
      <div className="timeline-card">
        <h2 className="type-h3">Risk Timeline</h2>
        <p className="lead">No saved assessments yet. Complete multiple assessments to see your trend.</p>
      </div>
    );
  }

  const width = 620;
  const height = 180;
  const pad = 24;
  const maxScore = Math.max(25, ...history.map((entry) => entry.score));
  const minScore = Math.min(0, ...history.map((entry) => entry.score));

  const points = history.map((entry, index) => {
    const x = pad + (index * (width - pad * 2)) / Math.max(1, history.length - 1);
    const y = height - pad - ((entry.score - minScore) / Math.max(1, maxScore - minScore)) * (height - pad * 2);
    return { x, y, entry };
  });

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");

  const first = history[0];
  const last = history[history.length - 1];
  const delta = last.score - first.score;
  const elapsedDays = Math.max(
    1,
    Math.round((new Date(last.timestamp).getTime() - new Date(first.timestamp).getTime()) / (1000 * 60 * 60 * 24)),
  );
  const elapsedMonths = Math.max(1, Math.round(elapsedDays / 30));

  return (
    <div className="timeline-card">
      <div className="timeline-head">
        <h2 className="type-h3">Risk Timeline</h2>
        <span className="chip">{history.length} saved assessments</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="timeline-chart" role="img" aria-label="Risk trend chart">
        <path d={path} fill="none" stroke="#c0392b" strokeWidth="3" />
        {points.map((point) => (
          <g key={point.entry.id}>
            <circle cx={point.x} cy={point.y} r="4.2" fill="#922b21" />
            <title>{`${formatDate(point.entry.timestamp)}: ${point.entry.score.toFixed(1)}%`}</title>
          </g>
        ))}
      </svg>
      <div className="timeline-labels">
        {history.map((entry) => (
          <span key={entry.id}>{formatDate(entry.timestamp)}</span>
        ))}
      </div>
      <div className="timeline-progress">
        <strong className="mono">{Math.abs(delta).toFixed(1)}%</strong>
        <span>
          {delta <= 0 ? "risk reduction" : "risk increase"} over {elapsedMonths} month
          {elapsedMonths === 1 ? "" : "s"}
        </span>
      </div>
      <div className="timeline-actions">
        <button type="button" className="btn btn-subtle" onClick={onExportCsv}>
          Export CSV
        </button>
        <button type="button" className="btn btn-subtle" onClick={onClearHistory}>
          Clear History
        </button>
      </div>
    </div>
  );
}

