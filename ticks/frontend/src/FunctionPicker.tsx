import React, { useMemo, useState } from "react";
import { Summary } from "./types";
import SummaryPanel from "./SummaryPanel";

interface Props {
  summary: Summary;
  onPick: (name: string) => void;
}

export default function FunctionPicker({ summary, onPick }: Props) {
  const [query, setQuery] = useState("");
  const [showSummary, setShowSummary] = useState(false);

  const filtered = useMemo(() => {
    if (!query) return summary.functions;
    const q = query.toLowerCase();
    return summary.functions.filter((f) => f.name.toLowerCase().includes(q));
  }, [query, summary.functions]);

  return (
    <div className="container">
      <div className="summary-bar">
        <h2>{summary.functions.length} functions</h2>
        <div className="subtle">
          {summary.total_cycles.toLocaleString()} cycle samples ·{" "}
          {summary.total_unweighted.toLocaleString()} IBS samples ·{" "}
          {summary.binary}
        </div>
        <div className="row-controls">
          <input
            className="search-input"
            placeholder="filter functions…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className="btn" onClick={() => setShowSummary(true)}>
            summary
          </button>
        </div>
      </div>

      <div className="func-list">
        <div className="func-row head">
          <div className="pct">Cycles%</div>
          <div className="num">Cycles</div>
          <div className="num">Samples</div>
          <div className="name">Function</div>
        </div>
        {filtered.map((f) => (
          <div
            key={f.name}
            className="func-row"
            onClick={() => onPick(f.name)}
          >
            <div className="pct">{f.cycles_pct.toFixed(2)}%</div>
            <div className="num">{f.cycles.toLocaleString()}</div>
            <div className="num">{f.samples.toLocaleString()}</div>
            <div className="name" title={f.name}>{f.name}</div>
          </div>
        ))}
      </div>

      {showSummary && (
        <SummaryPanel summary={summary} onClose={() => setShowSummary(false)} />
      )}
    </div>
  );
}
