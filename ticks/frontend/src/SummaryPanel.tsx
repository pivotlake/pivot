import React from "react";
import { Summary } from "./types";

const CACHE_LEVELS = ["L1", "LFB", "L2", "L3", "DRAM", "REM", "N-M"];

export default function SummaryPanel({
  summary,
  onClose,
}: {
  summary: Summary;
  onClose: () => void;
}) {
  const per = summary.cache_summary?.per_level || {};
  const totalMem = CACHE_LEVELS.reduce((sum, lvl) => sum + (per[lvl] || 0), 0);
  const l1 = per["L1"] || 0;
  const l1Misses = totalMem - l1;
  const pf = summary.pf_summary;

  return (
    <div className="overlay" onClick={onClose}>
      <div className="panel" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h3>Summary</h3>
          <button className="btn" onClick={onClose}>✕</button>
        </div>

        <Section title="Totals">
          <Row label="Cycle samples" value={summary.total_cycles} />
          <Row label="IBS samples (unweighted)" value={summary.total_unweighted} />
          <Row label="Weighted cost" value={summary.total_weighted} />
        </Section>

        {totalMem > 0 && (
          <Section title="Cache hierarchy">
            <Row label="Total memory samples" value={totalMem} />
            <Row label="L1 hit" value={l1} pct={pct(l1, totalMem)} />
            {l1Misses > 0 && (
              <>
                <Row label="L1 miss" value={l1Misses} pct={pct(l1Misses, totalMem)} />
                {CACHE_LEVELS.filter((lvl) => lvl !== "L1" && lvl !== "N-M").map((lvl) => {
                  const c = per[lvl] || 0;
                  if (!c) return null;
                  return (
                    <Row
                      key={lvl}
                      label={`  of L1 misses: ${lvl}`}
                      value={c}
                      pct={pct(c, l1Misses)}
                    />
                  );
                })}
              </>
            )}
          </Section>
        )}

        {pf && pf.dispatched > 0 && (
          <Section title="SW Prefetch (PMC)">
            <Row label="Dispatched" value={pf.dispatched} />
            <Row label="Filled" value={pf.fills} pct={pct(pf.fills, pf.dispatched)} />
            <Row
              label="Ineffective (DC hit)"
              value={pf.ineffective_dc}
              pct={pct(pf.ineffective_dc, pf.dispatched)}
            />
            <Row
              label="Ineffective (MAB match)"
              value={pf.ineffective_mab}
              pct={pct(pf.ineffective_mab, pf.dispatched)}
            />
          </Section>
        )}

        {summary.skipped_lines.length > 0 && (
          <Section title="Skipped samples">
            {summary.skipped_lines.map((s, i) => (
              <div key={i} className="skipped-line">{s}</div>
            ))}
          </Section>
        )}
      </div>
    </div>
  );
}

function pct(num: number, denom: number) {
  return denom > 0 ? (100 * num) / denom : 0;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="section">
      <div className="section-title">{title}</div>
      <div>{children}</div>
    </div>
  );
}

function Row({
  label,
  value,
  pct,
}: {
  label: string;
  value: number | string;
  pct?: number;
}) {
  return (
    <div className="row">
      <span className="row-label">{label}</span>
      <span className="row-value">{typeof value === "number" ? value.toLocaleString() : value}</span>
      {pct !== undefined && <span className="row-pct">{pct.toFixed(1)}%</span>}
    </div>
  );
}
