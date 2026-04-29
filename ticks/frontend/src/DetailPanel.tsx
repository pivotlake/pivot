import React from "react";
import { InstructionLine } from "./types";

const CACHE_LEVELS = ["L1", "LFB", "L2", "L3", "DRAM", "REM", "N-M"];

interface Props {
  line: InstructionLine;
  onClose: () => void;
}

export default function DetailPanel({ line, onClose }: Props) {
  const s = line.stats;
  const ibs = s.ibs;
  const totalCache = CACHE_LEVELS.reduce(
    (sum, lvl) => sum + (s.cache_counts[lvl] || 0),
    0,
  );

  const tlb = Object.entries(s.tlb_counts).filter(
    ([k, v]) => v > 0 && k !== "NA",
  );
  const ops = Object.entries(s.op_counts).filter(([k, v]) => v > 0 && k !== "N/A");
  const snoop = Object.entries(s.snoop_counts).filter(
    ([k, v]) => v > 0 && k !== "N/A" && k !== "None",
  );

  return (
    <div className="overlay" onClick={onClose}>
      <div className="panel" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h3>
            {line.addr}: {line.mnem} {line.operands}
          </h3>
          <button className="btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <Section title={`Total samples: ${s.total_samples}`}>
          <Row label="Cycles" value={line.cycles} />
          <Row label="Cycles %" value={`${line.cycles_pct.toFixed(2)}%`} />
        </Section>

        {totalCache > 0 && (
          <Section title="Cache level">
            {CACHE_LEVELS.map((lvl) => {
              const c = s.cache_counts[lvl] || 0;
              if (!c) return null;
              return (
                <Row
                  key={lvl}
                  label={lvl}
                  value={c}
                  pct={pct(c, totalCache)}
                  level={lvl}
                />
              );
            })}
          </Section>
        )}

        {tlb.length > 0 && (
          <Section title="TLB">
            {tlb.map(([k, v]) => (
              <Row key={k} label={tlbLabel(k)} value={v} />
            ))}
          </Section>
        )}

        {ops.length > 0 && (
          <Section title="Operations">
            {ops.map(([k, v]) => (
              <Row key={k} label={k} value={v} />
            ))}
          </Section>
        )}

        {snoop.length > 0 && (
          <Section title="Snoop (cache coherency)">
            {snoop.map(([k, v]) => (
              <Row key={k} label={k} value={v} />
            ))}
          </Section>
        )}

        {s.locked_count > 0 && (
          <Section title="Misc">
            <Row label="Locked ops" value={s.locked_count} />
          </Section>
        )}

        {ibs.sample_count > 0 && (
          <Section title="IBS latencies (cycles)">
            <Row label="Tag-to-retire avg" value={ibs.avg_tag_to_ret.toFixed(0)} />
            <Row label="Comp-to-retire avg" value={ibs.avg_comp_to_ret.toFixed(0)} />
            <Row label="DC miss latency avg" value={ibs.avg_dc_miss_lat.toFixed(0)} hot={ibs.avg_dc_miss_lat > 0} />
            <Row label="TLB refill latency avg" value={ibs.avg_tlb_refill_lat.toFixed(0)} />
            <Row label="Avg MABs in flight" value={ibs.avg_mabs.toFixed(1)} />
            <Row label="Max MABs" value={ibs.mabs_max} />
          </Section>
        )}

        {(ibs.dc_miss_count > 0 || ibs.l2_miss_count > 0) && (
          <Section title="Cache (raw IBS bits)">
            <Row label="DcMiss (L1)" value={ibs.dc_miss_count} ratio={ibs.sample_count} />
            <Row label="L2Miss" value={ibs.l2_miss_count} ratio={ibs.sample_count} />
          </Section>
        )}

        {(ibs.dc_l1_tlb_miss_count > 0 || ibs.dc_l2_tlb_miss_count > 0) && (
          <Section title="TLB (raw IBS bits)">
            <Row label="DcL1TlbMiss" value={ibs.dc_l1_tlb_miss_count} ratio={ibs.sample_count} />
            <Row label="DcL2TlbMiss" value={ibs.dc_l2_tlb_miss_count} ratio={ibs.sample_count} />
          </Section>
        )}

        {(ibs.brn_ret_count > 0 || ibs.brn_fuse_count > 0) && (
          <Section title="Branches (raw IBS bits)">
            <Row label="BrnRet" value={ibs.brn_ret_count} ratio={ibs.sample_count} />
            {ibs.brn_ret_count > 0 && (
              <>
                <Row
                  label="OpBrnMisp"
                  value={ibs.brn_misp_count}
                  ratio={ibs.brn_ret_count}
                  pct={pct(ibs.brn_misp_count, ibs.brn_ret_count)}
                  hot={ibs.brn_misp_count > 0}
                />
                <Row
                  label="OpBrnTaken"
                  value={ibs.brn_taken_count}
                  ratio={ibs.brn_ret_count}
                  pct={pct(ibs.brn_taken_count, ibs.brn_ret_count)}
                />
                {ibs.brn_return_count > 0 && (
                  <Row
                    label="OpReturn"
                    value={ibs.brn_return_count}
                    ratio={ibs.brn_ret_count}
                    pct={pct(ibs.brn_return_count, ibs.brn_ret_count)}
                  />
                )}
              </>
            )}
            {ibs.brn_fuse_count > 0 && (
              <Row label="BrnFuse" value={ibs.brn_fuse_count} ratio={ibs.sample_count} />
            )}
          </Section>
        )}
      </div>
    </div>
  );
}

function tlbLabel(k: string): string {
  if (k === "L1_HIT") return "L1 DTLB hit";
  if (k === "L2_HIT") return "L1 miss, L2 hit";
  if (k === "MISS") return "L1+L2 miss (walk)";
  return k;
}

function pct(num: number, denom: number): number {
  return denom > 0 ? (100 * num) / denom : 0;
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
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
  ratio,
  hot,
  level,
}: {
  label: string;
  value: number | string;
  pct?: number;
  ratio?: number;
  hot?: boolean;
  level?: string;
}) {
  const cls = `row ${hot ? "hot" : ""} ${level ? `lvl-${level.replace("-", "")}` : ""}`;
  return (
    <div className={cls}>
      <span className="row-label">{label}</span>
      <span className="row-value">
        {typeof value === "number" ? value.toLocaleString() : value}
      </span>
      {pct !== undefined && <span className="row-pct">{pct.toFixed(1)}%</span>}
      {ratio !== undefined && (
        <span className="row-ratio">/ {ratio.toLocaleString()}</span>
      )}
    </div>
  );
}
