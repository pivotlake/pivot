// Slide-in sidebar with per-instruction IBS detail — mirrors what
// ibs_annotate's TUI shows when you press Enter on an instruction.

import { useEffect, useState } from "react";
import { apiInsnDetail, type InsnStats } from "../api";
import { Spinner } from "./Spinner";

interface Props {
  symbol: string | null;
  offset: number | null;
  disasm: string | null;
  addr: string | null;
  onClose: () => void;
}

const SECTION_TITLE: React.CSSProperties = {
  fontSize: 10,
  textTransform: "uppercase",
  letterSpacing: 0.5,
  color: "oklch(0.5 0.01 250)",
  fontWeight: 600,
  marginBottom: 6,
};

const HR: React.CSSProperties = {
  height: 1,
  background: "oklch(0.92 0.005 250)",
  margin: "12px 0",
  border: 0,
};

function pct(n: number, d: number): string {
  if (d <= 0) return "";
  return ((100 * n) / d).toFixed(1) + "%";
}

function avg(sum: number, count: number): number {
  return count > 0 ? sum / count : 0;
}

function CountBar({
  label,
  count,
  total,
  hue,
  bold = false,
}: {
  label: string;
  count: number;
  total: number;
  hue: number;
  bold?: boolean;
}) {
  if (count === 0 && !bold) return null;
  const p = total > 0 ? count / total : 0;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "60px 70px 60px 1fr",
        alignItems: "center",
        height: 18,
        gap: 8,
        fontSize: 11.5,
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        fontVariantNumeric: "tabular-nums",
        color: bold ? "oklch(0.2 0.01 250)" : "oklch(0.35 0.01 250)",
        fontWeight: bold ? 600 : 400,
      }}
    >
      <span>{label}</span>
      <span style={{ textAlign: "right" }}>{count.toLocaleString()}</span>
      <span style={{ textAlign: "right" }}>{pct(count, total)}</span>
      <div
        style={{
          height: 8,
          borderRadius: 2,
          background: "oklch(0.96 0.003 250)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.min(100, p * 100)}%`,
            height: "100%",
            background: `oklch(0.78 0.16 ${hue})`,
          }}
        />
      </div>
    </div>
  );
}

function StatRow({
  label,
  primary,
  secondary,
}: {
  label: string;
  primary: string;
  secondary?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        height: 18,
        fontSize: 11.5,
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        fontVariantNumeric: "tabular-nums",
        color: "oklch(0.3 0.01 250)",
      }}
    >
      <span style={{ color: "oklch(0.45 0.01 250)" }}>{label}</span>
      <span>
        <strong style={{ fontWeight: 600, color: "oklch(0.2 0.01 250)" }}>
          {primary}
        </strong>
        {secondary && (
          <span style={{ marginLeft: 8, color: "oklch(0.55 0.01 250)" }}>
            {secondary}
          </span>
        )}
      </span>
    </div>
  );
}

export function InstructionDetail({
  symbol,
  offset,
  disasm,
  addr,
  onClose,
}: Props) {
  const [data, setData] = useState<InsnStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (symbol == null || offset == null) {
      setData(null);
      return;
    }
    // Clear previous data so a failing/slow fetch doesn't leave the user
    // staring at stale numbers from the previous instruction.
    setData(null);
    setError(null);
    setLoading(true);
    let cancelled = false;
    apiInsnDetail(symbol, offset)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [symbol, offset]);

  if (symbol == null || offset == null) return null;

  // ── Derived stats ─────────────────────────────────────────────────────
  const totalSamples = data ? data.total_samples : 0;
  const opTotal = data
    ? Object.values(data.op_counts).reduce((s, v) => s + v, 0)
    : 0;
  const tlbTotal = data
    ? Object.values(data.tlb_counts).reduce((s, v) => s + v, 0)
    : 0;
  const tlbHit =
    (data?.tlb_counts.L1Hit ?? 0) + (data?.tlb_counts.L2Hit ?? 0);
  const tlbMiss = data?.tlb_counts.Miss ?? 0;
  const widthEntries = data
    ? Object.entries(data.ibs.mem_width_counts)
        .map(([k, v]) => [Number(k), v] as [number, number])
        .sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        bottom: 0,
        width: 380,
        background: "oklch(0.995 0.002 250)",
        borderLeft: "1px solid oklch(0.9 0.005 250)",
        boxShadow: "-12px 0 32px oklch(0.2 0.02 250 / 0.06)",
        display: "flex",
        flexDirection: "column",
        zIndex: 30,
        fontFamily:
          '"Inter", -apple-system, BlinkMacSystemFont, system-ui, sans-serif',
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid oklch(0.93 0.005 250)",
          display: "flex",
          alignItems: "flex-start",
          gap: 10,
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div
            style={{
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: 0.5,
              color: "oklch(0.5 0.01 250)",
              marginBottom: 6,
            }}
          >
            Instruction detail
          </div>
          <div
            style={{
              fontFamily: "ui-monospace, SFMono-Regular, monospace",
              fontSize: 11.5,
              color: "oklch(0.45 0.06 280)",
              wordBreak: "break-all",
              marginBottom: 4,
            }}
          >
            {symbol}+0x{offset.toString(16)}
          </div>
          {disasm && (
            <div
              style={{
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
                fontSize: 12.5,
                color: "oklch(0.2 0.01 250)",
                fontWeight: 500,
              }}
            >
              {addr && (
                <span style={{ color: "oklch(0.55 0.05 280)", marginRight: 8 }}>
                  {addr}
                </span>
              )}
              {disasm}
            </div>
          )}
        </div>
        <button
          onClick={onClose}
          style={{
            border: "1px solid oklch(0.9 0.005 250)",
            background: "white",
            cursor: "pointer",
            width: 24,
            height: 24,
            borderRadius: 4,
            color: "oklch(0.45 0.01 250)",
            fontSize: 14,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flex: "0 0 auto",
          }}
        >
          ×
        </button>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "14px 16px" }}>
        {loading && (
          <div style={{ minHeight: 120 }}>
            <Spinner size={24} label="Loading…" />
          </div>
        )}
        {error && (
          <div
            style={{
              color: "oklch(0.4 0.18 25)",
              fontSize: 12,
              padding: 8,
            }}
          >
            {error}
          </div>
        )}
        {data && data.total_samples === 0 && data.cycles === 0 && (
          <div
            style={{
              padding: 12,
              border: "1px dashed oklch(0.9 0.005 250)",
              borderRadius: 6,
              color: "oklch(0.5 0.01 250)",
              fontSize: 12,
              fontStyle: "italic",
              textAlign: "center",
              background: "oklch(0.985 0.003 250)",
            }}
          >
            No samples landed on this instruction.
            <br />
            <span style={{ fontSize: 11 }}>
              Pick a hot row (tinted yellow) to see the IBS detail.
            </span>
          </div>
        )}
        {data && (data.total_samples > 0 || data.cycles > 0) && (
          <>
            <div style={{ marginBottom: 14 }}>
              <span
                style={{
                  fontSize: 11,
                  color: "oklch(0.45 0.01 250)",
                  textTransform: "uppercase",
                  letterSpacing: 0.4,
                  marginRight: 8,
                }}
              >
                Total samples
              </span>
              <strong
                style={{
                  fontSize: 14,
                  fontFamily: "ui-monospace, monospace",
                  color: "oklch(0.2 0.01 250)",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {totalSamples.toLocaleString()}
              </strong>
              {data.cycles > 0 && (
                <span
                  style={{
                    marginLeft: 14,
                    fontSize: 11,
                    color: "oklch(0.55 0.01 250)",
                    fontFamily: "ui-monospace, monospace",
                  }}
                >
                  · {data.cycles.toLocaleString()} cycles
                </span>
              )}
            </div>

            {/* Cache levels */}
            <div style={SECTION_TITLE}>Cache level</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <CountBar
                label="L1"
                count={data.cache_counts.L1 ?? 0}
                total={totalSamples}
                hue={140}
              />
              <CountBar
                label="LFB"
                count={data.cache_counts.LFB ?? 0}
                total={totalSamples}
                hue={140}
              />
              <CountBar
                label="L2"
                count={data.cache_counts.L2 ?? 0}
                total={totalSamples}
                hue={70}
              />
              <CountBar
                label="L3"
                count={data.cache_counts.L3 ?? 0}
                total={totalSamples}
                hue={70}
              />
              <CountBar
                label="DRAM"
                count={data.cache_counts.DRAM ?? 0}
                total={totalSamples}
                hue={25}
              />
              <CountBar
                label="REM"
                count={data.cache_counts.REMOTE ?? 0}
                total={totalSamples}
                hue={25}
              />
              <CountBar
                label="N-M"
                count={data.cache_counts.NonMemory ?? 0}
                total={totalSamples}
                hue={250}
              />
            </div>

            <hr style={HR} />

            {/* TLB */}
            <div style={SECTION_TITLE}>TLB</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <CountBar
                label="L1 hit"
                count={data.tlb_counts.L1Hit ?? 0}
                total={tlbTotal}
                hue={140}
              />
              <CountBar
                label="L2 hit"
                count={data.tlb_counts.L2Hit ?? 0}
                total={tlbTotal}
                hue={70}
              />
              <CountBar
                label="Miss"
                count={data.tlb_counts.Miss ?? 0}
                total={tlbTotal}
                hue={25}
              />
            </div>
            {tlbHit + tlbMiss > 0 && (
              <div style={{ marginTop: 6 }}>
                <StatRow
                  label="TLB miss rate"
                  primary={pct(tlbMiss, tlbHit + tlbMiss)}
                  secondary={`(${tlbMiss}/${tlbHit + tlbMiss})`}
                />
              </div>
            )}

            <hr style={HR} />

            {/* Snoop */}
            {Object.values(data.snoop_counts).some((v) => v > 0) && (
              <>
                <div style={SECTION_TITLE}>Snoop (cache coherency)</div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                  }}
                >
                  {(["None", "Hit", "HitM", "Miss"] as const).map((s) => (
                    <CountBar
                      key={s}
                      label={s}
                      count={data.snoop_counts[s] ?? 0}
                      total={totalSamples}
                      hue={295}
                    />
                  ))}
                </div>
                <hr style={HR} />
              </>
            )}

            {/* Operations */}
            <div style={SECTION_TITLE}>Operations</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <CountBar
                label="LOAD"
                count={data.op_counts.Load ?? 0}
                total={opTotal}
                hue={220}
              />
              <CountBar
                label="STORE"
                count={data.op_counts.Store ?? 0}
                total={opTotal}
                hue={295}
              />
              <CountBar
                label="N/A"
                count={data.op_counts.NA ?? 0}
                total={opTotal}
                hue={250}
              />
            </div>
            {data.locked_count > 0 && (
              <div style={{ marginTop: 6 }}>
                <StatRow
                  label="Locked"
                  primary={data.locked_count.toLocaleString()}
                />
              </div>
            )}

            <hr style={HR} />

            {/* Latency */}
            <div style={SECTION_TITLE}>Latency (cycles)</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <StatRow
                label="Tag-to-retire"
                primary={avg(
                  data.ibs.tag_to_ret_sum,
                  data.ibs.tag_to_ret_count,
                ).toFixed(1)}
                secondary={`${data.ibs.tag_to_ret_count} samples`}
              />
              <StatRow
                label="Comp-to-retire"
                primary={avg(
                  data.ibs.comp_to_ret_sum,
                  data.ibs.comp_to_ret_count,
                ).toFixed(1)}
                secondary={`${data.ibs.comp_to_ret_count} samples`}
              />
              {data.ibs.dc_miss_lat_count > 0 && (
                <StatRow
                  label="DC miss latency"
                  primary={avg(
                    data.ibs.dc_miss_lat_sum,
                    data.ibs.dc_miss_lat_count,
                  ).toFixed(1)}
                  secondary={`${data.ibs.dc_miss_lat_count} samples`}
                />
              )}
              {data.ibs.tlb_refill_lat_count > 0 && (
                <StatRow
                  label="TLB refill latency"
                  primary={avg(
                    data.ibs.tlb_refill_lat_sum,
                    data.ibs.tlb_refill_lat_count,
                  ).toFixed(1)}
                  secondary={`${data.ibs.tlb_refill_lat_count} samples`}
                />
              )}
            </div>

            <hr style={HR} />

            {/* MAB / MLP */}
            <div style={SECTION_TITLE}>Memory-Level Parallelism</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <StatRow
                label="Avg MABs in flight"
                primary={avg(
                  data.ibs.mabs_sum,
                  data.ibs.mabs_count,
                ).toFixed(2)}
                secondary={
                  data.ibs.mabs_max > 0
                    ? `(max ${data.ibs.mabs_max} ×${data.ibs.mabs_max_count}, ${data.ibs.mabs_count} samples)`
                    : ""
                }
              />
              {data.ibs.dc_miss_no_mab_count > 0 && (
                <StatRow
                  label="MAB coalesced (hit existing)"
                  primary={data.ibs.dc_miss_no_mab_count.toLocaleString()}
                />
              )}
            </div>

            {widthEntries.length > 0 && (
              <>
                <hr style={HR} />
                <div style={SECTION_TITLE}>Access width</div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                  }}
                >
                  {widthEntries.map(([w, c]) => (
                    <StatRow
                      key={w}
                      label={`${w} bytes`}
                      primary={c.toLocaleString()}
                    />
                  ))}
                </div>
              </>
            )}

            {(data.ibs.dc_miss_count > 0 ||
              data.ibs.l2_miss_count > 0 ||
              data.ibs.misaligned_count > 0 ||
              data.ibs.sw_pf_count > 0 ||
              data.ibs.dc_l1_tlb_miss_count > 0 ||
              data.ibs.dc_l2_tlb_miss_count > 0) && (
              <>
                <hr style={HR} />
                <div style={SECTION_TITLE}>Cache (raw IBS bits)</div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                  }}
                >
                  {data.ibs.dc_miss_count > 0 && (
                    <StatRow
                      label="DcMiss (L1)"
                      primary={`${data.ibs.dc_miss_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                  {data.ibs.l2_miss_count > 0 && (
                    <StatRow
                      label="L2Miss"
                      primary={`${data.ibs.l2_miss_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                  {data.ibs.dc_l1_tlb_miss_count > 0 && (
                    <StatRow
                      label="DcL1TlbMiss"
                      primary={`${data.ibs.dc_l1_tlb_miss_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                  {data.ibs.dc_l2_tlb_miss_count > 0 && (
                    <StatRow
                      label="DcL2TlbMiss"
                      primary={`${data.ibs.dc_l2_tlb_miss_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                  {data.ibs.misaligned_count > 0 && (
                    <StatRow
                      label="Misaligned"
                      primary={`${data.ibs.misaligned_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                  {data.ibs.sw_pf_count > 0 && (
                    <StatRow
                      label="SW prefetch"
                      primary={`${data.ibs.sw_pf_count} / ${data.ibs.sample_count}`}
                    />
                  )}
                </div>
              </>
            )}

            {(data.ibs.brn_ret_count > 0 || data.ibs.brn_fuse_count > 0) && (
              <>
                <hr style={HR} />
                <div style={SECTION_TITLE}>Branch</div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                  }}
                >
                  <StatRow
                    label="BrnRet"
                    primary={data.ibs.brn_ret_count.toLocaleString()}
                  />
                  {data.ibs.brn_misp_count > 0 && (
                    <StatRow
                      label="Mispredicted"
                      primary={`${data.ibs.brn_misp_count} (${pct(data.ibs.brn_misp_count, data.ibs.brn_ret_count)})`}
                    />
                  )}
                  {data.ibs.brn_taken_count > 0 && (
                    <StatRow
                      label="Taken"
                      primary={`${data.ibs.brn_taken_count} (${pct(data.ibs.brn_taken_count, data.ibs.brn_ret_count)})`}
                    />
                  )}
                  {data.ibs.brn_return_count > 0 && (
                    <StatRow
                      label="Return"
                      primary={data.ibs.brn_return_count.toLocaleString()}
                    />
                  )}
                  {data.ibs.brn_fuse_count > 0 && (
                    <StatRow
                      label="Fused"
                      primary={data.ibs.brn_fuse_count.toLocaleString()}
                    />
                  )}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
