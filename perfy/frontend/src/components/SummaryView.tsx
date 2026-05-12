// Pipeline Top-Down summary — AMD Zen4 PipelineL2 metric set,
// rendered as a stacked-bar overview + a few key diagnostic cards.
//
// Inputs come from `/api/pipeline_summary` (see crates/perfy-server/
// src/pipeline.rs for the underlying math). Categories that perf
// failed to populate (multiplexing dropped them, the recording was
// taken without `-M PipelineL2`, etc.) get rendered as "—" so the
// user can tell "missing" from "actually zero".

import type { PipelineSummary } from "../api";

interface Props {
  data: PipelineSummary;
  /// Optional time-window label (e.g. `"3.4 s – 6.1 s · 2.7 s window"`).
  /// Rendered in the section header so the user knows the metrics
  /// reflect the current timeline zoom rather than the whole run.
  rangeLabel?: string;
}

/// Top-Down L1 categories with their canonical hue (matches
/// Intel/AMD/Linux perf docs visualisations: green = good, red/amber
/// = stall, blue = mispred). Order is the conventional one — the
/// stacked bar reads left → right.
const TOPDOWN: Array<{
  key: keyof PipelineSummary;
  label: string;
  hue: number;
  hint: string;
}> = [
  {
    key: "retiring_pct",
    label: "Retiring",
    hue: 145,
    hint: "Dispatch slots that produced a retired op. Higher is better.",
  },
  {
    key: "bad_speculation_pct",
    label: "Bad speculation",
    hue: 15,
    hint:
      "Dispatched ops that never retired — squashed by branch mispredicts " +
      "or pipeline flushes.",
  },
  {
    key: "frontend_bound_pct",
    label: "Frontend bound",
    hue: 250,
    hint: "Slots starved of ops by the frontend (icache, branch redirect, fetch).",
  },
  {
    key: "backend_bound_pct",
    label: "Backend bound",
    hue: 35,
    hint:
      "Slots stalled because the backend couldn't accept ops — load/store " +
      "queue full, retire stall, etc.",
  },
];

function fmtPct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return `${v.toFixed(1)}%`;
}

function fmtNumber(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1e12) return `${(v / 1e12).toFixed(2)} T`;
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)} G`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)} M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(2)} k`;
  return v.toFixed(0);
}

function StackedBar({ data }: { data: PipelineSummary }) {
  // Sum of the 4 L1 categories. When < 100, perf multiplexed events
  // and some categories underflowed — show the gap as a neutral
  // "unaccounted" segment so the bar still totals 100% and the user
  // can see how much we couldn't measure.
  const segments = TOPDOWN.map((t) => ({
    label: t.label,
    pct: Math.max(0, data[t.key] as number),
    hue: t.hue,
  }));
  const accounted = segments.reduce((s, x) => s + x.pct, 0);
  const gap = Math.max(0, 100 - accounted);
  return (
    <div
      style={{
        display: "flex",
        height: 28,
        borderRadius: 5,
        overflow: "hidden",
        border: "1px solid oklch(0.9 0.005 250)",
        background: "white",
      }}
    >
      {segments.map((s) => {
        if (s.pct < 0.5) return null;
        return (
          <div
            key={s.label}
            title={`${s.label} · ${s.pct.toFixed(2)}%`}
            style={{
              width: `${s.pct}%`,
              background: `oklch(0.78 0.16 ${s.hue})`,
              borderRight: "1px solid oklch(0.65 0.16 currentColor / 0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 10.5,
              color: "oklch(0.2 0.05 250)",
              fontWeight: 500,
              overflow: "hidden",
              whiteSpace: "nowrap",
            }}
          >
            {s.pct >= 8 ? `${s.label} ${s.pct.toFixed(1)}%` : ""}
          </div>
        );
      })}
      {gap > 0.5 && (
        <div
          title={`Unaccounted (${gap.toFixed(1)}%) — likely event multiplexing.`}
          style={{
            width: `${gap}%`,
            background:
              "repeating-linear-gradient(45deg, oklch(0.93 0.005 250), oklch(0.93 0.005 250) 4px, oklch(0.97 0.005 250) 4px, oklch(0.97 0.005 250) 8px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 10,
            color: "oklch(0.5 0.01 250)",
          }}
        >
          {gap >= 8 ? `unaccounted ${gap.toFixed(1)}%` : ""}
        </div>
      )}
    </div>
  );
}

/// Internal sub-segment for a category card's mini split-bar.
interface SubSeg {
  label: string;
  pct: number;
  /// `true` for the brighter (primary) sub-category, `false` for the
  /// muted complement. The two are visually distinct so the user can
  /// see which is "good" vs "stalled" inside each parent.
  primary: boolean;
}

function SubBar({ hue, segs }: { hue: number; segs: SubSeg[] }) {
  const total = segs.reduce((s, x) => s + Math.max(0, x.pct), 0);
  if (total <= 0) return null;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        marginTop: 6,
      }}
    >
      <div
        style={{
          display: "flex",
          height: 6,
          borderRadius: 3,
          overflow: "hidden",
          background: "oklch(0.95 0.005 250)",
        }}
      >
        {segs.map((s) => {
          const w = Math.max(0, s.pct);
          if (w <= 0) return null;
          return (
            <div
              key={s.label}
              title={`${s.label} · ${s.pct.toFixed(2)}%`}
              style={{
                width: `${w}%`,
                background: s.primary
                  ? `oklch(0.65 0.18 ${hue})`
                  : `oklch(0.82 0.1 ${hue})`,
              }}
            />
          );
        })}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          fontSize: 10.5,
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
        }}
      >
        {segs.map((s) => (
          <div
            key={s.label}
            style={{
              display: "flex",
              justifyContent: "space-between",
              color: "oklch(0.45 0.01 250)",
            }}
          >
            <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: 1,
                  background: s.primary
                    ? `oklch(0.65 0.18 ${hue})`
                    : `oklch(0.82 0.1 ${hue})`,
                }}
              />
              {s.label}
            </span>
            <span style={{ color: "oklch(0.25 0.01 250)" }}>
              {s.pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CategoryCard({
  label,
  pct,
  hue,
  hint,
  subs,
}: {
  label: string;
  pct: number;
  hue: number;
  hint: string;
  /// Optional sub-breakdown segments (renders as a mini split-bar +
  /// a two-line legend underneath the headline %).
  subs?: SubSeg[];
}) {
  const ok = Number.isFinite(pct);
  return (
    <div
      style={{
        flex: 1,
        minWidth: 180,
        background: "white",
        border: "1px solid oklch(0.92 0.005 250)",
        borderRadius: 6,
        padding: "10px 12px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
      title={hint}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: 11,
          color: "oklch(0.5 0.01 250)",
          textTransform: "uppercase",
          letterSpacing: 0.4,
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 2,
            background: `oklch(0.72 0.16 ${hue})`,
          }}
        />
        {label}
      </div>
      <div
        style={{
          fontSize: 22,
          fontWeight: 600,
          color: ok ? "oklch(0.2 0.04 250)" : "oklch(0.6 0.01 250)",
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {ok ? `${pct.toFixed(1)}%` : "—"}
      </div>
      {subs && subs.length > 0 && <SubBar hue={hue} segs={subs} />}
    </div>
  );
}

function StatRow({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div
      title={hint}
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "6px 0",
        borderBottom: "1px solid oklch(0.96 0.005 250)",
        fontSize: 12,
      }}
    >
      <span style={{ color: "oklch(0.45 0.01 250)" }}>{label}</span>
      <span
        style={{
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontVariantNumeric: "tabular-nums",
          color: "oklch(0.2 0.01 250)",
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function SummaryView({ data, rangeLabel }: Props) {
  // Hard-unavailable: no data at all (no perf.stat.data, can't even
  // read cycles/ops_ret). Distinct from "partially available" where
  // we have *some* metrics but lost others to a missing group.
  const totallyEmpty =
    !data.available &&
    !Number.isFinite(data.retiring_pct) &&
    !Number.isFinite(data.frontend_bound_pct) &&
    !Number.isFinite(data.backend_bound_pct);
  if (totallyEmpty) {
    return (
      <div
        style={{
          padding: 32,
          color: "oklch(0.5 0.01 250)",
          fontSize: 13,
          lineHeight: 1.6,
          maxWidth: 720,
        }}
      >
        <div
          style={{
            fontSize: 14,
            fontWeight: 600,
            marginBottom: 8,
            color: "oklch(0.3 0.01 250)",
          }}
        >
          Pipeline summary unavailable
        </div>
        {data.error ?? (
          <>
            No <code>perf.stat.data</code> sibling found. Re-record with{" "}
            <code>perf stat record -M PipelineL2 -- &lt;workload&gt;</code>.
          </>
        )}
      </div>
    );
  }
  return (
    <div
      style={{
        padding: 24,
        maxWidth: 1100,
        display: "flex",
        flexDirection: "column",
        gap: 20,
        fontFamily:
          '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
      }}
    >
      {/* Top-of-page error banner — surfaced whenever the backend
          couldn't compute every metric. Lists the affected metrics
          and the recording command to fix it. */}
      {data.error && (
        <div
          role="alert"
          style={{
            background: "oklch(0.97 0.05 25)",
            border: "1px solid oklch(0.85 0.13 25)",
            borderLeft: "3px solid oklch(0.6 0.18 25)",
            borderRadius: 5,
            padding: "10px 14px",
            fontSize: 12,
            lineHeight: 1.5,
            color: "oklch(0.3 0.1 25)",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Some metrics are unavailable for this recording
          </div>
          {data.error}
        </div>
      )}

      {/* Section: Top-Down L1 */}
      <section>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            margin: "0 0 4px",
          }}
        >
          <h3
            style={{
              margin: 0,
              fontSize: 13,
              fontWeight: 600,
              color: "oklch(0.3 0.01 250)",
              textTransform: "uppercase",
              letterSpacing: 0.6,
            }}
          >
            Pipeline slot breakdown
          </h3>
          {rangeLabel && (
            <div
              style={{
                fontSize: 11,
                color: "oklch(0.5 0.01 250)",
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
              }}
            >
              {rangeLabel}
            </div>
          )}
        </div>
        <div
          style={{
            fontSize: 11,
            color: "oklch(0.55 0.01 250)",
            marginBottom: 10,
          }}
        >
          Where every dispatch slot went over the run. Each slot is
          either a retired op, a stall, or speculation that got
          squashed.
        </div>
        <StackedBar data={data} />
        <div
          style={{
            display: "flex",
            gap: 10,
            marginTop: 12,
            flexWrap: "wrap",
          }}
        >
          {TOPDOWN.map((t) => {
            // Each top-level category that has a Top-Down L2 split
            // gets a mini-bar with its two child segments. The
            // "primary" (=bright) child is the one users typically
            // care about diagnostically: memory-bound for backend,
            // fastpath for retiring, latency for frontend.
            let subs: SubSeg[] | undefined;
            if (t.key === "frontend_bound_pct") {
              subs = [
                {
                  label: "Latency (fully-stalled cycles)",
                  pct: data.frontend_latency_pct,
                  primary: true,
                },
                {
                  label: "Bandwidth (partial-stall)",
                  pct: data.frontend_bandwidth_pct,
                  primary: false,
                },
              ];
            } else if (t.key === "backend_bound_pct") {
              subs = [
                {
                  label: "Memory (load not complete)",
                  pct: data.backend_memory_pct,
                  primary: true,
                },
                {
                  label: "Core (execution units)",
                  pct: data.backend_core_pct,
                  primary: false,
                },
              ];
            } else if (t.key === "retiring_pct") {
              subs = [
                {
                  label: "Fastpath",
                  pct: data.retiring_fastpath_pct,
                  primary: true,
                },
                {
                  label: "Microcode",
                  pct: data.retiring_microcode_pct,
                  primary: false,
                },
              ];
            }
            return (
              <CategoryCard
                key={t.label}
                label={t.label}
                pct={data[t.key] as number}
                hue={t.hue}
                hint={t.hint}
                subs={subs}
              />
            );
          })}
        </div>
      </section>

      {/* Section: Single-number diagnostics */}
      <section>
        <h3
          style={{
            margin: "0 0 10px",
            fontSize: 13,
            fontWeight: 600,
            color: "oklch(0.3 0.01 250)",
            textTransform: "uppercase",
            letterSpacing: 0.6,
          }}
        >
          Key metrics
        </h3>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 18,
          }}
        >
          <div
            style={{
              background: "white",
              border: "1px solid oklch(0.92 0.005 250)",
              borderRadius: 6,
              padding: "8px 14px",
            }}
          >
            <StatRow
              label="IPC (ops / cycle)"
              value={data.ipc.toFixed(3)}
              hint="Retired ops per cycle. Zen4's theoretical max is 6."
            />
            <StatRow
              label="Branch mispredict rate"
              value={fmtPct(data.branch_misp_pct)}
              hint="Retired branch mispredicts as a percentage of retired ops."
            />
            <StatRow
              label="Microcoded ops"
              value={fmtPct(data.microcode_pct)}
              hint={
                "Microcoded ops as a percentage of retired ops. " +
                ">2% means expensive instructions (div, gather, complex string)."
              }
            />
            <StatRow
              label="Pipeline resync rate"
              value={fmtPct(data.resync_pct)}
              hint="Pipeline resyncs / NC redirects per retired op. Each resync flushes the pipeline."
            />
          </div>
          <div
            style={{
              background: "white",
              border: "1px solid oklch(0.92 0.005 250)",
              borderRadius: 6,
              padding: "8px 14px",
            }}
          >
            <StatRow
              label="Backend memory share"
              value={fmtPct(data.backend_memory_share * 100)}
              hint={
                "Of all backend stalls, the fraction caused by 'load not " +
                "complete'. Higher → memory-bound; lower → core-bound."
              }
            />
            <StatRow
              label="Total cycles"
              value={fmtNumber(data.total_cycles)}
            />
            <StatRow
              label="Retired ops"
              value={fmtNumber(data.total_ops_retired)}
            />
            <StatRow
              label="Dispatched ops"
              value={fmtNumber(data.total_ops_dispatched)}
              hint={
                "Ops that left the dispatch unit. Includes ones that " +
                "got squashed before retire (Bad Speculation)."
              }
            />
          </div>
        </div>
      </section>
    </div>
  );
}
