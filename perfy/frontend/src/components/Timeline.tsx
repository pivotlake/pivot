// Timeline — yellow line-graph tracks with dark-blue baseline tick marks.
// Click the disclosure caret to expand per-CPU rows. Click the row name to
// select that category for the flame graph. Drag in the canvas to zoom into
// a time range.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Category, StatResponse } from "../api";
import { CATEGORY_META, CATEGORY_ORDER } from "../api";

export interface TimelineCategory {
  id: Category;
  label: string;
  short: string;
  hue: number;
  /// `consolidated[i]` ∈ [0,1]: aggregated density for bin `i`.
  consolidated: Float32Array;
  /// `perCpu[cpu][i]` ∈ [0,1]: per-cpu density, missing cpus are absent.
  perCpu: Map<number, Float32Array>;
  /// Per-CPU peak (raw count, before normalization) — used in tooltips.
  rawPeak: number;
}

interface Props {
  cpus: number[];
  cats: TimelineCategory[];
  timeBins: number;
  durationMs: number;
  selectedCat: Category | null;
  setSelectedCat: (c: Category | null) => void;
  expandedCats: Set<Category>;
  toggleExpand: (c: Category) => void;
  /// `[loBin, hiBin]` inclusive — current zoom window in bin space.
  range: [number, number];
  /// Push a new zoom level onto the zoom-history stack. Called when
  /// the user marquees a sub-range; the toolbar renders the resulting
  /// stack as breadcrumb pills so they can step back one level at a
  /// time instead of jumping straight to the full view.
  pushRange: (r: [number, number]) => void;
  /// Pop the stack all the way back to the full recording. Wired to
  /// the double-click reset gesture.
  resetZoom: () => void;
  /// Optional perf-stat-derived series. When `available` is true we
  /// render two extra rows under the per-category tracks: memory
  /// throughput (B/s) and L3 read-miss latency (cycles). When the
  /// recording wasn't taken with stat.csv, this is null and the
  /// memory rows are simply absent.
  stat?: StatResponse | null;
}

const LABEL_WIDTH = 160;
const TRACK_HEIGHT = 44;
/// Per-CPU sub-track height. Kept slightly shorter than the
/// consolidated row above it so the visual hierarchy is obvious, but
/// tall enough that you can read individual CPU activity without
/// expanding the pane.
const SUB_TRACK_HEIGHT = 40;
const RULER_HEIGHT = 28;

function drawLineGraph(
  cvs: HTMLCanvasElement,
  values: Float32Array,
  range: [number, number],
) {
  const dpr = window.devicePixelRatio || 1;
  const w = cvs.clientWidth;
  const h = cvs.clientHeight;
  if (w === 0 || h === 0) return;
  cvs.width = w * dpr;
  cvs.height = h * dpr;
  const ctx = cvs.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const start = range[0];
  const end = range[1];
  const span = Math.max(1, end - start);
  const padTop = 1;
  const baselineY = h - 3;
  const usable = baselineY - padTop;

  // Sample points 1:1 with bins so we render real data — no synthetic jitter.
  const binsInRange = end - start + 1;
  const pts: Array<[number, number]> = new Array(binsInRange);
  for (let i = 0; i < binsInRange; i++) {
    const idx = start + i;
    const v = Math.max(0, Math.min(1, values[idx] || 0));
    const f = binsInRange === 1 ? 0 : i / (binsInRange - 1);
    pts[i] = [f * w, padTop + (1 - v) * usable];
  }

  ctx.fillStyle = "oklch(0.86 0.19 95)";
  ctx.beginPath();
  ctx.moveTo(0, baselineY);
  for (const [x, y] of pts) ctx.lineTo(x, y);
  ctx.lineTo(w, baselineY);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = "oklch(0.78 0.19 92)";
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "oklch(0.45 0.2 255)";
  for (let b = start; b <= end; b++) {
    const v = values[b] || 0;
    if (v < 0.03) continue;
    const f = (b - start) / span;
    const x = f * w;
    const th = Math.min(3, 1 + v * 2);
    ctx.fillRect(x - 0.5, baselineY, 1, th);
  }
}

function LineRow({
  values,
  range,
}: {
  values: Float32Array;
  range: [number, number];
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const draw = useCallback(() => {
    if (ref.current) drawLineGraph(ref.current, values, range);
  }, [values, range]);
  useEffect(() => {
    draw();
  }, [draw]);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [draw]);
  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: "100%", display: "block" }}
    />
  );
}

type Row =
  | { kind: "section"; title: string }
  | {
      kind: "cat";
      cat: TimelineCategory;
      values: Float32Array;
      height: number;
    }
  | {
      kind: "cpu";
      cat: TimelineCategory;
      cpu: number;
      values: Float32Array;
      height: number;
    }
  | {
      kind: "stat";
      /// Stable key for React + the row click logic. e.g. "throughput".
      id: string;
      label: string;
      /// Subtitle shown under the label (e.g. "12.4 GB/s peak").
      sub: string;
      values: Float32Array;
      height: number;
      hue: number;
    };

/// Format bytes/second using decimal SI units (KB = 1000 B, MB = 10^6,
/// GB = 10^9). Memory-bandwidth specs (DDR5-4800 = 38.4 GB/s, etc.)
/// are conventionally decimal, so a peak of `108.73e9 B/s` reads as
/// "108.7 GB/s" rather than a binary-units "101 GiB/s".
function formatBytesPerSec(bps: number): string {
  if (bps <= 0) return "0 B/s";
  const units = ["B/s", "KB/s", "MB/s", "GB/s", "TB/s"];
  let v = bps;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

export function Timeline({
  cpus,
  cats,
  timeBins,
  durationMs,
  selectedCat,
  setSelectedCat,
  expandedCats,
  toggleExpand,
  range,
  pushRange,
  resetZoom,
  stat,
}: Props) {
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    bin: number;
    row: number;
  } | null>(null);
  const [drag, setDrag] = useState<{
    startBin: number;
    currentBin: number;
  } | null>(null);
  const hostRef = useRef<HTMLDivElement | null>(null);

  const rows: Row[] = useMemo(() => {
    const out: Row[] = [];
    const visible = CATEGORY_ORDER.filter((id) => cats.find((c) => c.id === id));
    for (const id of visible) {
      const cat = cats.find((c) => c.id === id)!;
      out.push({ kind: "cat", cat, values: cat.consolidated, height: TRACK_HEIGHT });
      if (expandedCats.has(id)) {
        for (const cpu of cpus) {
          const series = cat.perCpu.get(cpu);
          if (!series) continue;
          out.push({
            kind: "cpu",
            cat,
            cpu,
            values: series,
            height: SUB_TRACK_HEIGHT,
          });
        }
      }
    }
    // perf-stat-derived rows go at the bottom (visually grouped with
    // the sampled tracks but conceptually distinct: these come from a
    // separate `perf stat -o stat.csv` run, share the same x-axis,
    // and have their own units rather than a shared "samples" denominator).
    if (stat && stat.available && stat.throughput_bps.length > 0) {
      const tPeak = Math.max(1, stat.throughput_peak);
      // For latency we plot the *delta* from the configured baseline.
      // Normalize against the positive peak so the axis is "0 = base,
      // top = max excursion". Sub-baseline values get clamped to 0 in
      // the canvas (they still show the signed value in the tooltip).
      const lPeak = Math.max(1, stat.latency_peak);
      const tNorm = new Float32Array(stat.throughput_bps.length);
      for (let i = 0; i < tNorm.length; i++) {
        tNorm[i] = stat.throughput_bps[i] / tPeak;
      }
      const lNorm = new Float32Array(stat.latency_clocks.length);
      for (let i = 0; i < lNorm.length; i++) {
        lNorm[i] = stat.latency_clocks[i] / lPeak;
      }
      out.push({
        kind: "stat",
        id: "throughput",
        label: "Memory throughput",
        sub: `peak ${formatBytesPerSec(stat.throughput_peak)}`,
        values: tNorm,
        height: TRACK_HEIGHT,
        hue: 140,
      });
      // Label always shows the **absolute** peak (delta + base) so it
      // reads the same regardless of the configured baseline; the
      // hover tooltip is where the user reads the delta.
      const latPeakAbs = stat.latency_peak + (stat.latency_base ?? 0);
      const latSub = `peak ${latPeakAbs.toFixed(0)} clk`;
      out.push({
        kind: "stat",
        id: "latency",
        label: "Memory latency",
        sub: latSub,
        values: lNorm,
        height: TRACK_HEIGHT,
        hue: 25,
      });
    }
    return out;
  }, [cats, cpus, expandedCats, stat]);

  const span = range[1] - range[0];

  const pxToBin = useCallback(
    (px: number, rect: DOMRect) => {
      const f =
        (px - rect.left - LABEL_WIDTH) / Math.max(1, rect.width - LABEL_WIDTH);
      return Math.round(range[0] + f * span);
    },
    [range, span],
  );

  function onMouseDown(e: React.MouseEvent) {
    if (!hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    if (e.clientX - rect.left < LABEL_WIDTH) return;
    const startBin = pxToBin(e.clientX, rect);
    setDrag({ startBin, currentBin: startBin });
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!hostRef.current) return;
    const rect = hostRef.current.getBoundingClientRect();
    if (drag) {
      setDrag({ ...drag, currentBin: pxToBin(e.clientX, rect) });
      return;
    }
    if (e.clientX - rect.left < LABEL_WIDTH) {
      setHover(null);
      return;
    }
    const bin = pxToBin(e.clientX, rect);
    if (bin < range[0] || bin > range[1]) {
      setHover(null);
      return;
    }
    let y = e.clientY - rect.top - RULER_HEIGHT;
    let idx = -1;
    for (let i = 0; i < rows.length; i++) {
      const rh = rows[i].kind === "section" ? 24 : (rows[i] as any).height;
      if (y < rh) {
        idx = i;
        break;
      }
      y -= rh;
    }
    if (idx < 0 || rows[idx].kind === "section") {
      setHover(null);
      return;
    }
    // Store viewport coordinates (clientX/clientY directly). The
    // tooltip is rendered as `position: fixed` so it can escape the
    // timeline's `overflow: auto` clipping when hovering near the
    // bottom of the pane.
    setHover({ x: e.clientX, y: e.clientY, bin, row: idx });
  }

  function onMouseUp() {
    if (drag) {
      const a = Math.min(drag.startBin, drag.currentBin);
      const b = Math.max(drag.startBin, drag.currentBin);
      if (b - a > 2) {
        pushRange([Math.max(0, a), Math.min(timeBins - 1, b)]);
      }
      setDrag(null);
    }
  }
  function onDoubleClick() {
    resetZoom();
  }

  const wrapWidth = hostRef.current ? hostRef.current.clientWidth : 1200;
  const trackPxWidth = Math.max(50, wrapWidth - LABEL_WIDTH);

  let dragOverlay: React.ReactNode = null;
  if (drag) {
    const lo = Math.min(drag.startBin, drag.currentBin);
    const hi = Math.max(drag.startBin, drag.currentBin);
    const x1 = LABEL_WIDTH + ((lo - range[0]) / span) * trackPxWidth;
    const x2 = LABEL_WIDTH + ((hi - range[0]) / span) * trackPxWidth;
    dragOverlay = (
      <div
        style={{
          position: "absolute",
          top: RULER_HEIGHT,
          bottom: 0,
          left: x1,
          width: x2 - x1,
          background: "oklch(0.65 0.16 250 / 0.14)",
          borderLeft: "1.5px solid oklch(0.55 0.18 250 / 0.7)",
          borderRight: "1.5px solid oklch(0.55 0.18 250 / 0.7)",
          pointerEvents: "none",
        }}
      />
    );
  }

  let tip: React.ReactNode = null;
  if (hover && !drag && rows[hover.row]) {
    const row = rows[hover.row] as Exclude<Row, { kind: "section" }>;
    const v = row.values[hover.bin] || 0;
    const isCpu = row.kind === "cpu";
    const isStatRow = row.kind === "stat";
    const headerLabel = isStatRow
      ? row.label
      : row.cat.label +
        (isCpu ? ` · cpu${(row as any).cpu.toString().padStart(2, "0")}` : "");
    // For stat rows we read raw values directly from the response
    // (bucket index lines up 1:1 with the rendered series). For
    // latency: when a baseline is configured we show **both** the
    // delta (= what the graph plots) and the absolute (= delta + base)
    // so the user can read either at a glance.
    const row2 = (label: string, value: string) => (
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span style={{ color: "oklch(0.6 0.01 250)" }}>{label}</span>
        <span>{value}</span>
      </div>
    );
    let valueRows: React.ReactNode;
    if (isStatRow && stat) {
      if (row.id === "throughput") {
        const raw = stat.throughput_bps[hover.bin] ?? 0;
        valueRows = row2("value", formatBytesPerSec(raw));
      } else {
        const delta = stat.latency_clocks[hover.bin] ?? 0;
        const absolute = delta + (stat.latency_base ?? 0);
        const sign = delta >= 0 ? "+" : "";
        if (stat.latency_base > 0) {
          valueRows = (
            <>
              {row2("delta", `${sign}${delta.toFixed(0)} clk`)}
              {row2("absolute", `${absolute.toFixed(0)} clk`)}
            </>
          );
        } else {
          valueRows = row2("value", `${absolute.toFixed(0)} clk`);
        }
      }
    } else {
      valueRows = row2("intensity", `${(v * 100).toFixed(1)}%`);
    }
    // `position: fixed` so the tooltip is anchored to the viewport,
    // not the timeline pane — avoids being clipped by the pane's
    // `overflow: auto` when hovering the bottom-most track. Offset
    // away from the cursor; flip horizontally if the tooltip would
    // run off the right edge of the window.
    const TIP_W = 220;
    const TIP_OFFSET = 14;
    const flipLeft = hover.x + TIP_OFFSET + TIP_W > window.innerWidth;
    const tipLeft = flipLeft ? hover.x - TIP_OFFSET - TIP_W : hover.x + TIP_OFFSET;
    tip = (
      <div
        style={{
          position: "fixed",
          left: Math.max(8, tipLeft),
          top: hover.y + TIP_OFFSET,
          background: "oklch(0.18 0.01 250)",
          color: "oklch(0.97 0.005 250)",
          padding: "8px 10px",
          fontSize: 11,
          fontFamily: "ui-monospace, monospace",
          borderRadius: 6,
          pointerEvents: "none",
          zIndex: 1000,
          minWidth: 180,
          boxShadow: "0 6px 24px oklch(0.2 0.02 250 / 0.18)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: 4,
          }}
        >
          <span style={{ color: "oklch(0.7 0.02 250)" }}>{headerLabel}</span>
          <span style={{ color: "oklch(0.85 0.16 95)" }}>●</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "oklch(0.6 0.01 250)" }}>t</span>
          <span>{((hover.bin / timeBins) * (durationMs / 1000)).toFixed(2)} s</span>
        </div>
        {valueRows}
      </div>
    );
  }

  // Ruler ticks
  const tickStep = Math.max(10, Math.round(span / 8));
  const ticks: React.ReactNode[] = [];
  for (
    let b = Math.ceil(range[0] / tickStep) * tickStep;
    b <= range[1];
    b += tickStep
  ) {
    const x = LABEL_WIDTH + ((b - range[0]) / span) * trackPxWidth;
    ticks.push(
      <div
        key={b}
        style={{
          position: "absolute",
          left: x,
          top: 0,
          bottom: 0,
          borderLeft: "1px solid oklch(0.94 0.005 250)",
        }}
      >
        <span
          style={{
            position: "absolute",
            top: 6,
            left: 5,
            fontSize: 10,
            color: "oklch(0.5 0.01 250)",
            fontFamily: "ui-monospace, monospace",
          }}
        >
          {((b / timeBins) * (durationMs / 1000)).toFixed(1)}s
        </span>
      </div>,
    );
  }

  return (
    <div
      ref={hostRef}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => setHover(null)}
      onDoubleClick={onDoubleClick}
      style={{
        position: "relative",
        userSelect: "none",
        cursor: drag ? "ew-resize" : "crosshair",
        overflow: "hidden",
        background: "white",
      }}
    >
      <div
        style={{
          position: "relative",
          height: RULER_HEIGHT,
          borderBottom: "1px solid oklch(0.92 0.005 250)",
          background: "oklch(0.99 0.003 250)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            width: LABEL_WIDTH,
            bottom: 0,
            borderRight: "1px solid oklch(0.92 0.005 250)",
            display: "flex",
            alignItems: "center",
            paddingLeft: 14,
            fontSize: 10,
            color: "oklch(0.5 0.01 250)",
            textTransform: "uppercase",
            letterSpacing: 0.5,
          }}
        >
          Tracks
        </div>
        {ticks}
      </div>

      {rows.map((row, i) => {
        if (row.kind === "section") {
          return (
            <div
              key={`s-${row.title}`}
              style={{
                height: 24,
                padding: "0 14px",
                display: "flex",
                alignItems: "center",
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 0.6,
                color: "oklch(0.5 0.01 250)",
                background: "oklch(0.97 0.005 250)",
                borderTop: "1px solid oklch(0.93 0.005 250)",
                borderBottom: "1px solid oklch(0.93 0.005 250)",
                fontWeight: 500,
              }}
            >
              {row.title}
            </div>
          );
        }
        const isCat = row.kind === "cat";
        const isStat = row.kind === "stat";
        const isExpanded = isCat && expandedCats.has(row.cat.id);
        const isSelected = isCat && selectedCat === row.cat.id;
        const rowKey = isCat
          ? row.cat.id
          : isStat
            ? `stat-${row.id}`
            : `${row.cat.id}-${(row as any).cpu}`;
        return (
          <div
            key={rowKey}
            style={{
              display: "flex",
              height: row.height,
              position: "relative",
              borderBottom:
                isCat || isStat ? "1px solid oklch(0.95 0.005 250)" : "none",
              background: isSelected
                ? "oklch(0.97 0.04 250 / 0.5)"
                : "transparent",
            }}
          >
            <div
              style={{
                width: LABEL_WIDTH,
                flex: "0 0 auto",
                borderRight: "1px solid oklch(0.92 0.005 250)",
                display: "flex",
                alignItems: "center",
                paddingLeft: isCat || isStat ? 6 : 28,
                fontSize: isCat || isStat ? 12 : 10,
                fontFamily:
                  isCat || isStat ? "inherit" : "ui-monospace, monospace",
                background: isSelected
                  ? "oklch(0.96 0.05 250 / 0.6)"
                  : "oklch(0.99 0.003 250)",
              }}
            >
              {isCat && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleExpand(row.cat.id);
                  }}
                  title={isExpanded ? "Collapse per-CPU" : "Expand per-CPU"}
                  style={{
                    width: 18,
                    height: 18,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    color: "oklch(0.45 0.01 250)",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    padding: 0,
                    marginRight: 2,
                  }}
                >
                  <span
                    style={{
                      display: "inline-block",
                      width: 0,
                      height: 0,
                      borderLeft: "4px solid transparent",
                      borderRight: "4px solid transparent",
                      borderTop: "5px solid currentColor",
                      transform: isExpanded
                        ? "rotate(0deg)"
                        : "rotate(-90deg)",
                      transition: "transform 0.15s",
                    }}
                  />
                </button>
              )}
              {isCat ? (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedCat(row.cat.id);
                  }}
                  style={{
                    flex: 1,
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    border: "none",
                    background: "transparent",
                    cursor: "pointer",
                    color: isSelected
                      ? `oklch(0.3 0.16 ${row.cat.hue})`
                      : "oklch(0.25 0.01 250)",
                    fontFamily: "inherit",
                    fontSize: 12,
                    fontWeight: isSelected ? 600 : 500,
                    padding: "4px 6px",
                    borderRadius: 4,
                    textAlign: "left",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 2,
                      background: `oklch(0.78 0.16 ${row.cat.hue})`,
                    }}
                  />
                  <span>{row.cat.label}</span>
                </button>
              ) : isStat ? (
                <div
                  style={{
                    flex: 1,
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "4px 6px",
                    color: "oklch(0.25 0.01 250)",
                  }}
                >
                  <span
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: 2,
                      background: `oklch(0.72 0.16 ${row.hue})`,
                    }}
                  />
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      lineHeight: 1.2,
                      minWidth: 0,
                    }}
                  >
                    <span style={{ fontSize: 12, fontWeight: 500 }}>
                      {row.label}
                    </span>
                    <span
                      style={{
                        fontSize: 10,
                        color: "oklch(0.55 0.01 250)",
                        fontFamily: "ui-monospace, monospace",
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {row.sub}
                    </span>
                  </div>
                </div>
              ) : (
                <span style={{ color: "oklch(0.5 0.01 250)" }}>
                  cpu
                  {(row as any).cpu.toString().padStart(2, "0")}
                </span>
              )}
            </div>
            <div
              style={{
                flex: 1,
                position: "relative",
                borderBottom: "1px solid oklch(0.96 0.005 250)",
              }}
            >
              <LineRow values={row.values} range={range} />
              {hover && hover.row === i && !drag && (
                <div
                  style={{
                    position: "absolute",
                    top: 0,
                    bottom: 0,
                    left:
                      ((hover.bin - range[0]) / span) * 100 + "%",
                    width: 1,
                    background: "oklch(0.25 0.01 250 / 0.5)",
                    pointerEvents: "none",
                  }}
                />
              )}
            </div>
          </div>
        );
      })}

      {dragOverlay}
      {tip}
    </div>
  );
}

// Helper for App: the canonical TimelineCategory list given API responses.
export function buildCategories(
  consolidated: { tracks: Array<{ category: Category; counts: number[]; peak: number }> },
  perCpu: { tracks: Array<{ category: Category; cpu: number | null; counts: number[]; peak: number }> },
): TimelineCategory[] {
  const out: TimelineCategory[] = [];
  for (const id of CATEGORY_ORDER) {
    const ct = consolidated.tracks.find((t) => t.category === id);
    if (!ct) continue;
    const peak = Math.max(1, ct.peak);
    const cons = new Float32Array(ct.counts.length);
    for (let i = 0; i < ct.counts.length; i++) cons[i] = ct.counts[i] / peak;
    // Each CPU is normalized to ITS OWN peak so a busy CPU fills its
    // sub-track to ~1.0 (matches samply). Sharing a single peak across
    // all CPUs would scale every other CPU down whenever any single
    // CPU briefly spiked higher, which is misleading: a fully-busy
    // CPU shouldn't read as 60% just because one other CPU had a
    // momentary outlier. The raw peak per CPU is preserved on the
    // outer category for tooltip display.
    const perCpuMap = new Map<number, Float32Array>();
    for (const t of perCpu.tracks) {
      if (t.category !== id || t.cpu == null) continue;
      const denom = Math.max(1, t.peak);
      const norm = new Float32Array(t.counts.length);
      for (let i = 0; i < t.counts.length; i++) norm[i] = t.counts[i] / denom;
      perCpuMap.set(t.cpu, norm);
    }
    out.push({
      id,
      label: CATEGORY_META[id].label,
      short: CATEGORY_META[id].short,
      hue: CATEGORY_META[id].hue,
      consolidated: cons,
      perCpu: perCpuMap,
      rawPeak: peak,
    });
  }
  return out;
}
