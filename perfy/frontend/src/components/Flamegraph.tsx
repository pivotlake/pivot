// Flame graph in the design's style: dense stacked yellow bars, tiny text.
// Top row is the root; each row below is one stack frame deeper.
// Click a frame -> opens code panel for that function.

import { useMemo, useState } from "react";
import type { FlameNode, FlameResponse } from "../api";

interface FlameCell {
  x: number;        // 0..100
  w: number;        // 0..100
  label: string;
  hot: boolean;
  total: number;
}

interface FlameRow {
  cells: FlameCell[];
}

const ROW_H = 16;

function buildRows(resp: FlameResponse): FlameRow[] {
  if (!resp.nodes.length) return [];
  // Layout: each node gets [x, w] in % of the row's width, derived from
  // parent's [x, w] and the child's `total` share of the parent.
  type Layout = { x: number; w: number; depth: number; label: string; total: number };
  const layout = new Map<number, Layout>();
  layout.set(0, { x: 0, w: 100, depth: 0, label: "(all)", total: resp.nodes[0]?.total ?? 0 });

  const childrenByParent = new Map<number, FlameNode[]>();
  for (const n of resp.nodes) {
    if (n.id === 0) continue;
    if (!childrenByParent.has(n.parent)) childrenByParent.set(n.parent, []);
    childrenByParent.get(n.parent)!.push(n);
  }

  // BFS so parents are laid out before children
  const queue: number[] = [0];
  while (queue.length) {
    const id = queue.shift()!;
    const parent = layout.get(id)!;
    const kids = childrenByParent.get(id) || [];
    kids.sort((a, b) => b.total - a.total);
    let xCursor = parent.x;
    const sumChildren = kids.reduce((s, c) => s + c.total, 0);
    for (const c of kids) {
      const w = sumChildren === 0 ? 0 : (c.total / parent.total) * parent.w;
      layout.set(c.id, {
        x: xCursor,
        w,
        depth: c.depth,
        label: c.label,
        total: c.total,
      });
      xCursor += w;
      queue.push(c.id);
    }
  }

  // Group by depth, skip root (depth 0, label "(all)").
  const byDepth = new Map<number, Layout[]>();
  for (const [id, info] of layout) {
    if (id === 0) continue;
    if (info.w < 0.05) continue; // drop sub-pixel cells
    if (!byDepth.has(info.depth)) byDepth.set(info.depth, []);
    byDepth.get(info.depth)!.push(info);
  }

  const maxDepth = Math.max(0, ...byDepth.keys());
  const rows: FlameRow[] = [];
  for (let d = 1; d <= maxDepth; d++) {
    const arr = byDepth.get(d) || [];
    arr.sort((a, b) => a.x - b.x);
    rows.push({
      cells: arr.map((l) => ({
        x: l.x,
        w: l.w,
        label: l.label,
        hot: l.w >= 8,
        total: l.total,
      })),
    });
  }
  return rows;
}

export function FlameGraph({
  data,
  onSelectFrame,
  selectedFrame,
}: {
  data: FlameResponse;
  onSelectFrame: (label: string) => void;
  selectedFrame: string | null;
}) {
  const rows = useMemo(() => buildRows(data), [data]);
  // Hover state: which cell is under the cursor + the cursor position so
  // the tooltip can follow the mouse. `null` when nothing is hovered.
  const [hover, setHover] = useState<{
    ri: number;
    ci: number;
    label: string;
    total: number;
    x: number;
    y: number;
  } | null>(null);
  const totalSamples = data.total > 0 ? data.total : 1;

  return (
    <div
      style={{
        background: "white",
        borderTop: "1px solid oklch(0.92 0.005 250)",
        borderBottom: "1px solid oklch(0.92 0.005 250)",
        padding: "8px 0",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "0 14px 6px",
        }}
      >
        <span
          style={{
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: 0.5,
            color: "oklch(0.45 0.01 250)",
          }}
        >
          Flame graph
        </span>
        <span style={{ fontSize: 11, color: "oklch(0.55 0.01 250)" }}>
          click a frame to open its code · {data.total.toLocaleString()} samples
        </span>
        <div style={{ flex: 1 }} />
        <span
          style={{
            fontSize: 10,
            color: "oklch(0.55 0.01 250)",
            fontFamily: "ui-monospace, monospace",
          }}
        >
          {rows.length} levels · root → leaves
        </span>
      </div>
      <div style={{ position: "relative", padding: "0 8px" }}>
        {rows.map((row, ri) => (
          <div
            key={ri}
            style={{ position: "relative", height: ROW_H, marginBottom: 1 }}
          >
            {row.cells.map((cell, ci) => {
              const isHot = cell.hot;
              const isSel = selectedFrame === cell.label;
              const isHover = hover?.ri === ri && hover.ci === ci;
              // Hover darkens the row by lowering lightness on the OKLCH
              // tone — keeps the same hue family (yellow) so the cell
              // still reads as part of the flame strip.
              const baseBg = isSel
                ? "oklch(0.65 0.2 95)"
                : isHot
                  ? "oklch(0.85 0.18 95)"
                  : "oklch(0.92 0.13 95)";
              const hoverBg = isSel
                ? "oklch(0.55 0.2 95)"
                : isHot
                  ? "oklch(0.72 0.2 90)"
                  : "oklch(0.82 0.16 90)";
              return (
                <div
                  key={ci}
                  onClick={() => onSelectFrame(cell.label)}
                  onMouseEnter={(e) =>
                    setHover({
                      ri,
                      ci,
                      label: cell.label,
                      total: cell.total,
                      x: e.clientX,
                      y: e.clientY,
                    })
                  }
                  onMouseMove={(e) =>
                    setHover((h) =>
                      h && h.ri === ri && h.ci === ci
                        ? { ...h, x: e.clientX, y: e.clientY }
                        : h,
                    )
                  }
                  onMouseLeave={() =>
                    setHover((h) =>
                      h && h.ri === ri && h.ci === ci ? null : h,
                    )
                  }
                  style={{
                    position: "absolute",
                    left: cell.x + "%",
                    width: cell.w + "%",
                    top: 0,
                    bottom: 0,
                    background: isHover ? hoverBg : baseBg,
                    border: isHover
                      ? "1px solid oklch(0.55 0.2 90)"
                      : "1px solid oklch(0.78 0.18 90)",
                    borderRadius: 1,
                    fontSize: 9.5,
                    fontFamily: "ui-monospace, SFMono-Regular, monospace",
                    color: isSel ? "white" : "oklch(0.25 0.06 80)",
                    display: "flex",
                    alignItems: "center",
                    paddingLeft: 4,
                    overflow: "hidden",
                    whiteSpace: "nowrap",
                    cursor: "pointer",
                    boxSizing: "border-box",
                    transition:
                      "background 80ms ease-out, border-color 80ms ease-out",
                  }}
                >
                  <span
                    style={{
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {cell.label}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      {hover && (
        <FlameTooltip
          x={hover.x}
          y={hover.y}
          label={hover.label}
          total={hover.total}
          totalSamples={totalSamples}
        />
      )}
    </div>
  );
}

/// Floating tooltip pinned to the cursor with a small offset. Anchored on
/// the right side of the cursor unless that would clip the viewport, in
/// which case we flip to the left. `pointerEvents: none` so hovering the
/// tooltip itself never interferes with the cell underneath.
function FlameTooltip({
  x,
  y,
  label,
  total,
  totalSamples,
}: {
  x: number;
  y: number;
  label: string;
  total: number;
  totalSamples: number;
}) {
  const PAD = 12;
  const MAX_W = 480;
  const flipLeft = x + PAD + MAX_W > window.innerWidth;
  const left = flipLeft ? x - PAD : x + PAD;
  const pct = (total / totalSamples) * 100;
  return (
    <div
      style={{
        position: "fixed",
        left,
        top: y + PAD,
        transform: flipLeft ? "translateX(-100%)" : undefined,
        maxWidth: MAX_W,
        background: "oklch(0.18 0.01 250)",
        color: "oklch(0.96 0.005 250)",
        border: "1px solid oklch(0.3 0.02 250)",
        borderRadius: 4,
        padding: "6px 9px",
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        fontSize: 11,
        lineHeight: 1.45,
        boxShadow: "0 4px 14px oklch(0.1 0.02 250 / 0.35)",
        pointerEvents: "none",
        zIndex: 1000,
        whiteSpace: "normal",
        wordBreak: "break-all",
      }}
    >
      <div style={{ fontWeight: 600, color: "oklch(0.95 0.04 95)" }}>
        {label}
      </div>
      <div style={{ marginTop: 4, color: "oklch(0.78 0.005 250)" }}>
        {total.toLocaleString()} samples · {pct >= 0.05 ? pct.toFixed(2) : "<0.05"}%
      </div>
    </div>
  );
}
