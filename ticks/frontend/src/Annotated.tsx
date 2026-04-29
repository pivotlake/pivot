import React, { useCallback, useEffect, useMemo, useState } from "react";
import { DisplayMode, FuncData, InstructionLine, Line, Summary } from "./types";
import DetailPanel from "./DetailPanel";

const CACHE_LEVELS = ["L1", "LFB", "L2", "L3", "DRAM", "REM", "N-M"];
const CACHE_WEIGHTS: Record<string, number> = {
  L1: 4,
  LFB: 9,
  L2: 14,
  L3: 50,
  DRAM: 250,
  REM: 400,
  "N-M": 1,
};
const NBSP = " ";
const INSTR_INDENT = 4;
const COL_WIDTH = 5;

const pad = (n: number) => NBSP.repeat(Math.max(0, n));

function heatClass(pct: number): string {
  if (pct >= 5.0) return "hot-bold";
  if (pct >= 1.0) return "hot";
  if (pct >= 0.3) return "warm";
  return "cold";
}

function fmtCacheValue(
  count: number,
  level: string,
  mode: DisplayMode,
  totalUw: number,
  totalW: number,
): string {
  if (!count) return pad(COL_WIDTH);
  if (mode === "absolute") return count.toString().padStart(COL_WIDTH);
  if (mode === "percent") {
    const p = totalUw ? (100 * count) / totalUw : 0;
    return p.toFixed(1).padStart(COL_WIDTH);
  }
  const w = CACHE_WEIGHTS[level] ?? 1;
  const p = totalW ? (100 * count * w) / totalW : 0;
  return p.toFixed(1).padStart(COL_WIDTH);
}

function fmtCyc(cycles: number, cyclesPct: number, mode: DisplayMode): string {
  if (!cycles) return pad(COL_WIDTH);
  if (mode === "absolute") return cycles.toString().padStart(COL_WIDTH);
  return cyclesPct.toFixed(1).padStart(COL_WIDTH);
}

interface RowProps {
  idx: number;
  line: Line;
  mode: DisplayMode;
  totals: { unweighted: number; weighted: number; cycles: number };
  addrWidth: number;
  gutterWidth: number;
  selected: boolean;
  searchHit: boolean;
  onSelect: (idx: number) => void;
}

function LineRow({
  idx,
  line,
  mode,
  totals,
  addrWidth,
  gutterWidth,
  selected,
  searchHit,
  onSelect,
}: RowProps) {
  const labelW = addrWidth ? addrWidth + 1 : 0;
  const labelSp = labelW ? 1 : 0;
  const gutterSp = gutterWidth ? 1 : 0;
  const className = `line ${selected ? "selected" : ""} ${
    searchHit ? "search-hit" : ""
  }`;

  if (line.type === "function_header") {
    const lead =
      (CACHE_LEVELS.length + 1) * (COL_WIDTH + 1) +
      labelW +
      labelSp +
      gutterWidth +
      gutterSp +
      INSTR_INDENT;
    return (
      <div className="line func-header" id={`line-${idx}`}>
        {pad(lead)}
        {line.name}:
      </div>
    );
  }
  if (line.type === "separator") {
    return (
      <div className="line separator" id={`line-${idx}`}>
        {"─".repeat(120)}
      </div>
    );
  }
  if (line.type === "source") {
    return (
      <div className={className} id={`line-${idx}`}>
        {pad((CACHE_LEVELS.length + 1) * (COL_WIDTH + 1))}
        {pad(labelW + labelSp)}
        {(line.gutter || []).map((c, i) => {
          const ch = c.kind === "lane" && c.glyph === "│" ? "│" : NBSP;
          return (
            <span key={i} className="gutter-cell">
              {ch}
            </span>
          );
        })}
        {pad(gutterSp)}
        <span className="source">{line.text}</span>
      </div>
    );
  }
  // instruction
  const cyc = line.cycles_pct || 0;
  const heat = heatClass(cyc);
  const isJump = /^j[a-z]+$/.test(line.mnem || "");
  const mnemCls = `mnem ${isJump ? "is-jump" : heat}`;
  const labelText = line.is_target
    ? `${line.offset.toString(16)}:`.padStart(labelW, NBSP)
    : pad(labelW);

  return (
    <div className={className} id={`line-${idx}`} onClick={() => onSelect(idx)}>
      {CACHE_LEVELS.map((lvl) => {
        const count = line.cache_counts[lvl] || 0;
        const text = fmtCacheValue(
          count,
          lvl,
          mode,
          totals.unweighted,
          totals.weighted,
        );
        return (
          <React.Fragment key={lvl}>
            <span className={`cost lvl-${lvl.replace("-", "")}`}>{text}</span>
            <span className="cost-sp">{NBSP}</span>
          </React.Fragment>
        );
      })}
      <span className={`cyc ${heat}`}>{fmtCyc(line.cycles, cyc, mode)}</span>
      <span className="cost-sp">{NBSP}</span>
      <span className="label">{labelText}</span>
      {pad(labelSp)}
      {(line.gutter || []).map((c, i) => (
        <span key={i} className={`gutter-cell ${c.kind}`}>
          {c.glyph}
        </span>
      ))}
      {pad(gutterSp + INSTR_INDENT)}
      <span className={mnemCls}>{(line.mnem || "").padEnd(8, NBSP)}</span>
      {NBSP}
      {NBSP}
      <span className="operands">{line.operands}</span>
    </div>
  );
}

interface AnnotatedProps {
  funcName: string;
  summary: Summary;
  onBack: () => void;
}

export default function Annotated({ funcName, onBack }: AnnotatedProps) {
  const [data, setData] = useState<FuncData | null>(null);
  const [mode, setMode] = useState<DisplayMode>("weighted");
  const [search, setSearch] = useState("");
  const [selectedLine, setSelectedLine] = useState<number | null>(null);

  useEffect(() => {
    setData(null);
    setSelectedLine(null);
    fetch(`/api/function/${encodeURIComponent(funcName)}`)
      .then((r) => r.json())
      .then(setData)
      .catch((e) => console.error("function fetch failed", e));
  }, [funcName]);

  const hottestIdx = useMemo(() => {
    if (!data) return -1;
    let best = 0;
    let hit = -1;
    data.lines.forEach((l, i) => {
      if (l.type === "instruction" && l.cycles > best) {
        best = l.cycles;
        hit = i;
      }
    });
    return hit;
  }, [data]);

  const searchMatches = useMemo(() => {
    if (!data || !search) return new Set<number>();
    const q = search.toLowerCase();
    const hits = new Set<number>();
    data.lines.forEach((l, i) => {
      let text = "";
      if (l.type === "instruction") text = `${l.addr} ${l.mnem} ${l.operands}`;
      else if (l.type === "source") text = l.text;
      else if (l.type === "function_header") text = l.name;
      if (text.toLowerCase().includes(q)) hits.add(i);
    });
    return hits;
  }, [data, search]);

  const scrollTo = useCallback((idx: number) => {
    document
      .getElementById(`line-${idx}`)
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
    setSelectedLine(idx);
  }, []);

  const goHottest = useCallback(() => {
    if (hottestIdx >= 0) scrollTo(hottestIdx);
  }, [hottestIdx, scrollTo]);

  const goNextSearch = useCallback(() => {
    if (!searchMatches.size) return;
    const arr = Array.from(searchMatches).sort((a, b) => a - b);
    const cur = selectedLine ?? -1;
    const next = arr.find((x) => x > cur) ?? arr[0];
    scrollTo(next);
  }, [searchMatches, selectedLine, scrollTo]);

  if (!data) return <div className="container">Loading…</div>;

  const totals = data.totals;
  const gutterWidth =
    data.max_lanes > 0 || data.lines.some((l) => "gutter" in l && (l as any).gutter?.length)
      ? data.max_lanes + 1
      : 0;

  const selectedInstr =
    selectedLine !== null && data.lines[selectedLine]?.type === "instruction"
      ? (data.lines[selectedLine] as InstructionLine)
      : null;

  return (
    <div className="container">
      <div className="toolbar">
        <button className="btn" onClick={onBack}>
          ← functions
        </button>
        <h3>{data.name}</h3>
      </div>
      <div className="controls">
        <div className="mode-buttons">
          <button
            className={`btn ${mode === "weighted" ? "active" : ""}`}
            onClick={() => setMode("weighted")}
          >
            weighted %
          </button>
          <button
            className={`btn ${mode === "percent" ? "active" : ""}`}
            onClick={() => setMode("percent")}
          >
            sample %
          </button>
          <button
            className={`btn ${mode === "absolute" ? "active" : ""}`}
            onClick={() => setMode("absolute")}
          >
            count
          </button>
        </div>
        <button className="btn" onClick={goHottest} disabled={hottestIdx < 0}>
          ↑ hottest
        </button>
        <input
          className="search-input"
          placeholder="search disassembly…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") goNextSearch();
          }}
        />
        <span className="subtle">
          {searchMatches.size > 0 ? `${searchMatches.size} matches` : ""}
        </span>
      </div>

      <pre className="annotated">
        {data.lines.map((line, i) => (
          <LineRow
            key={i}
            idx={i}
            line={line}
            mode={mode}
            totals={totals}
            addrWidth={data.addr_width}
            gutterWidth={gutterWidth}
            selected={selectedLine === i}
            searchHit={searchMatches.has(i)}
            onSelect={setSelectedLine}
          />
        ))}
      </pre>

      {selectedInstr && (
        <DetailPanel
          line={selectedInstr}
          onClose={() => setSelectedLine(null)}
        />
      )}
    </div>
  );
}
