// Code panel — source on the left, asm on the right.
// Two mode pickers (Cycles + Memory) up top control how counters render.
// Clicking a source line scrolls the asm pane to its first mapped insn;
// clicking an asm insn scrolls the source pane to the matching line.

import { useEffect, useMemo, useRef, useState } from "react";
import { Highlight, themes, type Language } from "prism-react-renderer";
import {
  CACHE_WEIGHTS,
  CATEGORY_META,
  type AnnotateLine,
  type AnnotateResponse,
  type SourcePane,
} from "../api";

const CAT_FOR_BAR = ["cycles", "dram", "l1", "l2", "l3"] as const;
type CatId = (typeof CAT_FOR_BAR)[number];

const CACHE_KEY: Record<CatId, string> = {
  cycles: "CYC",
  dram: "DRAM",
  l1: "L1",
  l2: "L2",
  l3: "L3",
};

export type CyclesMode = "relative" | "absolute" | "raw";
export type MemoryMode = "relative" | "weighted" | "raw";

interface Props {
  data: AnnotateResponse;
  cyclesMode: CyclesMode;
  /// Reserved for future use (e.g. a per-pane mode picker); unused today
  /// since the modes live in the toolbar.
  setCyclesMode?: (m: CyclesMode) => void;
  memoryMode: MemoryMode;
  setMemoryMode?: (m: MemoryMode) => void;
  totalProfileCycles: number;
  /// Called whenever the user selects an asm row. Passed `null` when the
  /// selection is cleared (clicked again, or clicked outside).
  onAsmSelect?: (insn: AnnotateLine | null) => void;
}

const COUNTER_COL_W = 60;
/// Width of the trailing actions slot (jump-button column). Reserved on
/// every asm row + the header so counters always land in the same x.
const ACTIONS_COL_W = 28;

// ── Display value selection ────────────────────────────────────────────────

function cyclesValue(
  insn: AnnotateLine,
  mode: CyclesMode,
  funcCycles: number,
  profileCycles: number,
): { display: string; pct: number } {
  const c = insn.cycles ?? 0;
  if (c <= 0) return { display: "", pct: 0 };
  if (mode === "raw") return { display: c.toLocaleString(), pct: 0 };
  if (mode === "absolute") {
    const pct = profileCycles > 0 ? (c / profileCycles) * 100 : 0;
    return { display: pct >= 0.05 ? pct.toFixed(2) : "", pct };
  }
  // relative
  const pct = funcCycles > 0 ? (c / funcCycles) * 100 : 0;
  return { display: pct >= 0.05 ? pct.toFixed(2) : "", pct };
}

function memoryValue(
  count: number,
  cat: "L1" | "L2" | "L3" | "DRAM",
  mode: MemoryMode,
  funcCacheTotals: Record<string, number>,
  funcWeightedTotal: number,
): { display: string; pct: number } {
  if (count <= 0) return { display: "", pct: 0 };
  if (mode === "raw") {
    // For shading we still want a sense of "hot" vs "cold", so derive a
    // pct from the function's own-category total. The display, though, is
    // the raw count.
    const tot = funcCacheTotals[cat] ?? 0;
    const pct = tot > 0 ? (count / tot) * 100 : 0;
    return { display: count.toLocaleString(), pct };
  }
  if (mode === "weighted") {
    const w = CACHE_WEIGHTS[cat] ?? 1;
    const pct = funcWeightedTotal > 0 ? ((count * w) / funcWeightedTotal) * 100 : 0;
    return { display: pct >= 0.05 ? pct.toFixed(2) : "", pct };
  }
  // relative-to-itself within the function
  const tot = funcCacheTotals[cat] ?? 0;
  const pct = tot > 0 ? (count / tot) * 100 : 0;
  return { display: pct >= 0.05 ? pct.toFixed(2) : "", pct };
}

function intensityBg(pct: number, hue: number): string {
  // 0% → transparent ish, 30%+ → strongly tinted.
  const p = Math.min(1, pct / 30);
  if (p <= 0) return "transparent";
  return `oklch(${0.99 - p * 0.1} ${0.01 + p * 0.13} ${hue})`;
}

function intensityFg(pct: number, hue: number): string {
  const p = Math.min(1, pct / 30);
  if (p < 0.15) return "oklch(0.45 0.01 250)";
  return `oklch(${0.4 - p * 0.16} ${0.06 + p * 0.16} ${hue})`;
}

function CounterCell({
  display,
  pct,
  hue,
}: {
  display: string;
  pct: number;
  hue: number;
}) {
  return (
    <div
      style={{
        // border-box so `width: 100%` == outer width and the
        // paddingRight actually narrows the flex content area instead
        // of widening the pill 6px past its parent. Without this the
        // flex-end alignment puts the text 6px to the right of where
        // the header label sits.
        boxSizing: "border-box",
        width: "100%",
        height: 18,
        borderRadius: 3,
        background: intensityBg(pct, hue),
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        paddingRight: 6,
        fontSize: 10.5,
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        color: intensityFg(pct, hue),
        fontVariantNumeric: "tabular-nums",
        fontWeight: 400 + Math.round(Math.min(1, pct / 30) * 300),
      }}
    >
      {display}
    </div>
  );
}

// ── Headers ────────────────────────────────────────────────────────────────

function HeaderRow({
  leftLabel,
  leftWidth,
  bodyLabel,
  bodySlot,
  showCounters,
}: {
  leftLabel: string;
  leftWidth: number;
  bodyLabel?: string;
  /// Optional ReactNode to render in the body column instead of a static
  /// label (e.g. a search input). When provided, `bodyLabel` is ignored.
  bodySlot?: React.ReactNode;
  showCounters: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        height: 26,
        borderBottom: "1px solid oklch(0.92 0.005 250)",
        background: "oklch(0.985 0.003 250)",
        fontSize: 10,
        textTransform: "uppercase",
        letterSpacing: 0.5,
        color: "oklch(0.5 0.01 250)",
        fontFamily: "ui-monospace, monospace",
      }}
    >
      <div
        style={{
          width: leftWidth,
          paddingLeft: 12,
          color: "oklch(0.6 0.01 250)",
        }}
      >
        {leftLabel}
      </div>
      <div
        style={{
          flex: 1,
          paddingLeft: 12,
          paddingRight: 8,
          overflow: "hidden",
          whiteSpace: "nowrap",
          textOverflow: "ellipsis",
          display: "flex",
          alignItems: "center",
        }}
      >
        {bodySlot ?? bodyLabel}
      </div>
      {showCounters &&
        CAT_FOR_BAR.map((cid) => (
          <div
            key={cid}
            style={{
              width: COUNTER_COL_W,
              // border-box keeps the outer width fixed at COUNTER_COL_W
              // regardless of padding, so header columns and body
              // columns share the same column width even though their
              // padding differs (header has 10px on the right to
              // mirror the CounterCell's own internal paddingRight).
              boxSizing: "border-box",
              padding: "0 10px 0 4px",
              textAlign: "right",
              borderLeft: "1px solid oklch(0.93 0.005 250)",
              color: `oklch(0.5 0.06 ${CATEGORY_META[cid].hue})`,
            }}
          >
            {CATEGORY_META[cid].short}
          </div>
        ))}
      {showCounters && (
        <div
          // Action slot — see asm pane row layout. We reserve the same
          // pixel width in the header so the asm "→" jump button never
          // shifts the counter columns.
          style={{ width: ACTIONS_COL_W, flexShrink: 0 }}
        />
      )}
    </div>
  );
}

// ── Source pane ────────────────────────────────────────────────────────────

function languageOf(file: string): Language | null {
  const ext = file.split(".").pop()?.toLowerCase() ?? "";
  switch (ext) {
    case "rs":
      return "rust";
    case "c":
    case "h":
      return "c";
    case "cc":
    case "cpp":
    case "cxx":
    case "hpp":
    case "hxx":
      return "cpp";
    case "py":
      return "python";
    case "js":
    case "jsx":
    case "mjs":
    case "cjs":
      return "javascript";
    case "ts":
    case "tsx":
      return "tsx";
    case "go":
      return "go";
    case "java":
      return "java";
    default:
      return null;
  }
}

function basename(path: string): string {
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}

/// Tiny inline search box: case-insensitive substring match, Enter cycles
/// forward (Shift+Enter cycles back), Esc clears. The "n / m" counter is
/// 1-indexed for display; an empty query hides the counter.
function SearchBar({
  placeholder,
  value,
  onChange,
  matchCount,
  currentIdx,
  onStep,
}: {
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  matchCount: number;
  currentIdx: number;
  /// Called with +1 (next) or -1 (prev). Wrapping is handled by the caller.
  onStep: (delta: number) => void;
}) {
  const showCounter = value.length > 0;
  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        background: "white",
        border: "1px solid oklch(0.9 0.005 250)",
        borderRadius: 4,
        padding: "0 6px",
        height: 20,
        textTransform: "none",
        letterSpacing: 0,
      }}
    >
      <span style={{ color: "oklch(0.6 0.01 250)", fontSize: 11 }}>⌕</span>
      <input
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onStep(e.shiftKey ? -1 : 1);
          } else if (e.key === "Escape") {
            e.preventDefault();
            onChange("");
          }
        }}
        style={{
          border: "none",
          outline: "none",
          background: "transparent",
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontSize: 11,
          color: "oklch(0.25 0.01 250)",
          width: 160,
        }}
      />
      {showCounter && (
        <span
          style={{
            fontSize: 10,
            color: matchCount > 0 ? "oklch(0.5 0.01 250)" : "oklch(0.55 0.16 25)",
            fontVariantNumeric: "tabular-nums",
            fontFamily: "ui-monospace, SFMono-Regular, monospace",
          }}
        >
          {matchCount > 0 ? `${currentIdx + 1}/${matchCount}` : "0"}
        </span>
      )}
    </div>
  );
}

/// Yellow background tint for rows that match the current search query.
const MATCH_BG = "oklch(0.95 0.12 95)";

function SourceTabs({
  panes,
  active,
  onSelect,
}: {
  panes: SourcePane[];
  active: string | null;
  onSelect: (file: string) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "stretch",
        height: 28,
        borderBottom: "1px solid oklch(0.92 0.005 250)",
        background: "oklch(0.985 0.003 250)",
        overflow: "auto",
      }}
    >
      {panes.map((p) => {
        const isSel = p.file === active;
        return (
          <button
            key={p.file}
            onClick={() => onSelect(p.file)}
            title={p.file}
            style={{
              border: "none",
              borderRight: "1px solid oklch(0.92 0.005 250)",
              background: isSel ? "white" : "transparent",
              color: isSel ? "oklch(0.25 0.08 250)" : "oklch(0.5 0.01 250)",
              fontWeight: isSel ? 600 : 400,
              padding: "0 12px",
              fontSize: 11,
              fontFamily: "ui-monospace, SFMono-Regular, monospace",
              cursor: "pointer",
              whiteSpace: "nowrap",
              borderBottom: isSel
                ? "2px solid oklch(0.6 0.18 250)"
                : "2px solid transparent",
              marginBottom: -1,
            }}
          >
            {basename(p.file)}
          </button>
        );
      })}
    </div>
  );
}

function SourcePaneView({
  pane,
  asmByLine,
  funcTotals,
  cyclesMode,
  memoryMode,
  totalProfileCycles,
  activeLine,
  setActiveLine,
  sourceRefs,
}: {
  pane: SourcePane;
  /// Already filtered to insns whose source_file === pane.file.
  asmByLine: Map<number, AnnotateLine[]>;
  funcTotals: AnnotateResponse["function_totals"];
  cyclesMode: CyclesMode;
  memoryMode: MemoryMode;
  totalProfileCycles: number;
  activeLine: number | null;
  setActiveLine: (n: number) => void;
  sourceRefs: React.MutableRefObject<Map<number, HTMLDivElement>>;
}) {
  const language = languageOf(pane.file);
  // Concatenate the visible source so prism can highlight a single block;
  // we'll render line-by-line below.
  const code = pane.lines.map((l) => l.text).join("\n");
  const funcWeighted = useMemo(() => {
    const cc = funcTotals.cache_counts;
    return (
      (cc.L1 ?? 0) * CACHE_WEIGHTS.L1 +
      (cc.L2 ?? 0) * CACHE_WEIGHTS.L2 +
      (cc.L3 ?? 0) * CACHE_WEIGHTS.L3 +
      (cc.DRAM ?? 0) * CACHE_WEIGHTS.DRAM
    );
  }, [funcTotals]);

  // Per-pane search state. Reset whenever the user switches files (a new
  // `pane` instance) so stale matches don't carry over.
  const [search, setSearch] = useState("");
  const [searchIdx, setSearchIdx] = useState(0);
  useEffect(() => {
    setSearch("");
    setSearchIdx(0);
  }, [pane.file]);

  const matchLines = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [] as number[];
    return pane.lines
      .filter((l) => l.text.toLowerCase().includes(q))
      .map((l) => l.line);
  }, [search, pane.lines]);

  // Clamp the cursor when the matches list shrinks (typing extends query).
  useEffect(() => {
    if (searchIdx >= matchLines.length) setSearchIdx(0);
  }, [matchLines.length, searchIdx]);

  // Auto-scroll to the first match when a fresh query becomes non-empty.
  useEffect(() => {
    if (!matchLines.length) return;
    const target = matchLines[searchIdx] ?? matchLines[0];
    const el = sourceRefs.current.get(target);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [matchLines, searchIdx, sourceRefs]);

  const matchSet = useMemo(() => new Set(matchLines), [matchLines]);

  const stepSearch = (delta: number) => {
    if (!matchLines.length) return;
    const next = (searchIdx + delta + matchLines.length) % matchLines.length;
    setSearchIdx(next);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minWidth: 0,
      }}
    >
      <div
        style={{
          flex: 1,
          overflow: "auto",
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontSize: 12,
        }}
      >
        {/* Header lives INSIDE the scroll container with position: sticky.
            That way, when a vertical scrollbar appears, header and body
            rows share the same available width — without sticky-inside
            the header would span the pane's full width while body rows
            would be narrower by the scrollbar gutter, shifting every
            counter column ~15px to the left of its header. */}
        <div
          style={{
            position: "sticky",
            top: 0,
            zIndex: 1,
            background: "white",
          }}
        >
          <HeaderRow
            leftLabel="line"
            leftWidth={56}
            bodySlot={
              <SearchBar
                placeholder={basename(pane.file)}
                value={search}
                onChange={(v) => {
                  setSearch(v);
                  setSearchIdx(0);
                }}
                matchCount={matchLines.length}
                currentIdx={searchIdx}
                onStep={stepSearch}
              />
            }
            showCounters={true}
          />
        </div>
        <Highlight
          code={code}
          language={(language ?? "clike") as Language}
          theme={themes.github}
        >
          {({ tokens, getTokenProps }) => (
            <>
              {tokens.map((tokenLine, idx) => {
                const ln = pane.lines[idx];
                if (!ln) return null;
                const hits = asmByLine.get(ln.line) ?? [];
                const cyclesAgg = hits.reduce((s, h) => s + (h.cycles ?? 0), 0);
                const cv = cyclesValue(
                  { cycles: cyclesAgg } as AnnotateLine,
                  cyclesMode,
                  funcTotals.cycles,
                  totalProfileCycles,
                );
                const isActive = activeLine === ln.line;
                const isMatch = matchSet.has(ln.line);
                const isCurrentMatch =
                  isMatch && matchLines[searchIdx] === ln.line;
                return (
                  <div
                    key={ln.line}
                    ref={(el) => {
                      if (el) sourceRefs.current.set(ln.line, el);
                      else sourceRefs.current.delete(ln.line);
                    }}
                    onClick={() => hits.length && setActiveLine(ln.line)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      minHeight: 22,
                      background: isActive
                        ? "oklch(0.95 0.05 250)"
                        : isMatch
                          ? MATCH_BG
                          : intensityBg(cv.pct, 38),
                      borderLeft: isCurrentMatch
                        ? "2px solid oklch(0.7 0.18 90)"
                        : isActive
                          ? "2px solid oklch(0.6 0.18 250)"
                          : "2px solid transparent",
                      cursor: hits.length ? "pointer" : "default",
                    }}
                  >
                    <div
                      style={{
                        width: 56,
                        paddingLeft: 12,
                        color: "oklch(0.6 0.01 250)",
                        fontVariantNumeric: "tabular-nums",
                        fontSize: 11,
                      }}
                    >
                      {ln.line}
                    </div>
                    <pre
                      style={{
                        margin: 0,
                        paddingLeft: 12,
                        flex: 1,
                        // Without `minWidth: 0` a flex item with
                        // `whiteSpace: pre` keeps its content's intrinsic
                        // width as its min, so long lines push the
                        // counters off the right edge of the row.
                        minWidth: 0,
                        whiteSpace: "pre",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {tokenLine.length === 0 ? (
                        " "
                      ) : (
                        tokenLine.map((token, ki) => (
                          // eslint-disable-next-line react/jsx-key
                          <span {...getTokenProps({ token })} key={ki} />
                        ))
                      )}
                    </pre>
                    {/* Counters: cycles + 4 cache levels */}
                    <div
                      style={{ width: COUNTER_COL_W, boxSizing: "border-box", padding: "2px 4px", borderLeft: "1px solid oklch(0.93 0.005 250)" }}
                    >
                      <CounterCell
                        display={cv.display + (cv.display && cyclesMode !== "raw" ? "%" : "")}
                        pct={cv.pct}
                        hue={CATEGORY_META.cycles.hue}
                      />
                    </div>
                    {(["dram", "l1", "l2", "l3"] as const).map((cid) => {
                      const cat = CACHE_KEY[cid] as "L1" | "L2" | "L3" | "DRAM";
                      const total = hits.reduce(
                        (s, h) => s + (h.cache_counts?.[cat] ?? 0),
                        0,
                      );
                      const mv = memoryValue(
                        total,
                        cat,
                        memoryMode,
                        funcTotals.cache_counts,
                        funcWeighted,
                      );
                      return (
                        <div
                          key={cid}
                          style={{ width: COUNTER_COL_W, boxSizing: "border-box", padding: "2px 4px", borderLeft: "1px solid oklch(0.93 0.005 250)" }}
                        >
                          <CounterCell
                            display={mv.display + (mv.display && memoryMode !== "raw" ? "%" : "")}
                            pct={mv.pct}
                            hue={CATEGORY_META[cid].hue}
                          />
                        </div>
                      );
                    })}
                    {/* Empty trailing slot — width matches the asm
                        pane's actions column so the source pane's
                        rightmost counter sits at the same x position
                        as the asm pane's, keeping the two panes
                        visually aligned. */}
                    <div style={{ width: ACTIONS_COL_W, flexShrink: 0 }} />
                  </div>
                );
              })}
            </>
          )}
        </Highlight>
      </div>
    </div>
  );
}

// ── Asm pane ───────────────────────────────────────────────────────────────

function AsmPaneView({
  data,
  funcTotals,
  cyclesMode,
  memoryMode,
  totalProfileCycles,
  activeAddr,
  activeLine,
  activeFile,
  jumpTargets,
  asmRefs,
  onSelectInsn,
  onJumpTo,
}: {
  data: AnnotateResponse;
  funcTotals: AnnotateResponse["function_totals"];
  cyclesMode: CyclesMode;
  memoryMode: MemoryMode;
  totalProfileCycles: number;
  activeAddr: string | null;
  activeLine: number | null;
  activeFile: string | null;
  jumpTargets: Set<number>;
  asmRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
  onSelectInsn: (insn: AnnotateLine | null) => void;
  onJumpTo: (offset: number) => void;
}) {
  // SVG overlay: drawn whenever the currently-selected asm row is a jump
  // with an in-function target. No separate pin state — selection alone
  // drives it.
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const innerRef = useRef<HTMLDivElement | null>(null);
  const [arcGeom, setArcGeom] = useState<{
    y1: number;
    y2: number;
  } | null>(null);

  // Look up the active jump (if any) — derived from `activeAddr`.
  const activeJump = useMemo(() => {
    if (!activeAddr) return null;
    for (const ln of data.lines) {
      if (
        ln.kind === "insn" &&
        ln.addr === activeAddr &&
        ln.jump_target_offset != null
      ) {
        return { fromAddr: ln.addr!, toOffset: ln.jump_target_offset! };
      }
    }
    return null;
  }, [activeAddr, data.lines]);

  // Recompute arc geometry whenever the active jump changes or layout shifts.
  useEffect(() => {
    if (!activeJump || !innerRef.current) {
      setArcGeom(null);
      return;
    }
    const fromEl = asmRefs.current.get(activeJump.fromAddr);
    let toAddr: string | null = null;
    for (const ln of data.lines) {
      if (ln.kind === "insn" && ln.offset === activeJump.toOffset) {
        toAddr = ln.addr ?? null;
        break;
      }
    }
    const toEl = toAddr ? asmRefs.current.get(toAddr) : null;
    if (!fromEl || !toEl) {
      setArcGeom(null);
      return;
    }
    setArcGeom({
      y1: fromEl.offsetTop + fromEl.offsetHeight / 2,
      y2: toEl.offsetTop + toEl.offsetHeight / 2,
    });
  }, [activeJump, data.lines, asmRefs]);
  const funcWeighted = useMemo(() => {
    const cc = funcTotals.cache_counts;
    return (
      (cc.L1 ?? 0) * CACHE_WEIGHTS.L1 +
      (cc.L2 ?? 0) * CACHE_WEIGHTS.L2 +
      (cc.L3 ?? 0) * CACHE_WEIGHTS.L3 +
      (cc.DRAM ?? 0) * CACHE_WEIGHTS.DRAM
    );
  }, [funcTotals]);

  // Index function offsets → addr for jump-to-target lookup.
  const offsetToAddr = useMemo(() => {
    const m = new Map<number, string>();
    for (const ln of data.lines) {
      if (ln.kind === "insn" && ln.offset != null && ln.addr) {
        m.set(ln.offset, ln.addr);
      }
    }
    return m;
  }, [data.lines]);

  // Search state — matches against `disasm` text and `addr`. Reset when
  // the user opens a different function (data.symbol changes).
  const [search, setSearch] = useState("");
  const [searchIdx, setSearchIdx] = useState(0);
  useEffect(() => {
    setSearch("");
    setSearchIdx(0);
  }, [data.symbol]);

  const matchAddrs = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [] as string[];
    const out: string[] = [];
    for (const ln of data.lines) {
      if (ln.kind !== "insn" || !ln.addr) continue;
      const hay = `${ln.addr} ${ln.disasm ?? ""}`.toLowerCase();
      if (hay.includes(q)) out.push(ln.addr);
    }
    return out;
  }, [search, data.lines]);

  useEffect(() => {
    if (searchIdx >= matchAddrs.length) setSearchIdx(0);
  }, [matchAddrs.length, searchIdx]);

  useEffect(() => {
    if (!matchAddrs.length) return;
    const target = matchAddrs[searchIdx] ?? matchAddrs[0];
    const el = asmRefs.current.get(target);
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [matchAddrs, searchIdx, asmRefs]);

  const matchAddrSet = useMemo(() => new Set(matchAddrs), [matchAddrs]);

  const stepSearch = (delta: number) => {
    if (!matchAddrs.length) return;
    const next = (searchIdx + delta + matchAddrs.length) % matchAddrs.length;
    setSearchIdx(next);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minWidth: 0,
        borderLeft: "1px solid oklch(0.92 0.005 250)",
      }}
    >
      <div
        ref={scrollRef}
        onClick={(e) => {
          // Click in empty space (not on a row): clear all selections.
          if (e.target === e.currentTarget) {
            onSelectInsn(null);
          }
        }}
        style={{
          flex: 1,
          overflow: "auto",
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontSize: 12,
        }}
      >
        {/* Sticky header inside the scroll container — see SourcePaneView
            for the rationale; in short, this is the only way to keep the
            header columns and body counter cells at the same x positions
            when a vertical scrollbar appears. */}
        <div
          style={{
            position: "sticky",
            top: 0,
            zIndex: 1,
            background: "white",
          }}
        >
          <HeaderRow
            leftLabel="addr"
            leftWidth={88}
            bodySlot={
              <SearchBar
                placeholder="disassembly"
                value={search}
                onChange={(v) => {
                  setSearch(v);
                  setSearchIdx(0);
                }}
                matchCount={matchAddrs.length}
                currentIdx={searchIdx}
                onStep={stepSearch}
              />
            }
            showCounters={true}
          />
        </div>
        <div ref={innerRef} style={{ position: "relative", minHeight: "100%" }}>
        {data.lines.map((ln, i) => {
          if (ln.kind === "function") {
            return (
              <div
                key={`fn-${i}`}
                style={{
                  padding: "8px 12px",
                  background: "oklch(0.97 0.005 250)",
                  borderTop: "1px solid oklch(0.93 0.005 250)",
                  borderBottom: "1px solid oklch(0.93 0.005 250)",
                  color: "oklch(0.45 0.06 280)",
                  fontWeight: 600,
                  fontSize: 11,
                }}
              >
                {ln.name}
              </div>
            );
          }
          if (ln.kind !== "insn" || ln.addr == null) return null;
          const isActive = activeAddr === ln.addr;
          // Dim only when there's a source-line selection AND this insn
          // doesn't map to it in the same file. Two files can share line
          // numbers, so we filter by `source_file` too.
          const dim =
            activeLine != null &&
            !isActive &&
            (ln.source_line !== activeLine ||
              (activeFile != null && ln.source_file !== activeFile));
          const isJumpTarget = ln.offset != null && jumpTargets.has(ln.offset);
          const cv = cyclesValue(ln, cyclesMode, funcTotals.cycles, totalProfileCycles);
          const isMatch = matchAddrSet.has(ln.addr);
          const isCurrentMatch = isMatch && matchAddrs[searchIdx] === ln.addr;
          // Search match wins over the cycle-intensity tint (so the user
          // can see the matches even on cold lines), but the active row
          // selection still wins over a match.
          const rowBg = isActive
            ? "oklch(0.95 0.05 250)"
            : isMatch
              ? MATCH_BG
              : dim
                ? "transparent"
                : intensityBg(cv.pct, 38);
          const isJump = ln.jump_target_offset != null;
          const jumpDir = isJump
            ? ln.jump_target_offset! > (ln.offset ?? 0)
              ? "↓"
              : "↑"
            : null;
          const targetAddr = isJump
            ? offsetToAddr.get(ln.jump_target_offset!)
            : null;
          const labelHex =
            isJump && ln.jump_target_offset != null
              ? ln.jump_target_offset.toString(16)
              : null;
          return (
            <div key={`a-${i}-${ln.addr}`}>
              {isJumpTarget && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    height: 16,
                    paddingLeft: 88,
                    color: "oklch(0.45 0.06 295)",
                    fontWeight: 600,
                    fontSize: 11,
                  }}
                >
                  {ln.offset!.toString(16)}:
                </div>
              )}
              <div
                ref={(el) => {
                  if (el && ln.addr) asmRefs.current.set(ln.addr, el);
                  else if (ln.addr) asmRefs.current.delete(ln.addr);
                }}
                onClick={() => {
                  if (activeAddr === ln.addr) {
                    onSelectInsn(null);
                  } else {
                    onSelectInsn(ln);
                  }
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  minHeight: 22,
                  background: rowBg,
                  // A search match should be visible even when its row is
                  // dimmed by a source-line selection, so override opacity
                  // when this row is one of the matches.
                  opacity: dim && !isMatch ? 0.32 : 1,
                  borderLeft: isCurrentMatch
                    ? "2px solid oklch(0.7 0.18 90)"
                    : isActive
                      ? "2px solid oklch(0.55 0.18 250)"
                      : "2px solid transparent",
                  cursor: "pointer",
                }}
              >
                <div
                  style={{
                    width: 88,
                    paddingLeft: 12,
                    color: "oklch(0.55 0.05 280)",
                    fontVariantNumeric: "tabular-nums",
                    fontSize: 11,
                  }}
                >
                  {ln.addr}
                </div>
                <pre
                  style={{
                    margin: 0,
                    paddingLeft: 12,
                    flex: 1,
                    // See note in SourcePaneView: `minWidth: 0` lets the
                    // flex item shrink past its content's intrinsic
                    // width so long disasm strings get ellipsized
                    // instead of shoving the counter columns off the
                    // right edge of the row.
                    minWidth: 0,
                    whiteSpace: "pre",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    color: "oklch(0.22 0.01 250)",
                  }}
                >
                  {jumpDir && (
                    <span
                      style={{
                        color: "oklch(0.45 0.18 250)",
                        marginRight: 4,
                      }}
                    >
                      {jumpDir}
                    </span>
                  )}
                  {renderAsm(ln.disasm ?? "", labelHex)}
                </pre>
                <div
                  style={{ width: COUNTER_COL_W, boxSizing: "border-box", padding: "2px 4px", borderLeft: "1px solid oklch(0.93 0.005 250)" }}
                >
                  <CounterCell
                    display={
                      cv.display + (cv.display && cyclesMode !== "raw" ? "%" : "")
                    }
                    pct={cv.pct}
                    hue={CATEGORY_META.cycles.hue}
                  />
                </div>
                {(["dram", "l1", "l2", "l3"] as const).map((cid) => {
                  const cat = CACHE_KEY[cid] as "L1" | "L2" | "L3" | "DRAM";
                  const v = ln.cache_counts?.[cat] ?? 0;
                  const mv = memoryValue(
                    v,
                    cat,
                    memoryMode,
                    funcTotals.cache_counts,
                    funcWeighted,
                  );
                  return (
                    <div
                      key={cid}
                      style={{ width: COUNTER_COL_W, boxSizing: "border-box", padding: "2px 4px", borderLeft: "1px solid oklch(0.93 0.005 250)" }}
                    >
                      <CounterCell
                        display={mv.display + (mv.display && memoryMode !== "raw" ? "%" : "")}
                        pct={mv.pct}
                        hue={CATEGORY_META[cid].hue}
                      />
                    </div>
                  );
                })}
                {/* Trailing actions slot — fixed width so the counter
                    columns line up vertically across rows whether or
                    not a row has a jump button. Click only consumes the
                    event when there's actually a jump (the empty slot
                    falls through to the row-level click handler). */}
                <div
                  style={{
                    width: ACTIONS_COL_W,
                    flexShrink: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {isJump && targetAddr && (
                    <button
                      title={`Jump to ${targetAddr}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onJumpTo(ln.jump_target_offset!);
                      }}
                      style={{
                        border: "1px solid oklch(0.9 0.005 250)",
                        background: "white",
                        borderRadius: 3,
                        padding: "1px 6px",
                        fontSize: 10,
                        color: "oklch(0.45 0.18 250)",
                        cursor: "pointer",
                        fontFamily: "ui-monospace, monospace",
                      }}
                    >
                      →
                    </button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
          {/* Pinned-jump arc — runs in the left gutter from source to
              target, with an arrowhead pointing at the target row. Drawn
              inside the inner wrapper so it scrolls with the rows. */}
          {arcGeom && (
            <svg
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                pointerEvents: "none",
                width: "100%",
                height: "100%",
                overflow: "visible",
              }}
            >
              <defs>
                <marker
                  id="jumpArrowHead"
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto"
                >
                  <path
                    d="M 0 0 L 10 5 L 0 10 z"
                    fill="oklch(0.65 0.14 240)"
                  />
                </marker>
              </defs>
              {/* horizontal stub at source */}
              <path
                d={`M 84 ${arcGeom.y1} L 14 ${arcGeom.y1} L 14 ${arcGeom.y2} L 80 ${arcGeom.y2}`}
                stroke="oklch(0.7 0.13 240)"
                strokeWidth="1.5"
                fill="none"
                markerEnd="url(#jumpArrowHead)"
              />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}

/// Render asm text with the destination label of a jump (if any) emphasised.
function renderAsm(text: string, labelHex: string | null) {
  if (!labelHex) return text;
  const idx = text.indexOf(labelHex);
  if (idx < 0) return text;
  return (
    <>
      {text.slice(0, idx)}
      <span style={{ color: "oklch(0.4 0.18 280)", fontWeight: 600 }}>
        {labelHex}
      </span>
      {text.slice(idx + labelHex.length)}
    </>
  );
}

// ── Top-level CodePanel ────────────────────────────────────────────────────

export function CodePanel({
  data,
  cyclesMode,
  memoryMode,
  totalProfileCycles,
  onAsmSelect,
}: Props) {
  const [activeAddr, setActiveAddr] = useState<string | null>(null);
  const [activeLine, setActiveLine] = useState<number | null>(null);
  // Currently-open source-file tab. Defaults to the heaviest pane (first
  // entry of `data.source_panes`) and follows the user when they click an
  // asm row whose `source_file` is a different file.
  const [activeFile, setActiveFile] = useState<string | null>(null);

  const asmRefs = useRef(new Map<string, HTMLDivElement>());
  const sourceRefs = useRef(new Map<number, HTMLDivElement>());

  /// Per-file map: file → (line → insns). The source pane uses only the
  /// entry for the file it's displaying, so one file's hits never bleed
  /// into another file's gutter.
  const asmByFileLine = useMemo(() => {
    const m = new Map<string, Map<number, AnnotateLine[]>>();
    for (const ln of data.lines) {
      if (ln.kind !== "insn") continue;
      if (ln.source_line == null || ln.source_file == null) continue;
      let inner = m.get(ln.source_file);
      if (!inner) {
        inner = new Map<number, AnnotateLine[]>();
        m.set(ln.source_file, inner);
      }
      const arr = inner.get(ln.source_line) ?? [];
      arr.push(ln);
      inner.set(ln.source_line, arr);
    }
    return m;
  }, [data]);

  const paneFiles = useMemo(
    () => new Set(data.source_panes.map((p) => p.file)),
    [data.source_panes],
  );

  const activePane = useMemo(
    () => data.source_panes.find((p) => p.file === activeFile) ?? null,
    [data.source_panes, activeFile],
  );

  const activeAsmByLine = useMemo(() => {
    if (!activeFile) return new Map<number, AnnotateLine[]>();
    return asmByFileLine.get(activeFile) ?? new Map();
  }, [asmByFileLine, activeFile]);

  const jumpTargets = useMemo(() => new Set(data.jump_targets), [data.jump_targets]);

  // Clicking an asm line: highlight that addr + its source line, scroll src.
  // If the insn's source file differs from the open tab, switch tabs and
  // scroll within the new pane on the next layout pass.
  const onSelectInsn = (insn: AnnotateLine | null) => {
    if (!insn || !insn.addr) {
      setActiveAddr(null);
      setActiveLine(null);
      onAsmSelect?.(null);
      return;
    }
    setActiveAddr(insn.addr);
    const wantFile = insn.source_file ?? null;
    const fileIsShown = wantFile != null && paneFiles.has(wantFile);
    if (insn.source_line != null && fileIsShown) {
      setActiveLine(insn.source_line);
      if (wantFile !== activeFile) {
        // Tab swap — refs for the target file haven't mounted yet; defer
        // the scroll until after the new pane renders.
        setActiveFile(wantFile);
        sourceRefs.current = new Map();
        const targetLine = insn.source_line;
        requestAnimationFrame(() => {
          const el = sourceRefs.current.get(targetLine);
          el?.scrollIntoView({ block: "center", behavior: "smooth" });
        });
      } else {
        const el = sourceRefs.current.get(insn.source_line);
        el?.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    } else {
      // Either no source mapping, or the mapping is to a file we don't
      // have a tab for (e.g. inlined stdlib). Clear the line highlight
      // so the asm pane doesn't dim itself against an unrelated line.
      setActiveLine(null);
    }
    onAsmSelect?.(insn);
  };

  // Clicking a source line: pick the first insn for that line, scroll asm.
  const onSelectLine = (line: number) => {
    setActiveLine(line);
    const hits = activeAsmByLine.get(line);
    if (hits && hits[0]?.addr) {
      setActiveAddr(hits[0].addr);
      const el = asmRefs.current.get(hits[0].addr);
      el?.scrollIntoView({ block: "center", behavior: "smooth" });
      // Also open the IBS-detail sidebar for the first asm row that
      // maps to this source line — saves the user a second click.
      onAsmSelect?.(hits[0]);
    } else {
      // Line has no asm mapping; close any previously-open sidebar so
      // it doesn't keep showing details for an unrelated instruction.
      onAsmSelect?.(null);
    }
  };

  // Jumping to a target offset: scroll the asm pane to the matching insn.
  // If that insn lives in another file, switch the source tab too so the
  // panes stay in sync.
  const onJumpTo = (targetOffset: number) => {
    for (const ln of data.lines) {
      if (ln.kind === "insn" && ln.offset === targetOffset && ln.addr) {
        setActiveAddr(ln.addr);
        if (ln.source_line != null) setActiveLine(ln.source_line);
        if (
          ln.source_file &&
          paneFiles.has(ln.source_file) &&
          ln.source_file !== activeFile
        ) {
          setActiveFile(ln.source_file);
          sourceRefs.current = new Map();
        }
        const el = asmRefs.current.get(ln.addr);
        el?.scrollIntoView({ block: "center", behavior: "smooth" });
        return;
      }
    }
  };

  // Switching tabs by hand. Clear refs for the *new* pane so the next
  // scroll lookup builds against the freshly mounted DOM. Also drop the
  // current row/line selection — the active row was tied to the old
  // file, so carrying it over would highlight an unrelated source line
  // in the new tab.
  const onSelectFile = (f: string) => {
    setActiveFile(f);
    sourceRefs.current = new Map();
    setActiveAddr(null);
    setActiveLine(null);
    onAsmSelect?.(null);
  };

  // Reset highlights + default to the heaviest source pane whenever the
  // user opens a different function.
  useEffect(() => {
    setActiveAddr(null);
    setActiveLine(null);
    onAsmSelect?.(null);
    setActiveFile(data.source_panes[0]?.file ?? null);
    sourceRefs.current = new Map();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.symbol]);

  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        minHeight: 0,
        background: "white",
      }}
    >
      {data.source_panes.length > 0 && activePane ? (
        <div
          style={{
            flex: 1,
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
          }}
        >
          {data.source_panes.length > 1 && (
            <SourceTabs
              panes={data.source_panes}
              active={activeFile}
              onSelect={onSelectFile}
            />
          )}
          <div style={{ flex: 1, minHeight: 0 }}>
            <SourcePaneView
              pane={activePane}
              asmByLine={activeAsmByLine}
              funcTotals={data.function_totals}
              cyclesMode={cyclesMode}
              memoryMode={memoryMode}
              totalProfileCycles={totalProfileCycles}
              activeLine={activeLine}
              setActiveLine={onSelectLine}
              sourceRefs={sourceRefs}
            />
          </div>
        </div>
      ) : (
        <div
          style={{
            flex: 1,
            minWidth: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "oklch(0.55 0.01 250)",
            fontStyle: "italic",
            fontSize: 12,
            borderRight: "1px solid oklch(0.92 0.005 250)",
            padding: 16,
            textAlign: "center",
          }}
        >
          No source-line debug info for this function. <br />
          Build with <code>debug = true</code> in the profiling profile so
          DWARF survives in the binary.
        </div>
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <AsmPaneView
          data={data}
          funcTotals={data.function_totals}
          cyclesMode={cyclesMode}
          memoryMode={memoryMode}
          totalProfileCycles={totalProfileCycles}
          activeAddr={activeAddr}
          activeLine={activeLine}
          activeFile={activeFile}
          jumpTargets={jumpTargets}
          asmRefs={asmRefs}
          onSelectInsn={onSelectInsn}
          onJumpTo={onJumpTo}
        />
      </div>
    </div>
  );
}

// Small segmented-toggle helper used by App for the two mode pickers.
export function ModeToggle<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T;
  onChange: (v: T) => void;
  options: Array<{ id: T; label: string }>;
  label: string;
}) {
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: 0.5,
          color: "oklch(0.5 0.01 250)",
        }}
      >
        {label}
      </span>
      <div
        style={{
          display: "inline-flex",
          borderRadius: 5,
          border: "1px solid oklch(0.9 0.005 250)",
          background: "oklch(0.985 0.003 250)",
          padding: 2,
          gap: 2,
        }}
      >
        {options.map((opt) => {
          const isSel = value === opt.id;
          return (
            <button
              key={opt.id}
              onClick={() => onChange(opt.id)}
              style={{
                border: "none",
                background: isSel ? "white" : "transparent",
                boxShadow: isSel
                  ? "0 1px 2px oklch(0.2 0.02 250 / 0.08)"
                  : "none",
                color: isSel ? "oklch(0.25 0.08 250)" : "oklch(0.5 0.01 250)",
                fontWeight: isSel ? 600 : 400,
                padding: "4px 10px",
                borderRadius: 3,
                fontSize: 11,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export type { CatId };
