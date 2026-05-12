export type Category = "cycles" | "l1" | "l2" | "l3" | "dram";

export interface Meta {
  perf_data: string;
  binary: string | null;
  cpus: number[];
  categories: Category[];
  time_start_ns: number;
  time_end_ns: number;
  duration_ns: number;
  stats: {
    total_samples: number;
    by_category: Record<string, number>;
    by_cpu: Record<string, Record<string, number>>;
  };
}

export interface Track {
  id: string;
  label: string;
  category: Category;
  cpu: number | null;
  counts: number[];
  peak: number;
}

export interface TracksResponse {
  buckets: number;
  consolidated: boolean;
  time_start_ns: number;
  time_end_ns: number;
  tracks: Track[];
}

export interface FlameNode {
  id: number;
  parent: number;
  label: string;
  self: number;
  total: number;
  depth: number;
}

export interface FlameResponse {
  total: number;
  nodes: FlameNode[];
}

export interface AnnotateLine {
  kind: "function" | "sep" | "source" | "insn";
  name?: string;
  text?: string;
  addr?: string;
  offset?: number;
  disasm?: string;
  cycles?: number;
  cache_counts?: Record<string, number>;
  samples?: number;
  source_line?: number;
  source_file?: string;
  jump_target_offset?: number;
}

export interface SourcePaneLine {
  line: number;
  text: string;
}

export interface SourcePane {
  file: string;
  start_line: number;
  end_line: number;
  lines: SourcePaneLine[];
}

export interface FunctionTotals {
  cycles: number;
  samples: number;
  cache_counts: Record<string, number>;
}

export interface AnnotateResponse {
  symbol: string;
  binary: string | null;
  totals: { samples: number; weight: number; cycles: number };
  function_totals: FunctionTotals;
  lines: AnnotateLine[];
  /// One pane per referenced source file, sorted heaviest-first.
  source_panes: SourcePane[];
  jump_targets: number[];
}

/// Cache-level cycle-cost weights (Zen 4). Used by the "Weighted" memory
/// mode to make all cache levels comparable.
export const CACHE_WEIGHTS: Record<string, number> = {
  L1: 4,
  L2: 14,
  L3: 50,
  DRAM: 250,
};

// ── Per-instruction detail (the IBS deep view) ───────────────────────────────

export interface IBSRaw {
  dc_miss_lat_sum: number;
  dc_miss_lat_count: number;
  tlb_refill_lat_sum: number;
  tlb_refill_lat_count: number;
  comp_to_ret_sum: number;
  comp_to_ret_count: number;
  tag_to_ret_sum: number;
  tag_to_ret_count: number;
  mabs_sum: number;
  mabs_count: number;
  mabs_max: number;
  mabs_max_count: number;
  sw_pf_count: number;
  misaligned_count: number;
  dc_miss_no_mab_count: number;
  dc_l1_tlb_miss_count: number;
  dc_l2_tlb_miss_count: number;
  dc_miss_count: number;
  l2_miss_count: number;
  mem_op_count: number;
  mem_width_counts: Record<string, number>;
  brn_ret_count: number;
  brn_misp_count: number;
  brn_taken_count: number;
  brn_return_count: number;
  brn_fuse_count: number;
  sample_count: number;
}

export interface PrefetchCounts {
  dispatched: number;
  ineffective_dc_hit: number;
  ineffective_mab_match: number;
  fills: number;
}

export interface InsnStats {
  /// "L1" | "LFB" | "L2" | "L3" | "DRAM" | "REM" | "N-M" → count
  cache_counts: Record<string, number>;
  /// "L1Hit" | "L2Hit" | "Miss" | "NA" → count
  tlb_counts: Record<string, number>;
  /// "Load" | "Store" | "NA" → count
  op_counts: Record<string, number>;
  /// "NA" | "None" | "Hit" | "HitM" | "Miss" → count
  snoop_counts: Record<string, number>;
  locked_count: number;
  total_weight: number;
  weight_count: number;
  total_samples: number;
  cycles: number;
  ibs: IBSRaw;
  prefetch: PrefetchCounts;
  token_stalls: Record<string, number>;
}

// ── perf stat tracks (memory throughput / latency) ─────────────────────────

export interface StatResponse {
  /// `false` if no perf.stat.data was found alongside the perf.data —
  /// the frontend then hides the memory-throughput / memory-latency rows.
  available: boolean;
  buckets: number;
  /// Bytes/second per bucket.
  throughput_bps: number[];
  /// Core clocks per L3 miss per bucket — already has `latency_base`
  /// subtracted, so values are deltas from the configured baseline
  /// (positive = above base, negative = below). Add `latency_base`
  /// back to get the absolute latency.
  latency_clocks: number[];
  throughput_peak: number;
  /// Peak of the *delta* series. May be negative if every observed
  /// latency was below the configured `--memory-base-latency`.
  latency_peak: number;
  /// Configured baseline latency in core clocks (set via
  /// `--memory-base-latency` on the server). When 0, the latency
  /// series is absolute.
  latency_base: number;
  duration_s: number;
}

// ── Pipeline Top-Down summary ───────────────────────────────────────────────

export interface PipelineSummary {
  /// `true` only when every required metric was computable. When
  /// `false`, `error` carries a human-readable message and the
  /// individual metric values may be `NaN` (rendered as "—").
  available: boolean;
  /// Optional error banner. Surfaced in the Summary tab whenever
  /// the backend couldn't compute one or more metrics (e.g. the
  /// recording lacked explicit event grouping).
  error?: string;
  /// Top-Down L1 (sums to ~100% when all inputs were measured).
  retiring_pct: number;
  frontend_bound_pct: number;
  backend_bound_pct: number;
  bad_speculation_pct: number;
  /// L2 sub-breakdowns. Each parent splits into two children that
  /// sum back to the parent (frontend: latency+bandwidth, backend:
  /// memory+core, retiring: fastpath+microcode).
  frontend_latency_pct: number;
  frontend_bandwidth_pct: number;
  backend_memory_share: number;
  backend_memory_pct: number;
  backend_core_pct: number;
  retiring_fastpath_pct: number;
  retiring_microcode_pct: number;
  /// Single-number diagnostics.
  ipc: number;
  branch_misp_pct: number;
  microcode_pct: number;
  resync_pct: number;
  /// Raw totals.
  total_cycles: number;
  total_ops_retired: number;
  total_ops_dispatched: number;
  total_branch_misp: number;
}

export const apiInsnDetail = (symbol: string, offset: number) =>
  getJSON<InsnStats>(
    `/api/insn_detail?function=${encodeURIComponent(symbol)}&offset=${offset}`,
  );

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${path}: ${r.status} ${txt}`);
  }
  return r.json();
}

export const api = {
  meta: () => getJSON<Meta>("/api/meta"),
  tracks: (categories: Category[], consolidated: boolean, buckets: number) =>
    getJSON<TracksResponse>(
      `/api/tracks?categories=${categories.join(",")}` +
        `&consolidated=${consolidated}` +
        `&buckets=${buckets}`,
    ),
  flamegraph: (
    trackId: string,
    timeRange?: { lo_ns: number; hi_ns: number },
  ) => {
    let url = `/api/flamegraph?track=${encodeURIComponent(trackId)}`;
    if (timeRange) {
      url += `&time_lo_ns=${timeRange.lo_ns}&time_hi_ns=${timeRange.hi_ns}`;
    }
    return getJSON<FlameResponse>(url);
  },
  annotate: (symbol: string) =>
    getJSON<AnnotateResponse>(`/api/annotate?function=${encodeURIComponent(symbol)}`),
  stat: (buckets: number) =>
    getJSON<StatResponse>(`/api/stat?buckets=${buckets}`),
  pipelineSummary: (timeRange?: { lo_ns: number; hi_ns: number }) => {
    let url = "/api/pipeline_summary";
    if (timeRange) {
      url += `?time_lo_ns=${timeRange.lo_ns}&time_hi_ns=${timeRange.hi_ns}`;
    }
    return getJSON<PipelineSummary>(url);
  },
};

// ── Category metadata ────────────────────────────────────────────────────────

export const CATEGORY_META: Record<
  Category,
  { label: string; short: string; hue: number }
> = {
  cycles: { label: "Cycles", short: "CYC",  hue: 38 },
  // L1/L2/L3 = "load was satisfied at this cache level" (i.e. hit
  // here after missing the levels above); DRAM = missed all caches
  // and went to memory.
  dram:   { label: "DRAM",   short: "DRAM", hue: 12 },
  l1:     { label: "L1 hit", short: "L1",   hue: 220 },
  l2:     { label: "L2 hit", short: "L2",   hue: 260 },
  l3:     { label: "L3 hit", short: "L3",   hue: 295 },
};

export const CATEGORY_ORDER: Category[] = ["cycles", "dram", "l1", "l2", "l3"];
