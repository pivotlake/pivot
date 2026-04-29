export type DisplayMode = "weighted" | "percent" | "absolute";

export interface FunctionSummary {
  name: string;
  cycles: number;
  samples: number;
  weighted_cost: number;
  cycles_pct: number;
}

export interface Summary {
  perf_data: string;
  binary: string;
  total_unweighted: number;
  total_weighted: number;
  total_cycles: number;
  skipped_lines: string[];
  cache_summary: { per_level: Record<string, number> };
  pf_summary: {
    dispatched: number;
    fills: number;
    ineffective_dc: number;
    ineffective_mab: number;
  };
  functions: FunctionSummary[];
}

export type GutterKind = "lane" | "arm" | "longarrow" | "blank";
export interface GutterCell {
  glyph: string;
  kind: GutterKind;
}

export interface IBSRaw {
  sample_count: number;
  avg_dc_miss_lat: number;
  avg_tlb_refill_lat: number;
  avg_comp_to_ret: number;
  avg_tag_to_ret: number;
  avg_mabs: number;
  mabs_max: number;
  dc_miss_count: number;
  l2_miss_count: number;
  dc_l1_tlb_miss_count: number;
  dc_l2_tlb_miss_count: number;
  brn_ret_count: number;
  brn_misp_count: number;
  brn_taken_count: number;
  brn_return_count: number;
  brn_fuse_count: number;
}

export interface InsnStats {
  cache_counts: Record<string, number>;
  tlb_counts: Record<string, number>;
  op_counts: Record<string, number>;
  snoop_counts: Record<string, number>;
  locked_count: number;
  total_samples: number;
  cycles: number;
  ibs: IBSRaw;
}

export interface InstructionLine {
  type: "instruction";
  addr: string;
  offset: number;
  mnem: string;
  operands: string;
  is_target: boolean;
  gutter: GutterCell[];
  cache_counts: Record<string, number>;
  cache_weighted_pct: Record<string, number>;
  cycles: number;
  cycles_pct: number;
  samples: number;
  stats: InsnStats;
}

export interface SourceLine {
  type: "source";
  text: string;
  gutter: GutterCell[];
}

export interface FuncHeaderLine {
  type: "function_header";
  name: string;
}

export interface SeparatorLine {
  type: "separator";
}

export type Line = InstructionLine | SourceLine | FuncHeaderLine | SeparatorLine;

export interface FuncData {
  name: string;
  lines: Line[];
  max_lanes: number;
  addr_width: number;
  totals: { cycles: number; unweighted: number; weighted: number };
}
