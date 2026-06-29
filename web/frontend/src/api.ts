// Types and fetchers for the pivotdb web backend (`/api/*`).

export interface ColumnInfo {
  name: string;
  col_type: string;
}

export interface FileInfo {
  path: string;
  size: number;
}

export interface TableOverview {
  name: string;
  location: string;
  columns: ColumnInfo[];
  file_count: number;
  total_bytes: number;
  partition_by: string[];
  sort_by: string[];
  row_count: number | null;
  rows_per_sec: number | null;
  compaction: Compaction | null;
}

export interface SignalInfo {
  signal: string;
  table: string;
}

export interface Ingest {
  kind: string;
  addr: string;
  flush_rows: number;
  flush_secs: number;
  signals: SignalInfo[];
  /** Cumulative rows ingested by this receiver. */
  rows: number;
  bytes: number;
  files: number;
  flushes: number;
  rows_per_sec: number;
  last_flush_unix_ms: number;
}

export interface Compaction {
  compactions: number;
  files_merged_in: number;
  files_written: number;
  bytes_written: number;
  last_run_unix_ms: number;
}

export interface DiskInfo {
  name: string;
  mount: string;
  total: number;
  available: number;
}

export interface SystemInfo {
  cpu_avg: number;
  cpu_per_core: number[];
  mem_used: number;
  mem_total: number;
  process_mem: number;
  process_cpu: number;
  disks: DiskInfo[];
}

/** What's selected in the flow graph and shown in the bottom drawer. */
export type Selection = { type: "table"; name: string } | { type: "ingest"; addr: string };

export interface Overview {
  store: string;
  tables: TableOverview[];
  ingests: Ingest[];
  compaction: Compaction | null;
  system: SystemInfo | null;
  connected: boolean;
  error?: string;
}

export interface QueryColumn {
  name: string;
  col_type: string;
}

export interface QueryResult {
  columns: QueryColumn[];
  rows: (string | null)[][];
  row_count: number;
  elapsed_ms: number;
  error?: string;
}

export async function fetchOverview(): Promise<Overview> {
  const res = await fetch("/api/overview");
  if (!res.ok) throw new Error(`overview request failed: ${res.status}`);
  return res.json();
}

export interface FilesPage {
  items: FileInfo[];
  total: number;
}

export async function fetchFiles(table: string, offset: number, limit: number): Promise<FilesPage> {
  const res = await fetch(
    `/api/tables/${encodeURIComponent(table)}/files?offset=${offset}&limit=${limit}`,
  );
  if (!res.ok) throw new Error(`files request failed: ${res.status}`);
  return res.json();
}

export interface RowGroupsPage {
  columns: QueryColumn[];
  rows: (string | null)[][];
  has_more: boolean;
}

export async function fetchRowGroups(
  table: string,
  offset: number,
  limit: number,
  file?: string,
): Promise<RowGroupsPage> {
  const fileParam = file == null ? "" : `&file=${encodeURIComponent(file)}`;
  const res = await fetch(
    `/api/tables/${encodeURIComponent(table)}/rowgroups?offset=${offset}&limit=${limit}${fileParam}`,
  );
  if (!res.ok) throw new Error(`rowgroups request failed: ${res.status}`);
  return res.json();
}

export async function runQuery(sql: string, signal?: AbortSignal): Promise<QueryResult> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql }),
    signal,
  });
  if (!res.ok) throw new Error(`query request failed: ${res.status}`);
  return res.json();
}
