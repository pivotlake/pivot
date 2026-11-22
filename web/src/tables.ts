// The pivot server's tables and their data files, as the web backend reads
// them from the server's `system.tables` and `system.table_files`.

import { readBackend } from "./backend";

export type TableSummary = {
  datastore: string;
  schema: string;
  name: string;
  // The table's `system.tables.id`.
  id: string;
  // The table's sort columns, in order.
  sortingKeys: string[];
  // The table's partition columns, in order.
  partitionKey: string[];
  totalRows: number;
  bytes: number;
  bytesUncompressed: number;
  fileCount: number;
};

// A column's smallest or largest value in a file: a number or a boolean
// where the column's type is one, and text for anything else, dates and
// timestamps included.
export type Bound = number | string | boolean;

export type TableFile = {
  path: string;
  // `column=value` pairs, or empty for an unpartitioned table.
  partition: string;
  bytes: number;
  bytesUncompressed: number;
  // The file's bounds on every column that records them.
  bounds: Record<string, { min: Bound; max: Bound }>;
};

// The sizes the server's compacter selects merges by. The server does not
// report them, so the backend is told them when it starts.
export type CompactionSettings = {
  // The size a merged file is cut at. A file below half of it is small.
  targetBytes: number;
  // The total a pile of small files merges at.
  mergeTargetBytes: number;
  // The count at which a pile below that total may merge anyway.
  minFiles: number;
  // How many mutually overlapping files one layout merge rewrites.
  layoutCliqueSize: number;
};

export type TableFilesReading = {
  files: TableFile[];
  compaction: CompactionSettings;
};

export function fetchTables(): Promise<{ tables: TableSummary[] }> {
  return readBackend("/api/tables");
}

export function fetchTableFiles(tableId: string): Promise<TableFilesReading> {
  return readBackend(`/api/tables/${tableId}/files`);
}
