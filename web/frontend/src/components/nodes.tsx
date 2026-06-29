import { Handle, Position, type NodeProps } from "reactflow";
import type { ColumnInfo, Compaction } from "../api";
import { ago, bytes, count } from "../format";

export interface IngestNodeData {
  kind: string;
  addr: string;
  totalRate: number;
  rows: number;
  bytes: number;
  files: number;
}

export interface TableNodeData {
  name: string;
  rowCount: number | null;
  rate: number | null;
  columns: ColumnInfo[];
  fileCount: number;
  compaction: Compaction | null;
}

const MAX_COLS = 7;

// A "broadcasting telemetry" glyph for the receiver - a signal source emitting
// outward, evoking OpenTelemetry's collector. Teal when data is flowing.
function TelemetryIcon({ live }: { live: boolean }) {
  return (
    <svg
      className={`tel-icon ${live ? "live" : ""}`}
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    >
      <circle cx="12" cy="12" r="2.2" fill="currentColor" stroke="none" />
      <path d="M8 8a5.5 5.5 0 0 0 0 8" />
      <path d="M16 8a5.5 5.5 0 0 1 0 8" />
      <path d="M5 5a9.5 9.5 0 0 0 0 14" />
      <path d="M19 5a9.5 9.5 0 0 1 0 14" />
    </svg>
  );
}

export function IngestNode({ data, selected }: NodeProps<IngestNodeData>) {
  const live = data.totalRate > 0;
  return (
    <div className={`node ${selected ? "selected" : ""}`}>
      <div className="nhead">
        <div className="kind">
          <TelemetryIcon live={live} />
          {data.kind} receiver
        </div>
        <div className="title">{data.addr}</div>
      </div>
      <div className="nbody">
        <div className="big tnum">{count(data.rows)}</div>
        <div className="sub">
          rows received · {bytes(data.bytes)} · {data.files} files
        </div>
      </div>
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

export function TableNode({ data, selected }: NodeProps<TableNodeData>) {
  const live = (data.rate ?? 0) > 0;
  const c = data.compaction;
  const shown = data.columns.slice(0, MAX_COLS);
  return (
    <div className={`node table-node ${selected ? "selected" : ""}`}>
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <div className="nhead">
        <div className="kind">
          <span className={`pip ${live ? "live" : ""}`} />
          table
        </div>
        <div className="title-row">
          <span className="title">{data.name}</span>
          <span className="rowcount tnum">{count(data.rowCount)}</span>
        </div>
        <div className="sub">
          {data.columns.length} columns · {data.fileCount} files
        </div>
      </div>

      <div className="schema">
        {shown.map((col) => (
          <div className="col" key={col.name}>
            <span className="cname">{col.name}</span>
            <span className="ctype">{col.col_type}</span>
          </div>
        ))}
        {data.columns.length > MAX_COLS && (
          <div className="col more">+{data.columns.length - MAX_COLS} more</div>
        )}
      </div>

      <div className="nfoot">
        {c && c.compactions > 0 ? (
          <>
            <span className="comp">
              <span className="glyph">⟲</span> {count(c.compactions)} merges ·{" "}
              {count(c.files_merged_in)}→{count(c.files_written)} files
            </span>
            <span className="when">{ago(c.last_run_unix_ms)}</span>
          </>
        ) : (
          <span className="comp idle">no compactions yet</span>
        )}
      </div>
    </div>
  );
}

export const nodeTypes = { ingest: IngestNode, table: TableNode };
