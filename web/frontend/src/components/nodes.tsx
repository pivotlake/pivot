import { Handle, Position, type NodeProps } from "reactflow";
import type { ColumnInfo, Compaction } from "../api";
import { ago, count } from "../format";

export interface TableNodeData {
  name: string;
  rowCount: number | null;
  rate: number | null;
  columns: ColumnInfo[];
  fileCount: number;
  compaction: Compaction | null;
}

const MAX_COLS = 7;

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

export const nodeTypes = { table: TableNode };
