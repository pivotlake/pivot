import { useEffect, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import type { Overview, Selection } from "../api";
import FlowGraph from "./FlowGraph";
import SystemMetrics from "./SystemMetrics";
import Compactor from "./Compactor";
import TableDetail from "./TableDetail";

interface Props {
  overview: Overview | null;
  reachable: boolean;
}

export default function OverviewTab({ overview, reachable }: Props) {
  const [selected, setSelected] = useState<Selection | null>(null);

  // Default to the first table once, when data first arrives.
  useEffect(() => {
    if (!selected && overview && overview.tables.length > 0) {
      setSelected({ type: "table", name: overview.tables[0].name });
    }
  }, [overview, selected]);

  if (!reachable) {
    return (
      <div className="empty">
        <div>Waiting for the engine</div>
        <div className="sub">
          start pivotdb-server with <code>--http-bind</code>, then open this page on that port
        </div>
      </div>
    );
  }

  const selectedNodeId = selected?.type === "table" ? `table:${selected.name}` : null;

  const drawer = (
    <TableDetail
      table={
        selected?.type === "table"
          ? (overview?.tables.find((t) => t.name === selected.name) ?? null)
          : null
      }
      store={overview?.store ?? ""}
    />
  );

  return (
    <div className="overview">
      <div className="flow-pane">
        <div className="pane-head">
          <h2>Tables</h2>
          <p>The catalog's tables and their live row counts, updating live.</p>
        </div>

        {overview && overview.tables.length === 0 && (
          <div className="flow-hint">
            No tables yet. Connect with psql, <code>CREATE TABLE</code> and <code>INSERT</code> to
            see data land here.
          </div>
        )}

        <div className="dock">
          <SystemMetrics system={overview?.system ?? null} />
          <Compactor compaction={overview?.compaction ?? null} />
        </div>

        <ReactFlowProvider>
          <FlowGraph overview={overview} onSelect={setSelected} selectedNodeId={selectedNodeId} />
        </ReactFlowProvider>
      </div>

      <div className="drawer">{drawer}</div>
    </div>
  );
}
