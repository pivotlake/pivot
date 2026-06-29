import { useEffect, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import type { Overview, Selection } from "../api";
import FlowGraph from "./FlowGraph";
import SystemMetrics from "./SystemMetrics";
import Compactor from "./Compactor";
import TableDetail from "./TableDetail";
import ReceiverDetail from "./ReceiverDetail";

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

  const selectedNodeId =
    selected?.type === "table"
      ? `table:${selected.name}`
      : selected?.type === "ingest"
        ? `ingest:${selected.addr}`
        : null;

  const drawer =
    selected?.type === "ingest" ? (
      <ReceiverDetail
        ingest={overview?.ingests.find((i) => i.addr === selected.addr) ?? null}
        tables={overview?.tables ?? []}
      />
    ) : (
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
          <h2>Data flow</h2>
          <p>Ingest receivers and the tables they feed, updating live.</p>
        </div>

        {overview && overview.ingests.length === 0 && (
          <div className="flow-hint">
            No ingest receivers configured. Start the server with{" "}
            <code>--otel 'addr=127.0.0.1:4317,logs'</code> to see live ingest flow into a table.
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
