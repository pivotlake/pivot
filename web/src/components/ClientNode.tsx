import { useEffect } from "react";
import { Handle, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import type { ClientFlowNode } from "../flowTypes";
import { formatBytes, formatDuration } from "../format";
import { placeSlot } from "../graphLayout";
import type { ServerConnection } from "../serverConnections";

function describeActivity(
  connection: ServerConnection,
  readAtMs: number,
): { headline: string; detail: string } {
  // The server's clock stamped the instants and the backend's clock the
  // reading, so a small skew between them must not show as a negative time.
  const inStateFor = formatDuration(Math.max(0, readAtMs - connection.stateChangedAtMs));
  switch (connection.state) {
    case "idle":
      return {
        headline: "idle",
        detail: `connected ${formatDuration(Math.max(0, readAtMs - connection.connectedAtMs))}`,
      };
    case "query":
      return { headline: `running query · ${inStateFor}`, detail: connection.statement };
    case "copy":
      return { headline: `COPY · ${formatBytes(connection.copyBytes)}`, detail: connection.statement };
  }
}

export function ClientNode({ id, data }: NodeProps<ClientFlowNode>) {
  const { connection, readAtMs, slot, layout } = data;
  const { headline, detail } = describeActivity(connection, readAtMs);

  // React Flow measures where a handle is once, so it has to be told when a
  // change of layout moves the handle to another side of the card.
  const { clientHandle } = placeSlot(layout, slot);
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, clientHandle, updateNodeInternals]);

  return (
    <div className={`client-card is-${connection.state}`}>
      <Handle type="source" position={clientHandle} isConnectable={false} />

      <div className="client-heading">
        <span className={connection.applicationName ? "client-application" : "client-application is-unnamed"}>
          {connection.applicationName || "unnamed client"}
        </span>
        <span className="client-id">#{connection.id}</span>
      </div>
      <div className="client-origin">
        {connection.userName}@{connection.clientAddress}
      </div>

      <div className="client-activity">
        <span className="activity-indicator" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
        <span className="activity-headline">{headline}</span>
      </div>
      <div className="client-detail" title={detail}>
        {detail}
      </div>
    </div>
  );
}
