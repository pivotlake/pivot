import { useEffect } from "react";
import { Handle, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import type { WebBackendFlowNode } from "../flowTypes";
import { placeWebBackend } from "../graphLayout";

// The web backend's own connection to the server: the one this page is read
// through. It is drawn as a fixture of the page, not as one of the server's
// clients: it says what it is and nothing of what it is doing, since what it
// is doing is always the query that draws the page.
export function WebBackendNode({ id, data }: NodeProps<WebBackendFlowNode>) {
  const { connection, order, layout } = data;

  // React Flow measures where a handle is once, so it has to be told when a
  // change of layout moves the handle to another side of the card.
  const { clientHandle } = placeWebBackend(layout, order);
  const updateNodeInternals = useUpdateNodeInternals();
  useEffect(() => {
    updateNodeInternals(id);
  }, [id, clientHandle, updateNodeInternals]);

  return (
    <div className="web-backend-card">
      <Handle type="source" position={clientHandle} isConnectable={false} />
      <div className="web-backend-name">{connection.applicationName}</div>
      <div className="web-backend-role">this page's connection</div>
    </div>
  );
}
