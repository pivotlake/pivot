import { useEffect } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  useNodesState,
  type Node,
} from "reactflow";
import type { Overview, Selection } from "../api";
import { nodeTypes, type TableNodeData } from "./nodes";

interface Props {
  overview: Overview | null;
  onSelect: (selection: Selection) => void;
  selectedNodeId: string | null;
}

function buildNodes(overview: Overview): Node[] {
  return overview.tables.map((t, j) => {
    const data: TableNodeData = {
      name: t.name,
      rowCount: t.row_count,
      rate: t.rows_per_sec,
      columns: t.columns,
      fileCount: t.file_count,
      compaction: t.compaction,
    };
    return {
      id: `table:${t.name}`,
      type: "table",
      position: { x: 520, y: 40 + j * 360 },
      data,
    };
  });
}

export default function FlowGraph({ overview, onSelect, selectedNodeId }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);

  // Reconcile on each poll: refresh data + selection but keep any positions the
  // user has dragged a node to.
  useEffect(() => {
    if (!overview) return;
    const next = buildNodes(overview);
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]));
      return next.map((n) => ({
        ...n,
        position: byId.get(n.id)?.position ?? n.position,
        selected: n.id === selectedNodeId,
      }));
    });
  }, [overview, selectedNodeId, setNodes]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={[]}
      nodeTypes={nodeTypes}
      onNodesChange={onNodesChange}
      onNodeClick={(_, node) => {
        if (node.type === "table") {
          onSelect({ type: "table", name: (node.data as TableNodeData).name });
        }
      }}
      nodesConnectable={false}
      edgesFocusable={false}
      fitView
      fitViewOptions={{ padding: 0.3, maxZoom: 0.85 }}
      minZoom={0.3}
      maxZoom={1.5}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color="#d7dae3" />
      <Controls showInteractive={false} position="bottom-left" />
    </ReactFlow>
  );
}
