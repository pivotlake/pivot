import { useEffect } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
} from "reactflow";
import type { Overview, Selection } from "../api";
import { nodeTypes, type IngestNodeData, type TableNodeData } from "./nodes";
import { edgeTypes, type FlowEdgeData } from "./FlowEdge";

interface Props {
  overview: Overview | null;
  onSelect: (selection: Selection) => void;
  selectedNodeId: string | null;
}

function buildGraph(overview: Overview): { nodes: Node[]; edges: Edge[] } {
  const rateByTable = new Map<string, number>();
  for (const t of overview.tables) rateByTable.set(t.name, t.rows_per_sec ?? 0);

  const nodes: Node[] = [];
  const edges: Edge[] = [];

  overview.ingests.forEach((ingest, i) => {
    const id = `ingest:${ingest.addr}`;
    const data: IngestNodeData = {
      kind: ingest.kind.toUpperCase(),
      addr: ingest.addr,
      totalRate: ingest.rows_per_sec,
      rows: ingest.rows,
      bytes: ingest.bytes,
      files: ingest.files,
    };
    nodes.push({
      id,
      type: "ingest",
      position: { x: 40, y: 90 + i * 220 },
      data,
    });

    for (const signal of ingest.signals) {
      const target = `table:${signal.table}`;
      const r = rateByTable.get(signal.table) ?? 0;
      const data: FlowEdgeData = { signal: signal.signal, rate: r };
      edges.push({
        id: `${id}->${target}:${signal.signal}`,
        source: id,
        target,
        type: "flow",
        data,
      });
    }
  });

  // Tables that receive ingest come first (top), in the order their receivers
  // reference them; the rest follow.
  const ingestTargets: string[] = [];
  for (const ingest of overview.ingests) {
    for (const s of ingest.signals) {
      if (!ingestTargets.includes(s.table)) ingestTargets.push(s.table);
    }
  }
  const ordered = [
    ...overview.tables.filter((t) => ingestTargets.includes(t.name)),
    ...overview.tables.filter((t) => !ingestTargets.includes(t.name)),
  ];

  ordered.forEach((t, j) => {
    const data: TableNodeData = {
      name: t.name,
      rowCount: t.row_count,
      rate: t.rows_per_sec,
      columns: t.columns,
      fileCount: t.file_count,
      compaction: t.compaction,
    };
    nodes.push({
      id: `table:${t.name}`,
      type: "table",
      position: { x: 520, y: 40 + j * 360 },
      data,
    });
  });

  return { nodes, edges };
}

export default function FlowGraph({ overview, onSelect, selectedNodeId }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Reconcile on each poll: refresh data + selection but keep any positions the
  // user has dragged a node to.
  useEffect(() => {
    if (!overview) return;
    const next = buildGraph(overview);
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]));
      return next.nodes.map((n) => ({
        ...n,
        position: byId.get(n.id)?.position ?? n.position,
        selected: n.id === selectedNodeId,
      }));
    });
    setEdges(next.edges);
  }, [overview, selectedNodeId, setNodes, setEdges]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onNodeClick={(_, node) => {
        if (node.type === "table") {
          onSelect({ type: "table", name: (node.data as TableNodeData).name });
        } else if (node.type === "ingest") {
          onSelect({ type: "ingest", addr: (node.data as IngestNodeData).addr });
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
