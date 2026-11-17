import { useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  ReactFlow,
  useNodesState,
  type ReactFlowInstance,
} from "@xyflow/react";
import { ClientNode } from "./components/ClientNode";
import { ConnectionEdge } from "./components/ConnectionEdge";
import { ServerNode } from "./components/ServerNode";
import type { ClientFlowNode, ConnectionFlowEdge, FlowNode, ServerFlowNode } from "./flowTypes";
import { chooseLayout, GRAPH_BOUNDS, placeSlot, type GraphLayout } from "./graphLayout";
import {
  fetchConnections,
  measureCopyRates,
  type ConnectionsReading,
  type ServerConnection,
} from "./serverConnections";
import { serverHost } from "./serverHost";
import { fetchMemory, type MemoryReading } from "./serverMemory";
import { CompactionSimulation, type Compaction } from "./simulation";

const CONNECTIONS_POLL_MS = 1000;
const SIMULATION_STEP_MS = 250;

const SERVER_NODE_ID = "server";

// How much of the view the graph spans, each way, when the view is fitted.
// The rest is open canvas around it, to drag clients out into.
const GRAPH_SHARE_OF_VIEW = 0.5;

// Room around the fitted area, with more at the top for the page title.
const FIT_PADDING = { top: "72px", bottom: "24px", x: "16px" } as const;

const nodeTypes = { server: ServerNode, client: ClientNode };
const edgeTypes = { connection: ConnectionEdge };

// What the page knows of the server: nothing yet, the latest reading of its
// connections and memory, or why the latest reading failed. A failure shows
// nothing rather than what was last read, which may no longer be true.
type ServerView =
  | { status: "loading" }
  | {
      status: "live";
      reading: ConnectionsReading;
      copyRates: Map<number, number>;
      memory: MemoryReading;
    }
  | { status: "failed"; reason: string };

// Brings the graph's nodes in line with what was read of the server. A node
// already on the graph keeps its object, and with it where the user dragged
// it and the size React Flow measured; a new client takes the first free
// slot. A change of layout puts every client back in its slot.
function syncNodes(
  current: FlowNode[],
  view: ServerView,
  compaction: Compaction,
  layout: GraphLayout,
): FlowNode[] {
  const connections: ServerConnection[] = view.status === "live" ? view.reading.connections : [];
  const readAtMs = view.status === "live" ? view.reading.readAtMs : 0;

  const serverData: ServerFlowNode["data"] = {
    host: serverHost,
    reading:
      view.status === "live"
        ? {
            pivotAddress: view.reading.pivotAddress,
            clientCount: connections.length,
            queriesRunning: connections.filter((connection) => connection.state === "query").length,
            memoryUsage: view.memory.usage,
          }
        : undefined,
    compaction,
  };
  const placedServer = current.find((node): node is ServerFlowNode => node.type === "server");
  const serverNode: ServerFlowNode = placedServer
    ? { ...placedServer, data: serverData }
    : { id: SERVER_NODE_ID, type: "server", position: { x: 0, y: 0 }, data: serverData };

  const connectedNodeIds = new Set(connections.map((connection) => `connection-${connection.id}`));
  const placedClients = new Map<string, ClientFlowNode>();
  const takenSlots = new Set<number>();
  for (const node of current) {
    if (node.type === "client" && connectedNodeIds.has(node.id)) {
      placedClients.set(node.id, node);
      takenSlots.add(node.data.slot);
    }
  }

  const clientNodes = connections.map((connection): ClientFlowNode => {
    const nodeId = `connection-${connection.id}`;
    const placed = placedClients.get(nodeId);
    if (placed) {
      const position =
        placed.data.layout === layout ? placed.position : placeSlot(layout, placed.data.slot).position;
      return { ...placed, position, data: { ...placed.data, connection, readAtMs, layout } };
    }
    let slot = 0;
    while (takenSlots.has(slot)) slot += 1;
    takenSlots.add(slot);
    return {
      id: nodeId,
      type: "client",
      position: placeSlot(layout, slot).position,
      data: { connection, readAtMs, slot, layout },
    };
  });

  return [serverNode, ...clientNodes];
}

export function App() {
  // Reads the server again a moment after each reading comes back, so a slow
  // read never has another queue up behind it.
  const [view, setView] = useState<ServerView>({ status: "loading" });
  useEffect(() => {
    let isStopped = false;
    let nextPoll: number | undefined;
    let previousReading: ConnectionsReading | undefined;
    const poll = async () => {
      try {
        const [reading, memory] = await Promise.all([fetchConnections(), fetchMemory()]);
        if (isStopped) return;
        const copyRates = previousReading
          ? measureCopyRates(previousReading, reading)
          : new Map<number, number>();
        previousReading = reading;
        setView({ status: "live", reading, copyRates, memory });
      } catch (error) {
        if (isStopped) return;
        previousReading = undefined;
        setView({ status: "failed", reason: error instanceof Error ? error.message : String(error) });
      }
      nextPoll = window.setTimeout(poll, CONNECTIONS_POLL_MS);
    };
    poll();
    return () => {
      isStopped = true;
      window.clearTimeout(nextPoll);
    };
  }, []);

  const [simulation] = useState(() => new CompactionSimulation(Date.now()));
  const [compaction, setCompaction] = useState(() => simulation.advance(Date.now()));
  useEffect(() => {
    const timer = setInterval(() => setCompaction(simulation.advance(Date.now())), SIMULATION_STEP_MS);
    return () => clearInterval(timer);
  }, [simulation]);

  const [viewportSize, setViewportSize] = useState(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }));
  useEffect(() => {
    const measureViewport = () =>
      setViewportSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener("resize", measureViewport);
    return () => window.removeEventListener("resize", measureViewport);
  }, []);
  const layout = chooseLayout(viewportSize.width, viewportSize.height);

  // Fits the view once React Flow is ready and again whenever the window or
  // the layout changes.
  const [flow, setFlow] = useState<ReactFlowInstance<FlowNode, ConnectionFlowEdge>>();
  useEffect(() => {
    // The graph's bounds are centred on the origin, so dividing them widens
    // the fitted area evenly around the graph.
    const graph = GRAPH_BOUNDS[layout];
    const fittedArea = {
      x: graph.x / GRAPH_SHARE_OF_VIEW,
      y: graph.y / GRAPH_SHARE_OF_VIEW,
      width: graph.width / GRAPH_SHARE_OF_VIEW,
      height: graph.height / GRAPH_SHARE_OF_VIEW,
    };
    flow?.fitBounds(fittedArea, { padding: FIT_PADDING });
  }, [flow, layout, viewportSize]);

  const [nodes, setNodes, onNodesChange] = useNodesState<FlowNode>([]);
  useEffect(() => {
    setNodes((current) => syncNodes(current, view, compaction, layout));
  }, [view, compaction, layout, setNodes]);

  const edges = useMemo(
    () =>
      nodes.flatMap((node): ConnectionFlowEdge[] =>
        node.type === "client"
          ? [
              {
                id: `${node.id}-to-server`,
                type: "connection",
                source: node.id,
                target: SERVER_NODE_ID,
                targetHandle: placeSlot(node.data.layout, node.data.slot).serverHandle,
                data: {
                  state: node.data.connection.state,
                  copyBytesPerSecond:
                    view.status === "live" ? view.copyRates.get(node.data.connection.id) : undefined,
                },
              },
            ]
          : [],
      ),
    [nodes, view],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onInit={setFlow}
      nodeOrigin={[0.5, 0.5]}
      minZoom={0.2}
      maxZoom={2}
      nodesConnectable={false}
      elementsSelectable={false}
      nodesFocusable={false}
      edgesFocusable={false}
      deleteKeyCode={null}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.5} />
      <Controls showInteractive={false} />

      <Panel position="top-left" className="title-panel">
        <div className="wordmark">
          pivot
          <span className="wordmark-cursor" aria-hidden="true" />
        </div>
        {view.status === "loading" && (
          <div className="connections-notice">Reading the server…</div>
        )}
        {view.status === "live" && (
          <div className="connections-notice">
            Live connections and memory of {view.reading.pivotAddress}. Compaction is simulated.
          </div>
        )}
        {view.status === "failed" && (
          <div className="connections-notice is-failed" role="alert">
            Cannot read the server: {view.reason}
          </div>
        )}
      </Panel>
    </ReactFlow>
  );
}
