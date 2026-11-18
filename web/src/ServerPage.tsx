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
import { WebBackendNode } from "./components/WebBackendNode";
import type {
  ClientFlowNode,
  ConnectionFlowEdge,
  FlowNode,
  ServerFlowNode,
  WebBackendFlowNode,
} from "./flowTypes";
import {
  chooseLayout,
  GRAPH_BOUNDS,
  placeSlot,
  placeWebBackend,
  type GraphLayout,
} from "./graphLayout";
import {
  fetchConnections,
  measureCopyRates,
  type ConnectionsReading,
  type ServerConnection,
} from "./serverConnections";
import { HEADER_HEIGHT_PX } from "./headerHeight";
import { fetchMemory, type MemoryReading } from "./serverMemory";

const CONNECTIONS_POLL_MS = 1000;

const SERVER_NODE_ID = "server";

// How much of the view the graph spans, each way, when the view is fitted.
// The rest is open canvas around it, to drag clients out into.
const GRAPH_SHARE_OF_VIEW = 0.5;

// Room around the fitted area, with more at the top for the page's notice.
const FIT_PADDING = { top: "44px", bottom: "24px", x: "16px" } as const;

const nodeTypes = { server: ServerNode, client: ClientNode, webBackend: WebBackendNode };
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
  identity: ServerFlowNode["data"]["identity"],
  layout: GraphLayout,
): FlowNode[] {
  const everyConnection: ServerConnection[] = view.status === "live" ? view.reading.connections : [];
  const readAtMs = view.status === "live" ? view.reading.readAtMs : 0;
  // The server's clients are every connection but the web backends' own,
  // which are neither counted nor drawn among them.
  const connections = everyConnection.filter((connection) => !connection.isWebBackend);
  const webBackends = everyConnection.filter((connection) => connection.isWebBackend);

  const serverData: ServerFlowNode["data"] = {
    identity,
    reading:
      view.status === "live"
        ? {
            pivotAddress: view.reading.pivotAddress,
            clientCount: connections.length,
            queriesRunning: connections.filter((connection) => connection.state === "query").length,
            memoryUsage: view.memory.usage,
          }
        : undefined,
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

  const webBackendNodes = webBackends.map((connection, order): WebBackendFlowNode => {
    const nodeId = `connection-${connection.id}`;
    const placed = current.find(
      (node): node is WebBackendFlowNode => node.type === "webBackend" && node.id === nodeId,
    );
    const isWherePlaced = placed && placed.data.layout === layout && placed.data.order === order;
    return {
      ...placed,
      id: nodeId,
      type: "webBackend",
      position: isWherePlaced ? placed.position : placeWebBackend(layout, order).position,
      data: { connection, order, layout },
    };
  });

  return [serverNode, ...clientNodes, ...webBackendNodes];
}

export function ServerPage() {
  // Reads the server again a moment after each reading comes back, so a slow
  // read never has another queue up behind it.
  const [view, setView] = useState<ServerView>({ status: "loading" });
  const [identity, setIdentity] = useState<ServerFlowNode["data"]["identity"]>();
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
        setIdentity({ pivotHost: reading.pivotHost, machine: reading.machine });
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

  // The graph fills the window below the app's header.
  const [viewportSize, setViewportSize] = useState(() => ({
    width: window.innerWidth,
    height: window.innerHeight - HEADER_HEIGHT_PX,
  }));
  useEffect(() => {
    const measureViewport = () =>
      setViewportSize({ width: window.innerWidth, height: window.innerHeight - HEADER_HEIGHT_PX });
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
    setNodes((current) => syncNodes(current, view, identity, layout));
  }, [view, identity, layout, setNodes]);

  const edges = useMemo(
    () =>
      nodes.flatMap((node): ConnectionFlowEdge[] => {
        if (node.type === "client") {
          return [
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
          ];
        }
        if (node.type === "webBackend") {
          return [
            {
              id: `${node.id}-to-server`,
              type: "connection",
              source: node.id,
              target: SERVER_NODE_ID,
              targetHandle: placeWebBackend(node.data.layout, node.data.order).serverHandle,
              data: { state: "fixture" },
            },
          ];
        }
        return [];
      }),
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

      <Panel position="top-left">
        {view.status === "loading" && (
          <div className="connections-notice">Reading the server…</div>
        )}
        {view.status === "live" && (
          <div className="connections-notice">
            Live connections and memory of {view.reading.pivotAddress}.
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
