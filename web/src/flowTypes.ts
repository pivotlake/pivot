import type { Edge, Node } from "@xyflow/react";
import type { GraphLayout } from "./graphLayout";
import type { ConnectionState, Machine, ServerConnection } from "./serverConnections";
import type { MemoryUse } from "./serverMemory";

export type ServerFlowNode = Node<
  {
    // Which server this is, as the last reading that succeeded said: it is
    // kept through later failures, since a server that stops answering is
    // still the same server. Absent until a first reading succeeds.
    identity?: {
      // The host the backend reaches the server at.
      pivotHost: string;
      // The machine the server runs on, when the backend knows it.
      machine: Machine | null;
    };
    // What the latest reading of the server says about it. Absent until a
    // reading succeeds and after one fails, when nothing is known.
    reading?: {
      // The server's `host:port`.
      pivotAddress: string;
      clientCount: number;
      queriesRunning: number;
      // The buffer pool, by NUMA node and use.
      memoryUsage: MemoryUse[];
    };
  },
  "server"
>;

export type ClientFlowNode = Node<
  {
    connection: ServerConnection;
    // When the connection was read, which its instants are measured against.
    readAtMs: number;
    // Which place around the server the client was given, in which layout.
    slot: number;
    layout: GraphLayout;
  },
  "client"
>;

// A web backend's own connection, drawn as a fixture of the page apart from
// the server's clients.
export type WebBackendFlowNode = Node<
  {
    connection: ServerConnection;
    // Which of the web backends connected this one is, from 0.
    order: number;
    layout: GraphLayout;
  },
  "webBackend"
>;

export type FlowNode = ServerFlowNode | ClientFlowNode | WebBackendFlowNode;

export type ConnectionFlowEdge = Edge<
  {
    // What the client is doing, or that the connection is a web backend's,
    // which is drawn the same whatever it is doing.
    state: ConnectionState | "fixture";
    // Bytes a second a copying client is sending, once two readings of the
    // copy have measured it.
    copyBytesPerSecond?: number;
  },
  "connection"
>;
