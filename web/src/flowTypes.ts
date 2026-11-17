import type { Edge, Node } from "@xyflow/react";
import type { GraphLayout } from "./graphLayout";
import type { ConnectionState, ServerConnection } from "./serverConnections";
import type { ServerHost } from "./serverHost";
import type { MemoryUse } from "./serverMemory";
import type { Compaction } from "./simulation";

export type ServerFlowNode = Node<
  {
    host: ServerHost;
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
    compaction: Compaction;
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

export type FlowNode = ServerFlowNode | ClientFlowNode;

export type ConnectionFlowEdge = Edge<
  {
    state: ConnectionState;
    // Bytes a second a copying client is sending, once two readings of the
    // copy have measured it.
    copyBytesPerSecond?: number;
  },
  "connection"
>;
