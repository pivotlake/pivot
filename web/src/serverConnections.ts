// The pivot server's live connections, as the web backend (`web/server`)
// reads them from the server's `system.connections` table.

import { readBackend } from "./backend";

export type ConnectionState = "idle" | "query" | "copy";

// One row of `system.connections`. Instants are milliseconds since the Unix
// epoch.
export type ServerConnection = {
  id: number;
  userName: string;
  // Empty when the client did not name itself.
  applicationName: string;
  clientAddress: string;
  connectedAtMs: number;
  state: ConnectionState;
  stateChangedAtMs: number;
  // Empty while idle.
  statement: string;
  copyBytes: number;
  // Whether the connection is a web backend's: the one this page is read
  // through, or another page's.
  isWebBackend: boolean;
};

// The machine a pivot server runs on.
export type Machine = {
  hostname: string;
  cores: number;
  memoryBytes: number;
  architecture: string;
  operatingSystem: string;
};

export type ConnectionsReading = {
  // When the backend read the connections, by its clock: what the instants in
  // `connections` are measured against, so the browser's clock never matters.
  readAtMs: number;
  // The pivot server the connections belong to, as `host:port`.
  pivotAddress: string;
  // The host of that address alone.
  pivotHost: string;
  // The machine the server runs on. A server does not report it, so the
  // backend knows it only when the server is on the backend's own machine,
  // and it is null otherwise.
  machine: Machine | null;
  connections: ServerConnection[];
};

export function fetchConnections(): Promise<ConnectionsReading> {
  return readBackend("/api/connections");
}

// How fast each copying connection is sending, in bytes a second, measured
// between two readings. A copy seen for the first time has no rate yet.
export function measureCopyRates(
  previous: ConnectionsReading,
  current: ConnectionsReading,
): Map<number, number> {
  const elapsedSeconds = (current.readAtMs - previous.readAtMs) / 1000;
  const rates = new Map<number, number>();
  for (const connection of current.connections) {
    const before = previous.connections.find((candidate) => candidate.id === connection.id);
    const isSameCopy =
      connection.state === "copy" &&
      before?.state === "copy" &&
      before.stateChangedAtMs === connection.stateChangedAtMs;
    if (isSameCopy && elapsedSeconds > 0) {
      rates.set(connection.id, (connection.copyBytes - before.copyBytes) / elapsedSeconds);
    }
  }
  return rates;
}
