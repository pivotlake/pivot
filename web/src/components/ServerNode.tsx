import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { ServerFlowNode } from "../flowTypes";
import { pivotVersion } from "../serverHost";
import { MemorySection } from "./MemorySection";

const GIBIBYTE = 2 ** 30;

// Shown in place of what the page has no reading for.
const UNKNOWN_COUNT = "–";

// One handle per side, named after the side, for connections to arrive at.
const SERVER_SIDES = [Position.Left, Position.Right, Position.Top, Position.Bottom];

export function ServerNode({ data }: NodeProps<ServerFlowNode>) {
  const { identity, reading } = data;
  const machine = identity?.machine;

  return (
    <div className="server-card">
      {SERVER_SIDES.map((side) => (
        <Handle key={side} type="target" id={side} position={side} isConnectable={false} />
      ))}

      <header className="server-identity">
        <div className="server-eyebrow">
          <span className={reading ? "live-dot" : "live-dot is-unknown"} aria-hidden="true" />
          <span>pivot server</span>
          <span className="server-version">v{pivotVersion}</span>
        </div>
        {/* The server is named by its machine when that is known, and by the
            host it is reached at otherwise. */}
        <h1 className="server-hostname">
          {identity ? (machine?.hostname ?? identity.pivotHost) : UNKNOWN_COUNT}
        </h1>
        <ul className="server-facts">
          {reading && <li>{reading.pivotAddress}</li>}
          {machine && (
            <>
              <li>{machine.cores} cores</li>
              <li>{Math.round(machine.memoryBytes / GIBIBYTE)} GiB</li>
              <li>
                {machine.operatingSystem}/{machine.architecture}
              </li>
            </>
          )}
        </ul>
      </header>

      <section className="server-counters">
        <div className="server-counter">
          <div className="section-label">Queries running</div>
          <div className="counter-value">{reading ? reading.queriesRunning : UNKNOWN_COUNT}</div>
        </div>
        <div className="server-counter">
          <div className="section-label">Clients connected</div>
          <div className="counter-value">{reading ? reading.clientCount : UNKNOWN_COUNT}</div>
        </div>
      </section>

      {reading && <MemorySection usage={reading.memoryUsage} />}
    </div>
  );
}
