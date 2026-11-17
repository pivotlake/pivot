import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { ServerFlowNode } from "../flowTypes";
import { MemorySection } from "./MemorySection";

const GIBIBYTE = 2 ** 30;

// Shown in place of a count the page has no reading for.
const UNKNOWN_COUNT = "–";

// One handle per side, named after the side, for connections to arrive at.
const SERVER_SIDES = [Position.Left, Position.Right, Position.Top, Position.Bottom];

export function ServerNode({ data }: NodeProps<ServerFlowNode>) {
  const { host, reading, compaction } = data;
  const isCompacting = compaction.kind === "running";
  const filesRemaining = isCompacting ? compaction.filesToMerge - compaction.filesMerged : 0;
  const mergedFraction = isCompacting ? compaction.filesMerged / compaction.filesToMerge : 0;

  return (
    <div className="server-card">
      {SERVER_SIDES.map((side) => (
        <Handle key={side} type="target" id={side} position={side} isConnectable={false} />
      ))}

      <header className="server-identity">
        <div className="server-eyebrow">
          <span className={reading ? "live-dot" : "live-dot is-unknown"} aria-hidden="true" />
          <span>pivot server</span>
          <span className="server-version">v{host.version}</span>
        </div>
        <h1 className="server-hostname">{host.hostname}</h1>
        <ul className="server-facts">
          {reading && <li>{reading.pivotAddress}</li>}
          <li>{host.cores} cores</li>
          <li>{Math.round(host.memoryBytes / GIBIBYTE)} GiB</li>
          <li>
            {host.platform}/{host.architecture}
          </li>
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

      <section className={isCompacting ? "compaction is-running" : "compaction"}>
        <div className="compaction-heading">
          <span className="section-label">
            Compaction <span className="simulated-tag">simulated</span>
          </span>
          <span className="compaction-table">{isCompacting ? compaction.table : "idle"}</span>
        </div>
        <div
          className="compaction-track"
          role="progressbar"
          aria-label="Compaction progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(mergedFraction * 100)}
        >
          <div className="compaction-fill" style={{ width: `${mergedFraction * 100}%` }} />
        </div>
        <div className="compaction-figures">
          {isCompacting ? (
            <>
              <span>
                <strong>{filesRemaining}</strong> files remaining
              </span>
              <span>
                {compaction.filesMerged} / {compaction.filesToMerge} merged
              </span>
            </>
          ) : (
            <>
              <span>
                <strong>0</strong> files remaining
              </span>
              <span>
                last: {compaction.finishedTable}, {compaction.finishedFiles} files
              </span>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
