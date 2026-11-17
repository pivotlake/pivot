import { useSyncExternalStore } from "react";
import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "@xyflow/react";
import type { ConnectionFlowEdge } from "../flowTypes";
import { formatBytes } from "../format";

const PARTICLE_COUNT = 3;
const PARTICLE_TRAVEL_SECONDS = 1.5;

const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

function subscribeToReducedMotion(notify: () => void): () => void {
  reducedMotionQuery.addEventListener("change", notify);
  return () => reducedMotionQuery.removeEventListener("change", notify);
}

// A client's connection to the server: a dotted line while the client is only
// connected, and a stream of particles travelling to the server while it
// sends COPY data.
export function ConnectionEdge({
  id,
  sourceX,
  sourceY,
  sourcePosition,
  targetX,
  targetY,
  targetPosition,
  data,
}: EdgeProps<ConnectionFlowEdge>) {
  const prefersReducedMotion = useSyncExternalStore(
    subscribeToReducedMotion,
    () => reducedMotionQuery.matches,
  );
  if (!data) {
    throw new Error(`connection edge ${id} has no state`);
  }
  const { state, copyBytesPerSecond } = data;
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      {state === "copy" && <path d={path} className="connection-glow" />}
      <BaseEdge id={id} path={path} className={`connection-path is-${state}`} />
      {state === "copy" &&
        !prefersReducedMotion &&
        Array.from({ length: PARTICLE_COUNT }, (_, particle) => (
          <circle key={particle} r={3.5} className="connection-particle">
            <animateMotion
              path={path}
              dur={`${PARTICLE_TRAVEL_SECONDS}s`}
              begin={`${(-particle * PARTICLE_TRAVEL_SECONDS) / PARTICLE_COUNT}s`}
              repeatCount="indefinite"
            />
          </circle>
        ))}
      {copyBytesPerSecond !== undefined && (
        <EdgeLabelRenderer>
          <div
            className="connection-rate"
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
          >
            {formatBytes(copyBytesPerSecond)}/s
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
