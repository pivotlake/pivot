import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from "reactflow";
import { rate as fmtRate } from "../format";

export interface FlowEdgeData {
  signal: string;
  rate: number;
}

const TEAL = "#0fb5a6";
const IDLE = "#cdd2dd";

// Busier streams flow faster (shorter dash-cycle period).
function flowDuration(rate: number): number {
  if (rate <= 0) return 0;
  return Math.max(0.35, Math.min(1.4, 9000 / (rate + 1)));
}

export default function FlowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps<FlowEdgeData>) {
  const [path, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const r = data?.rate ?? 0;
  const live = r > 0;

  // A flowing dashed stroke when live (animated via CSS, see styles.css). No
  // moving SVG node, so nothing flashes at the origin when the edge re-renders.
  const style = live
    ? {
        stroke: TEAL,
        strokeWidth: 2,
        strokeDasharray: "5 5",
        animation: `flow-dash ${flowDuration(r)}s linear infinite`,
      }
    : { stroke: IDLE, strokeWidth: 1.5 };

  return (
    <>
      <BaseEdge id={id} path={path} style={style} />
      <EdgeLabelRenderer>
        <div
          className="edge-label"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}
        >
          <span className="sig">{data?.signal}</span>
          <span className={`rate ${live ? "live" : ""}`}>{live ? fmtRate(r) : "0 rows/s"}</span>
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

export const edgeTypes = { flow: FlowEdge };
