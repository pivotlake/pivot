import { useLayoutEffect, useRef, useState } from "react";
import { useViewport } from "@xyflow/react";
import { formatMemory } from "../format";
import {
  groupMemoryByNode,
  MEMORY_PARTS,
  type MemoryPartKey,
  type MemoryUse,
  type NodeMemory,
} from "../serverMemory";

// What a name written inside a bar needs: the width of one character of the
// bar's monospace text, and the room kept clear on either side of the name.
const BAR_CHARACTER_WIDTH_PX = 6;
const BAR_LABEL_PADDING_PX = 3;

function formatShare(bytes: number, totalBytes: number): string {
  return `${Math.round((bytes / totalBytes) * 100)}%`;
}

// One NUMA node's memory as a single bar split by use, each part named
// inside it where the name fits. Pointing at a part raises a tooltip above
// it with the part's amount and share of the node.
function NodeBar({ memory }: { memory: NodeMemory }) {
  const [hoveredKey, setHoveredKey] = useState<MemoryPartKey>();
  // The tooltip is drawn inside the graph, which scales with the zoom, and is
  // scaled back so it reads the same however far out the graph is.
  const { zoom } = useViewport();

  // The card has a fixed width, so the bar is measured once.
  const bar = useRef<HTMLDivElement>(null);
  const [barWidth, setBarWidth] = useState(0);
  useLayoutEffect(() => {
    if (!bar.current) {
      throw new Error("the memory bar is not mounted");
    }
    setBarWidth(bar.current.offsetWidth);
  }, []);

  // Each part with where it starts along the bar, as the bytes before it.
  let bytesBefore = 0;
  const parts = MEMORY_PARTS.filter((part) => memory.partBytes[part.key] > 0).map((part) => {
    const placed = { ...part, bytes: memory.partBytes[part.key], startBytes: bytesBefore };
    bytesBefore += placed.bytes;
    return placed;
  });
  const hovered = parts.find((part) => part.key === hoveredKey);

  return (
    <div className="memory-node">
      <div className="memory-node-heading">
        <span>node {memory.node}</span>
        <span className="memory-node-pinned">{formatMemory(memory.partBytes.pinned)} pinned</span>
      </div>
      <div className="memory-bar-frame">
        <div
          ref={bar}
          className="memory-bar"
          role="img"
          aria-label={`NUMA node ${memory.node}: ${parts
            .map(
              (part) =>
                `${part.name} ${formatMemory(part.bytes)}, ${formatShare(part.bytes, memory.totalBytes)}`,
            )
            .join("; ")}`}
        >
          {parts.map((part) => {
            const width = (part.bytes / memory.totalBytes) * barWidth;
            const label = part.barLabels.find(
              (candidate) => candidate.length * BAR_CHARACTER_WIDTH_PX + 2 * BAR_LABEL_PADDING_PX <= width,
            );
            return (
              <div
                key={part.key}
                className={`memory-segment is-${part.key}`}
                style={{ flexGrow: part.bytes }}
                onMouseEnter={() => setHoveredKey(part.key)}
                onMouseLeave={() => setHoveredKey(undefined)}
              >
                {label}
              </div>
            );
          })}
        </div>
        {hovered && (
          <div
            className="memory-tooltip"
            role="tooltip"
            style={{
              left: ((hovered.startBytes + hovered.bytes / 2) / memory.totalBytes) * barWidth,
              transform: `translateX(-50%) scale(${1 / zoom})`,
            }}
          >
            <strong>{hovered.name}</strong>
            {formatMemory(hovered.bytes)} · {formatShare(hovered.bytes, memory.totalBytes)}
          </div>
        )}
      </div>
    </div>
  );
}

// How the server's buffer pool is used: a bar for each NUMA node, which is
// where memory is pinned.
export function MemorySection({ usage }: { usage: MemoryUse[] }) {
  const nodes = groupMemoryByNode(usage);
  const poolBytes = nodes.reduce((bytes, memory) => bytes + memory.totalBytes, 0);

  return (
    <section className="memory">
      <div className="memory-heading">
        <span className="section-label">Buffer pool</span>
        <span className="memory-total">{formatMemory(poolBytes)}</span>
      </div>
      {nodes.map((memory) => (
        <NodeBar key={memory.node} memory={memory} />
      ))}
    </section>
  );
}
