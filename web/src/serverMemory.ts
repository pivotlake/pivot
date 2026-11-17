// How the pivot server's buffer pool is used, as the web backend reads it
// from the server's `system.memory_blocks` table.

import { readBackend } from "./backend";

// The blocks of the pool in one state on one NUMA node.
export type MemoryUse = {
  node: number;
  // `free`, `pinned`, or the name of the cache holding the blocks.
  state: string;
  blockCount: number;
  bytes: number;
};

export type MemoryReading = {
  usage: MemoryUse[];
};

export function fetchMemory(): Promise<MemoryReading> {
  return readBackend("/api/memory");
}

// The uses of memory the page tells apart, in the order it draws them. Each
// has the names it can be written with inside its part of a bar, longest
// first, for the bar to pick the longest that fits; a part too narrow for
// any of them goes unnamed.
export const MEMORY_PARTS = [
  { key: "pinned", name: "pinned", barLabels: ["pinned"] },
  {
    key: "decompressed_cache",
    name: "decompressed cache",
    barLabels: ["decompressed cache", "decompressed"],
  },
  {
    key: "compressed_cache",
    name: "compressed cache",
    barLabels: ["compressed cache", "compressed"],
  },
  // Every cache the page has no name of its own for.
  { key: "other", name: "other caches", barLabels: ["other caches", "other"] },
  { key: "free", name: "free", barLabels: ["free"] },
] as const;

export type MemoryPartKey = (typeof MEMORY_PARTS)[number]["key"];

// One NUMA node's share of the pool.
export type NodeMemory = {
  node: number;
  totalBytes: number;
  partBytes: Record<MemoryPartKey, number>;
};

// The pool node by node, in node order.
export function groupMemoryByNode(usage: MemoryUse[]): NodeMemory[] {
  const byNode = new Map<number, NodeMemory>();
  for (const use of usage) {
    let nodeMemory = byNode.get(use.node);
    if (!nodeMemory) {
      nodeMemory = {
        node: use.node,
        totalBytes: 0,
        partBytes: { pinned: 0, decompressed_cache: 0, compressed_cache: 0, other: 0, free: 0 },
      };
      byNode.set(use.node, nodeMemory);
    }
    const isNamedPart = MEMORY_PARTS.some((part) => part.key === use.state);
    nodeMemory.partBytes[isNamedPart ? (use.state as MemoryPartKey) : "other"] += use.bytes;
    nodeMemory.totalBytes += use.bytes;
  }
  return [...byNode.values()].sort((left, right) => left.node - right.node);
}
