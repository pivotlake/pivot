// Placing a column's bounds along a line, for drawing files by the range of
// keys they hold.

import type { Bound } from "./tables";

// The number a bound stands at on a line, for the bounds that have one:
// numbers, booleans, and text that starts as a date or a timestamp.
function readAsNumber(bound: Bound): number | undefined {
  if (typeof bound === "number") return bound;
  if (typeof bound === "boolean") return bound ? 1 : 0;
  if (!/^\d{4}-\d{2}-\d{2}/.test(bound)) return undefined;
  const milliseconds = Date.parse(bound.replace(" ", "T"));
  return Number.isNaN(milliseconds) ? undefined : milliseconds;
}

// Orders two bounds of one column: by their numbers when both have one, and
// as text otherwise.
export function compareBounds(left: Bound, right: Bound): number {
  const leftNumber = readAsNumber(left);
  const rightNumber = readAsNumber(right);
  if (leftNumber !== undefined && rightNumber !== undefined) {
    return leftNumber - rightNumber;
  }
  const leftText = String(left);
  const rightText = String(right);
  return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
}

export type KeyScale = {
  // Where a value's own stretch of the line starts, the line running from 0
  // to 1 and each value's stretch being `valueShare` long.
  locate(bound: Bound): number;
  smallest: Bound;
  largest: Bound;
  // How much of the line one value of the column takes: the room there is
  // inside a value for whatever sorts the rows that share it. The same for
  // every value. Close to nothing for a column whose values are all
  // different, such as a precise timestamp, and a good part of the line for
  // one with few values, such as a timestamp rounded to the minute over an
  // hour.
  valueShare: number;
};

// The scale over `bounds`, of which there is at least one.
//
// When every bound has a number, values are placed by it, so distances along
// the line are distances between keys, and one value takes as much of the
// line as the smallest gap between two different bounds. Otherwise values
// are placed by their rank among the different bounds, evenly.
export function buildKeyScale(bounds: Bound[]): KeyScale {
  const sorted = [...bounds].sort(compareBounds);
  const smallest = sorted[0];
  const largest = sorted[sorted.length - 1];
  const distinct = sorted.filter(
    (bound, index) => index === 0 || compareBounds(sorted[index - 1], bound) !== 0,
  );
  if (distinct.length === 1) {
    return { locate: () => 0, smallest, largest, valueShare: 1 };
  }

  const numbers = distinct.map(readAsNumber);
  if (numbers.every((number) => number !== undefined)) {
    const values = numbers as number[];
    const low = values[0];
    const smallestGap = Math.min(...values.slice(1).map((value, index) => value - values[index]));
    const length = values[values.length - 1] - low + smallestGap;
    return {
      locate: (bound) => ((readAsNumber(bound) as number) - low) / length,
      smallest,
      largest,
      valueShare: smallestGap / length,
    };
  }

  const rankByText = new Map(distinct.map((bound, rank) => [String(bound), rank]));
  return {
    locate: (bound) => {
      const rank = rankByText.get(String(bound));
      if (rank === undefined) {
        throw new Error(`${String(bound)} is not one of the bounds the scale was built over`);
      }
      return rank / distinct.length;
    },
    smallest,
    largest,
    valueShare: 1 / distinct.length,
  };
}
