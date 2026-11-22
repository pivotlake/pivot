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

// What stands at a place on a scale's line: the value whose stretch holds
// the place, and how far into that stretch the place is, 0 to 1.
export type Standing = { value: Bound; within: number };

// The value at `number` on a line of `bounds`, written the way they are, for
// bounds that all have numbers.
function writeNumberAsBound(number: number, bounds: Bound[]): Bound {
  const [sample] = bounds;
  if (typeof sample === "boolean") return number >= 0.5;
  if (typeof sample === "number") {
    return bounds.every((bound) => Number.isInteger(bound)) ? Math.round(number) : Number(number.toPrecision(6));
  }
  const instant = new Date(number);
  // A date alone is read as UTC and so is a timestamp that names its zone,
  // while one that names none is read as local time. Each is written back
  // the way it was read.
  if (sample.length === 10) return instant.toISOString().slice(0, 10);
  if (/(Z|[+-]\d{2}(:?\d{2})?)$/.test(sample)) return instant.toISOString();
  const pad = (part: number) => String(part).padStart(2, "0");
  const day = `${instant.getFullYear()}-${pad(instant.getMonth() + 1)}-${pad(instant.getDate())}`;
  const time = `${pad(instant.getHours())}:${pad(instant.getMinutes())}:${pad(instant.getSeconds())}`;
  const milliseconds = instant.getMilliseconds();
  const fraction = milliseconds === 0 ? "" : `.${String(milliseconds).padStart(3, "0")}`;
  return `${day}${sample[10]}${time}${fraction}`;
}

export type KeyScale = {
  // Where a value's own stretch of the line starts, the line running from 0
  // to 1 and each value's stretch being `valueShare` long.
  locate(bound: Bound): number;
  // What stands at `position` of the line, 0 to 1.
  readAt(position: number): Standing;
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
// are placed by their rank among the different bounds as text, evenly.
export function buildKeyScale(bounds: Bound[]): KeyScale {
  const numbers = bounds.map(readAsNumber);
  if (numbers.every((number) => number !== undefined)) {
    // The bounds in order with one of each number, and their numbers.
    const numbered = bounds
      .map((bound, index) => ({ bound, number: numbers[index] as number }))
      .sort((left, right) => left.number - right.number)
      .filter((entry, index, sorted) => index === 0 || sorted[index - 1].number !== entry.number);
    if (numbered.length === 1) return buildOneValueScale(numbered[0].bound);

    const distinct = numbered.map((entry) => entry.bound);
    const values = numbered.map((entry) => entry.number);
    const low = values[0];
    const lastIndex = values.length - 1;
    const smallestGap = Math.min(...values.slice(1).map((value, index) => value - values[index]));
    const length = values[lastIndex] - low + smallestGap;
    return {
      locate: (bound) => ((readAsNumber(bound) as number) - low) / length,
      readAt: (position) => {
        const number = low + position * length;
        // The last bound at or before the place, whose stretch is the only
        // one that can hold it.
        let index = 0;
        while (index < lastIndex && values[index + 1] <= number) index += 1;
        const pastBound = number - values[index];
        // The line ends where the largest bound's stretch does, so whatever
        // is past that bound is in its stretch, however the sums round.
        if (index === lastIndex || pastBound <= smallestGap) {
          return { value: distinct[index], within: Math.min(pastBound / smallestGap, 1) };
        }
        // No bound's stretch holds the place. Every value has a stretch as
        // long as a bound's, so the place is in the stretch of the value a
        // whole number of stretches past the smallest bound.
        const stretchesPast = Math.floor((number - low) / smallestGap);
        return {
          value: writeNumberAsBound(low + stretchesPast * smallestGap, distinct),
          within: (number - low) / smallestGap - stretchesPast,
        };
      },
      valueShare: smallestGap / length,
    };
  }

  // The bounds as text, in order, one of each. Text is what tells these
  // bounds apart: two of them can be written differently and still have the
  // same number.
  const distinct = [...new Map(bounds.map((bound) => [String(bound), bound])).entries()]
    .sort(([leftText], [rightText]) => (leftText < rightText ? -1 : leftText > rightText ? 1 : 0))
    .map(([, bound]) => bound);
  if (distinct.length === 1) return buildOneValueScale(distinct[0]);

  const rankByText = new Map(distinct.map((bound, rank) => [String(bound), rank]));
  return {
    locate: (bound) => {
      const rank = rankByText.get(String(bound));
      if (rank === undefined) {
        throw new Error(`${String(bound)} is not one of the bounds the scale was built over`);
      }
      return rank / distinct.length;
    },
    readAt: (position) => {
      const rank = Math.min(Math.floor(position * distinct.length), distinct.length - 1);
      return { value: distinct[rank], within: position * distinct.length - rank };
    },
    valueShare: 1 / distinct.length,
  };
}

// The scale of a column that holds `value` and nothing else, which has the
// whole line.
function buildOneValueScale(value: Bound): KeyScale {
  return {
    locate: () => 0,
    readAt: (position) => ({ value, within: position }),
    valueShare: 1,
  };
}
