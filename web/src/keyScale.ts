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
  // to 1.
  locate(bound: Bound): number;
  // How much of the line a value's stretch takes: the room there is inside
  // the value for whatever sorts the rows that share it.
  measure(bound: Bound): number;
  // What stands at `position` of the line, 0 to 1.
  readAt(position: number): Standing;
};

// Whether a bound has a number to be placed by.
export function isNumbered(bound: Bound): boolean {
  return readAsNumber(bound) !== undefined;
}

// The scale that places `bounds` by their numbers, so distances along the
// line are distances between keys. Every bound has a number, and at least
// two of the numbers differ. One value takes as much of the line as the
// smallest gap between two different bounds, which is close to nothing for
// a column whose values are all different, such as a precise timestamp.
export function buildNumberScale(bounds: Bound[]): KeyScale {
  // The bounds in order with one of each number, and their numbers.
  const numbered = bounds
    .map((bound) => ({ bound, number: readAsNumber(bound) as number }))
    .sort((left, right) => left.number - right.number)
    .filter((entry, index, sorted) => index === 0 || sorted[index - 1].number !== entry.number);
  const distinct = numbered.map((entry) => entry.bound);
  const values = numbered.map((entry) => entry.number);
  const low = values[0];
  const lastIndex = values.length - 1;
  const smallestGap = Math.min(...values.slice(1).map((value, index) => value - values[index]));
  const length = values[lastIndex] - low + smallestGap;
  return {
    locate: (bound) => ((readAsNumber(bound) as number) - low) / length,
    measure: () => smallestGap / length,
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
  };
}

// The scale that gives every different value among `bounds` a stretch of its
// own, the values in order and the stretches side by side filling the line.
// Distances along it say nothing of how far apart two keys are.
//
// `held` are the values files hold alone, one for each such file, all of
// them among `bounds`. A value's stretch is as long as the number of files
// that hold it, and one more: the files of a value many files hold are drawn
// inside its stretch, and need the room.
export function buildSectionScale(bounds: Bound[], held: Bound[]): KeyScale {
  // Bounds that all have numbers are told apart and ordered by them, and
  // any others by their text: two bounds can be written differently and
  // still have the same number.
  const areNumbered = bounds.every(isNumbered);
  const readKey = (bound: Bound) => (areNumbered ? String(readAsNumber(bound)) : String(bound));
  const distinct = [...new Map(bounds.map((bound) => [readKey(bound), bound])).values()].sort((left, right) => {
    if (areNumbered) return (readAsNumber(left) as number) - (readAsNumber(right) as number);
    const leftText = String(left);
    const rightText = String(right);
    return leftText < rightText ? -1 : leftText > rightText ? 1 : 0;
  });

  const holderCounts = new Map<string, number>();
  for (const bound of held) {
    const key = readKey(bound);
    holderCounts.set(key, (holderCounts.get(key) ?? 0) + 1);
  }
  const weights = distinct.map((bound) => 1 + (holderCounts.get(readKey(bound)) ?? 0));
  const totalWeight = weights.reduce((total, weight) => total + weight, 0);

  // Each value's stretch, in the values' order.
  const stretches: { value: Bound; start: number; length: number }[] = [];
  let weightBefore = 0;
  for (const [index, value] of distinct.entries()) {
    stretches.push({ value, start: weightBefore / totalWeight, length: weights[index] / totalWeight });
    weightBefore += weights[index];
  }
  const stretchByKey = new Map(stretches.map((stretch) => [readKey(stretch.value), stretch]));
  const findStretch = (bound: Bound) => {
    const stretch = stretchByKey.get(readKey(bound));
    if (!stretch) {
      throw new Error(`${String(bound)} is not one of the bounds the scale was built over`);
    }
    return stretch;
  };

  return {
    locate: (bound) => findStretch(bound).start,
    measure: (bound) => findStretch(bound).length,
    readAt: (position) => {
      // The last stretch that starts at or before the place.
      let index = 0;
      while (index + 1 < stretches.length && stretches[index + 1].start <= position) index += 1;
      const { value, start, length } = stretches[index];
      return { value, within: Math.min((position - start) / length, 1) };
    },
  };
}
