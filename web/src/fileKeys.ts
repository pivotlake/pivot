// What a file holds of its table's sort columns, in words: for a tooltip, and
// for writing on the file's block where the block is wide enough.

import { compareBounds } from "./keyScale";
import type { Bound, TableFile } from "./tables";

// One sort column and the file's bounds on it.
export type FileKey = { column: string; min: Bound; max: Bound };

// The width of one character of a block's label, which is set in the page's
// fixed-width face at 10px.
const LABEL_CHARACTER_PX = 6.1;
// The room a label keeps clear at each end of its block: the block's outline
// and the gap between it and the writing.
const LABEL_MARGIN_PX = 6;

// A file's bounds on `sortBy`, its table's sort columns in order. Stops short
// at a column the file records no bounds on.
export function readFileKeys(file: TableFile, sortBy: string[]): FileKey[] {
  const keys: FileKey[] = [];
  for (const column of sortBy) {
    const bounds = file.bounds[column];
    if (!bounds) break;
    keys.push({ column, ...bounds });
  }
  return keys;
}

// How many of a file's keys, from the first, say where its rows are in the
// sort order: its one value of each column down to the first column it holds
// several values of, and its range there. The bounds on the columns below
// that are over all the file's rows, and place it no further.
function countPlacingKeys(keys: FileKey[]): number {
  const firstRange = keys.findIndex((key) => compareBounds(key.min, key.max) !== 0);
  return firstRange === -1 ? keys.length : firstRange + 1;
}

// A key as a block's label writes it: its one value, or its range. A range
// of timestamps within one day gives the day once.
function writeKey({ min, max }: FileKey): string {
  if (compareBounds(min, max) === 0) return String(min);
  const from = String(min);
  const to = String(max);
  const startsWithDay = /^\d{4}-\d{2}-\d{2}[T ]\d/;
  if (startsWithDay.test(from) && startsWithDay.test(to) && from.slice(0, 10) === to.slice(0, 10)) {
    return `${from} → ${to.slice(11)}`;
  }
  return `${from} → ${to}`;
}

// What to write on a file's block that is `widthPx` wide: the most of the
// file's keys that fits. All of them are tried first, each with its column's
// name and then without the names, then one sort column fewer from the end,
// and so on down to the keys that place the file, and last the range among
// those alone. Nothing when the block is too narrow for any of them, or the
// file has no keys.
export function chooseBlockLabel(keys: FileKey[], widthPx: number): string | undefined {
  if (keys.length === 0) return undefined;
  const room = (widthPx - 2 * LABEL_MARGIN_PX) / LABEL_CHARACTER_PX;
  const placingCount = countPlacingKeys(keys);
  const candidates: string[] = [];
  for (let count = keys.length; count >= placingCount; count -= 1) {
    const written = keys.slice(0, count);
    candidates.push(
      written.map((key) => `${key.column} ${writeKey(key)}`).join(" · "),
      written.map(writeKey).join(" · "),
    );
  }
  candidates.push(writeKey(keys[placingCount - 1]));
  return candidates.find((candidate) => candidate.length <= room);
}
