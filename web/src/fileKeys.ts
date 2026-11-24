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

// The longest a value is written in full where room is short, and how much
// of a longer one is then written, before the mark that it goes on.
const LONGEST_WHOLE_VALUE = 13;
const SHORTENED_VALUE_LENGTH = 8;
// The fewest characters a label cut short is worth writing, its mark among
// them.
const LEAST_CUT_LABEL = 4;

const STARTS_WITH_DAY = /^\d{4}-\d{2}-\d{2}/;

// A long value as its start, which is what tells an id or a name from its
// neighbours. A date or a timestamp is left whole: its start is the part
// its neighbours share.
function shortenValue(value: string): string {
  if (value.length <= LONGEST_WHOLE_VALUE || STARTS_WITH_DAY.test(value)) return value;
  return `${value.slice(0, SHORTENED_VALUE_LENGTH)}…`;
}

// A key as a block's label writes it: its one value, or its range, with
// long values shortened if `isShortened`. A range of timestamps within one
// day gives the day once, and shortened gives only the times.
function writeKey({ min, max }: FileKey, isShortened: boolean): string {
  const write = (bound: Bound) => (isShortened ? shortenValue(String(bound)) : String(bound));
  if (compareBounds(min, max) === 0) return write(min);
  const from = String(min);
  const to = String(max);
  const startsWithDayAndTime = /^\d{4}-\d{2}-\d{2}[T ]\d/;
  if (
    startsWithDayAndTime.test(from) &&
    startsWithDayAndTime.test(to) &&
    from.slice(0, 10) === to.slice(0, 10)
  ) {
    // Shortened, the day goes from both ends: within a day the times are
    // what tell the file from its neighbours.
    return isShortened ? `${from.slice(11)} → ${to.slice(11)}` : `${from} → ${to.slice(11)}`;
  }
  return `${write(min)} → ${write(max)}`;
}

// What to write on a file's block that is `widthPx` wide: the most of
// `keys` that fits, which are the file's keys below the sections it is drawn
// under. All of them are tried first, each with its column's name and then
// without the names, then one sort column fewer from the end, and so on down
// to the keys that place the file, and the range among those alone. Then the
// keys that place the file with their long values shortened, and last the
// start of those keys, as much as there is room for. Nothing when the block
// is too narrow for even that, or there are no keys.
export function chooseBlockLabel(keys: FileKey[], widthPx: number): string | undefined {
  if (keys.length === 0) return undefined;
  const room = Math.floor((widthPx - 2 * LABEL_MARGIN_PX) / LABEL_CHARACTER_PX);
  const placing = keys.slice(0, countPlacingKeys(keys));
  const range = placing[placing.length - 1];
  const candidates: string[] = [];
  for (let count = keys.length; count >= placing.length; count -= 1) {
    const written = keys.slice(0, count);
    candidates.push(
      written.map((key) => `${key.column} ${writeKey(key, false)}`).join(" · "),
      written.map((key) => writeKey(key, false)).join(" · "),
    );
  }
  const placingWhole = placing.map((key) => writeKey(key, false)).join(" · ");
  candidates.push(
    writeKey(range, false),
    placing.map((key) => writeKey(key, true)).join(" · "),
    writeKey(range, true),
  );
  const fitting = candidates.find((candidate) => candidate.length <= room);
  if (fitting) return fitting;
  // The start of a date or a timestamp is the part its neighbours share,
  // and says nothing alone.
  if (room < LEAST_CUT_LABEL || STARTS_WITH_DAY.test(placingWhole)) return undefined;
  return `${placingWhole.slice(0, room - 1).trimEnd()}…`;
}
