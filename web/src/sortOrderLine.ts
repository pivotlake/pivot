// A table's sort order as a line from 0 to 1, for drawing files by where
// their rows fall in it.

import { buildNumberScale, buildSectionScale, compareBounds, isNumbered, type KeyScale } from "./keyScale";
import type { Bound, TableFile } from "./tables";
import type { Span } from "./zoom";

// A value of a sort column that files hold alone, as the stretch of the line
// those files are drawn in.
export type Section = Span & {
  // Which of the sort columns the value is of, the first being 0. A section
  // of a later column lies inside one of each column before it.
  level: number;
  column: string;
  value: Bound;
  // How many files hold the value alone.
  fileCount: number;
};

export type SortOrderLine = {
  // Where a file lies on the line, and how many sections it lies inside: one
  // of each of that many leading sort columns.
  //
  // A file is followed down the columns for as long as it holds one value of
  // each, into that value's stretch. At the first column it holds several
  // values of, it is drawn from the middle of its first value's stretch to
  // the middle of its last's: its bounds do not say how much of either end
  // value it holds. Two files that share only an end value, as the outputs
  // of one merge side by side do, then meet without overlapping, which is
  // how the server's compacter treats ranges that touch at one value too. A
  // file with one value of every column takes that value's whole stretch.
  placeFile(file: TableFile): Span & { sectionDepth: number };
  // The keys at the two ends of a part of the line: the values of the
  // columns there, from the first column down to the first the two ends
  // differ on.
  nameEnds(part: Span): [string, string];
  // Every section of the line, of every column but the last.
  sections: Section[];
};

// The files that share one value of each sort column above one of them, and
// how they are placed along that column.
type Group = {
  scale: KeyScale;
  // The values of the column that files of the group hold alone, each with
  // the group of the files holding it, by where the value's stretch starts
  // on `scale`. Empty for the last column, below which nothing is placed.
  sections: Map<number, { value: Bound; fileCount: number; group: Group }>;
};

// The group of `files`, which share one value of each of `columns` before
// `level`, along the column at `level`.
//
// Where no file holds one value of the column alone, and its bounds have
// numbers, the files are ranges along it and are placed by those numbers.
// Otherwise every value has a section, as long as the files in it are many,
// so the files that share a value are not squeezed into a sliver of the line
// for having a value in common.
function buildGroup(files: TableFile[], columns: string[], level: number): Group {
  const column = columns[level];
  const bounds = files.flatMap((file) => [file.bounds[column].min, file.bounds[column].max]);
  const holders = files.filter(
    (file) => compareBounds(file.bounds[column].min, file.bounds[column].max) === 0,
  );
  if (holders.length === 0 && bounds.every(isNumbered)) {
    return { scale: buildNumberScale(bounds), sections: new Map() };
  }

  const scale = buildSectionScale(
    bounds,
    holders.map((file) => file.bounds[column].min),
  );
  const sections: Group["sections"] = new Map();
  if (level + 1 < columns.length) {
    const holdersByStart = new Map<number, TableFile[]>();
    for (const file of holders) {
      const start = scale.locate(file.bounds[column].min);
      holdersByStart.set(start, [...(holdersByStart.get(start) ?? []), file]);
    }
    for (const [start, held] of holdersByStart) {
      sections.set(start, {
        value: held[0].bounds[column].min,
        fileCount: held.length,
        group: buildGroup(held, columns, level + 1),
      });
    }
  }
  return { scale, sections };
}

// The line over the bounds `files` have on `columns`, the table's sort
// columns in order. There is at least one file, and every file has bounds on
// every one of the columns.
//
// Rows are sorted by the first column, within one value of it by the second,
// and so on. So the line is the first column's values in order, a value that
// files hold alone has a stretch of the line along which the second column
// runs for those files, a value of the second has a stretch of that, and so
// on down.
export function buildSortOrderLine(files: TableFile[], columns: string[]): SortOrderLine {
  const whole = buildGroup(files, columns, 0);

  const sections: Section[] = [];
  const collectSections = (group: Group, level: number, start: number, length: number) => {
    for (const [sectionStart, section] of group.sections) {
      const span = {
        start: start + sectionStart * length,
        end: start + (sectionStart + group.scale.measure(section.value)) * length,
      };
      sections.push({
        ...span,
        level,
        column: columns[level],
        value: section.value,
        fileCount: section.fileCount,
      });
      collectSections(section.group, level + 1, span.start, span.end - span.start);
    }
  };
  collectSections(whole, 0, 0, 1);

  return {
    placeFile: (file) => {
      // The stretch of the line the file has been followed into so far, and
      // the group of the files in it.
      let group = whole;
      let start = 0;
      let length = 1;
      let sectionDepth = 0;
      for (const column of columns) {
        const { min, max } = file.bounds[column];
        const { scale } = group;
        if (compareBounds(min, max) !== 0) {
          return {
            start: start + (scale.locate(min) + scale.measure(min) / 2) * length,
            end: start + (scale.locate(max) + scale.measure(max) / 2) * length,
            sectionDepth,
          };
        }
        const valueStart = scale.locate(min);
        start += valueStart * length;
        length *= scale.measure(min);
        const section = group.sections.get(valueStart);
        if (!section) break;
        group = section.group;
        sectionDepth += 1;
      }
      return { start, end: start + length, sectionDepth };
    },
    nameEnds: (part) => {
      const startValues: string[] = [];
      const endValues: string[] = [];
      // Where the two ends are in the stretch both have been followed into.
      let group = whole;
      let { start, end } = part;
      for (;;) {
        const atStart = group.scale.readAt(start);
        const atEnd = group.scale.readAt(end);
        startValues.push(String(atStart.value));
        endValues.push(String(atEnd.value));
        if (compareBounds(atStart.value, atEnd.value) !== 0) break;
        const section = group.sections.get(group.scale.locate(atStart.value));
        if (!section) break;
        group = section.group;
        start = atStart.within;
        end = atEnd.within;
      }
      return [startValues.join(" · "), endValues.join(" · ")];
    },
    sections,
  };
}
