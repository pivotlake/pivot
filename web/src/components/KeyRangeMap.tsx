import { useMemo, useState } from "react";
import { formatMemory } from "../format";
import { buildKeyScale, compareBounds, type KeyScale } from "../keyScale";
import type { TableFile } from "../tables";

// The least of the map's width a file is drawn with, so a file holding one
// key is still a mark that can be seen and pointed at.
const NARROWEST_BAR = 0.006;
const ROW_HEIGHT_PX = 18;
// How many of a table's sort columns the axis names before it trails off.
const NAMED_SORT_COLUMNS = 3;

// A file as it is drawn: from where its first row falls in the table's sort
// order to where its last does, 0 to 1, in one of the map's rows.
type Bar = {
  file: TableFile;
  start: number;
  end: number;
  row: number;
};

function readFileName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

// Places files along the table's sort order, 0 to 1, by their bounds on
// `columns`, the table's sort columns in order.
//
// Rows are sorted by the first column, within one value of it by the second,
// and so on. So the line is the first column's range, each value of the
// first column has a stretch of the line to itself along which the second
// column runs, each value of the second has a stretch of that, and so on
// down.
//
// A file is followed down the columns for as long as it holds one value of
// each, into that value's stretch. At the first column it holds several
// values of, it is drawn from the middle of its first value's stretch to the
// middle of its last's: its bounds do not say how much of either end value
// it holds. Two files that share only an end value, as the outputs of one
// merge side by side do, then meet without overlapping, which is how the
// server's compacter treats ranges that touch at one value too. A file with
// one value of every column takes that value's whole stretch.
function buildSortOrderScale(
  files: TableFile[],
  columns: string[],
): (file: TableFile) => { start: number; end: number } {
  const scales: KeyScale[] = columns.map((column) =>
    buildKeyScale(files.flatMap((file) => [file.bounds[column].min, file.bounds[column].max])),
  );
  return (file) => {
    // The stretch of the line the file has been followed into so far.
    let stretchStart = 0;
    let stretchLength = 1;
    for (const [level, column] of columns.entries()) {
      const { min, max } = file.bounds[column];
      const scale = scales[level];
      if (compareBounds(min, max) !== 0) {
        const middle = scale.valueShare / 2;
        return {
          start: stretchStart + (scale.locate(min) + middle) * stretchLength,
          end: stretchStart + (scale.locate(max) + middle) * stretchLength,
        };
      }
      stretchStart += scale.locate(min) * stretchLength;
      stretchLength *= scale.valueShare;
    }
    return { start: stretchStart, end: stretchStart + stretchLength };
  };
}

// A partition's large files drawn in the table's sort order: each file a bar
// from its first row to its last. The widest files are placed first, each in
// the highest row where it overlaps nothing already there, so files that
// span everything pile up at the top and narrow ones settle beneath them.
//
// Two bars one above the other are two files holding keys from the same
// stretch of the sort order, which a query for a key there has to open both
// of, and which a layout merge rewrites into files side by side. A partition
// whose large files are fully sorted is a single row.
export function KeyRangeMap({ files, sortBy }: { files: TableFile[]; sortBy: string[] }) {
  const [hoveredPath, setHoveredPath] = useState<string>();

  const { bars, rowCount, columns, firstScale } = useMemo(() => {
    const drawable = files.filter((file) => file.bounds[sortBy[0]]);
    if (drawable.length === 0) {
      return { bars: [], rowCount: 0, columns: [], firstScale: undefined };
    }
    // The sort columns files are placed by: from the first, for as long as
    // every file drawn has bounds on them.
    const firstUnbounded = sortBy.findIndex((column) => drawable.some((file) => !file.bounds[column]));
    const columns = firstUnbounded === -1 ? sortBy : sortBy.slice(0, firstUnbounded);
    const placeFile = buildSortOrderScale(drawable, columns);

    const spans = drawable
      .map((file) => {
        const { start, end } = placeFile(file);
        return { file, start, end: Math.max(end, start + NARROWEST_BAR) };
      })
      .sort((left, right) => right.end - right.start - (left.end - left.start) || left.start - right.start);

    // What each row holds so far, for finding the highest row a span fits in.
    const rows: { start: number; end: number }[][] = [];
    const bars: Bar[] = spans.map((span) => {
      let row = rows.findIndex((placed) =>
        placed.every((other) => span.end <= other.start || other.end <= span.start),
      );
      if (row === -1) {
        row = rows.length;
        rows.push([]);
      }
      rows[row].push(span);
      return { ...span, row };
    });

    const firstScale = buildKeyScale(
      drawable.flatMap((file) => [file.bounds[columns[0]].min, file.bounds[columns[0]].max]),
    );
    return { bars, rowCount: rows.length, columns, firstScale };
  }, [files, sortBy]);

  if (!firstScale) {
    return (
      <p className="page-note">
        None of these files records bounds on <code>{sortBy[0]}</code>, so they cannot be placed by key.
      </p>
    );
  }

  const hovered = bars.find((bar) => bar.file.path === hoveredPath);
  // What the tooltip says of the hovered file's keys: its one value of each
  // sort column down to the first it holds several of, and its range there.
  const hoveredKeys: string[] = [];
  if (hovered) {
    for (const column of columns) {
      const { min, max } = hovered.file.bounds[column];
      if (compareBounds(min, max) !== 0) {
        hoveredKeys.push(`${column} ${String(min)} to ${String(max)}`);
        break;
      }
      hoveredKeys.push(`${column} ${String(min)}`);
    }
  }

  return (
    <div className="key-map">
      <div className="key-map-plot" style={{ height: rowCount * ROW_HEIGHT_PX + 1 }}>
        {bars.map((bar) => (
          <div
            key={bar.file.path}
            className={bar === hovered ? "key-map-file is-outlined" : "key-map-file"}
            style={{
              left: `${bar.start * 100}%`,
              width: `${(bar.end - bar.start) * 100}%`,
              top: bar.row * ROW_HEIGHT_PX + 2,
              height: ROW_HEIGHT_PX - 3,
            }}
            onMouseEnter={() => setHoveredPath(bar.file.path)}
            onMouseLeave={() => setHoveredPath(undefined)}
          />
        ))}
        {hovered && (
          <div
            className="tooltip"
            role="tooltip"
            style={{
              left: `${((hovered.start + hovered.end) / 2) * 100}%`,
              bottom: `calc(100% - ${hovered.row * ROW_HEIGHT_PX}px + 8px)`,
              transform: "translateX(-50%)",
            }}
          >
            <strong>{readFileName(hovered.file.path)}</strong>
            {formatMemory(hovered.file.bytes)}
            {hoveredKeys.map((keys) => (
              <span key={keys} className="tooltip-line">
                {keys}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="key-map-axis">
        <span>{String(firstScale.smallest)}</span>
        <span className="key-map-axis-name">
          {columns.length === 1
            ? columns[0]
            : `${columns[0]}, then ${columns.slice(1, NAMED_SORT_COLUMNS).join(", ")}${columns.length > NAMED_SORT_COLUMNS ? ", …" : ""}`}
        </span>
        <span>{String(firstScale.largest)}</span>
      </div>
      {bars.length < files.length && (
        <p className="page-note">
          {files.length - bars.length} files record no bounds on {sortBy[0]} and are not drawn.
        </p>
      )}
    </div>
  );
}
