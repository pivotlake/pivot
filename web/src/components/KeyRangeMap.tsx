import { useMemo, useState } from "react";
import { chooseBlockLabel, readFileKeys } from "../fileKeys";
import { formatMemory } from "../format";
import { compareBounds } from "../keyScale";
import { buildSortOrderLine, type Section } from "../sortOrderLine";
import type { TableFile } from "../tables";
import { placeInView, useZoom, WIDTH_ONLY, ZoomControls, ZoomSurface, type Span } from "../zoom";

// The least of the map's width a file is drawn with where it has the room,
// so a file holding one key is still a mark that can be seen and pointed at.
const NARROWEST_BAR = 0.006;
// The narrowest bar that has room for an outline around its fill.
const LEAST_OUTLINED_BAR_PX = 4;
// The height of a row drawn in full.
const ROW_HEIGHT_PX = 20;
// The tallest the map is drawn. More rows than fit in it at their full
// height are squeezed into it, and zooming in gives them their height back.
const TALLEST_PLOT_PX = 320;
// The share of its row's height a bar fills, and of it that is clear above
// the bar.
const BAR_HEIGHT_SHARE = 0.85;
const BAR_TOP_SHARE = 0.1;
// The shortest row whose bars have room for their keys.
const LEAST_LABELLED_ROW_PX = 15;
// The shortest row whose bars have room for an outline around their fill.
const LEAST_OUTLINED_ROW_PX = 6;
// How many of a table's sort columns the axis names before it trails off.
const NAMED_SORT_COLUMNS = 3;

// A file as it is drawn: from where its first row falls in the table's sort
// order to where its last does, 0 to 1, in one of the map's rows.
type Bar = Span & {
  file: TableFile;
  row: number;
  // How many of the leading sort columns the file holds one value of that
  // is a section, marked above the bar and not written on it.
  sectionDepth: number;
  // Where the next thing in its row starts, which is as far as the bar can
  // be widened to be seen. Past the line's end for the last in a row.
  roomEnd: number;
};

// A section as it is drawn: a mark across its stretch of the line, in the
// row above its files.
type SectionMark = Section & { row: number };

// A file or a section's mark, while the rows are packed. A file's depth is
// the number of sections it lies under. A mark's is half a step past the
// files that span its section, and half a step short of the files inside it.
type Packed = Span & {
  depth: number;
  drawn: { file: TableFile; sectionDepth: number } | { section: Section };
};

function readFileName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

// A partition's large files drawn in the table's sort order: each file a bar
// from its first row to its last. A file too narrow to be seen is drawn
// wider where its row has the room, and files side by side that are narrower
// still run together into a strip.
//
// Two bars one above the other are two files holding keys from the same
// stretch of the sort order, which a query for a key there has to open both
// of, and which a layout merge rewrites into files side by side. A partition
// whose large files are fully sorted is a single row.
//
// A value of a sort column that files hold alone is a section, as wide as
// those files are many. It is marked by a ruled line with the value on it,
// in the row where the separation happens: below the files that span the
// section, and above the files inside it, whose own sections are marked
// further down in turn. A bar's keys start below the sections it lies under.
// A double click on a section's mark zooms to the section.
//
// The map zooms into a part of the sort order. Files keep their rows as it
// does, and a bar wide enough for it has the file's keys written on it. A
// partition with more rows than the map has height for is drawn with all of
// them squeezed into it, and zooms into a part of its rows as well. A double
// click on a bar zooms to its file.
export function KeyRangeMap({ files, sortBy }: { files: TableFile[]; sortBy: string[] }) {
  const [hoveredPath, setHoveredPath] = useState<string>();

  const { bars, marks, rowCount, columns, line } = useMemo(() => {
    const drawable = files.filter((file) => file.bounds[sortBy[0]]);
    if (drawable.length === 0) {
      return { bars: [], marks: [], rowCount: 0, columns: [], line: undefined };
    }
    // The sort columns files are placed by: from the first, for as long as
    // every file drawn has bounds on them.
    const firstUnbounded = sortBy.findIndex((column) => drawable.some((file) => !file.bounds[column]));
    const columns = firstUnbounded === -1 ? sortBy : sortBy.slice(0, firstUnbounded);
    const line = buildSortOrderLine(drawable, columns);

    // Rows are packed by where things lie, so two files share a row exactly
    // when they do not overlap, however narrow they are. The shallowest go
    // first, and among those the widest.
    const packed: Packed[] = [
      ...drawable.map((file) => {
        const { start, end, sectionDepth } = line.placeFile(file);
        return { start, end, depth: sectionDepth, drawn: { file, sectionDepth } };
      }),
      ...line.sections.map((section) => ({
        start: section.start,
        end: section.end,
        depth: section.level + 0.5,
        drawn: { section },
      })),
    ].sort(
      (left, right) =>
        left.depth - right.depth ||
        right.end - right.start - (left.end - left.start) ||
        left.start - right.start,
    );

    // What each row holds so far. Each thing goes in the highest row it fits
    // in that is below everything shallower it overlaps: a section's mark
    // below the files that span the section, and its files below the mark.
    const rows: Packed[][] = [];
    for (const next of packed) {
      const isClearOf = (other: Span) => next.end <= other.start || other.end <= next.start;
      let row = 0;
      for (const [above, placed] of rows.entries()) {
        if (placed.some((other) => other.depth < next.depth && !isClearOf(other))) {
          row = above + 1;
        }
      }
      while (row < rows.length && !rows[row].every(isClearOf)) row += 1;
      if (row === rows.length) rows.push([]);
      rows[row].push(next);
    }

    const bars: Bar[] = [];
    const marks: SectionMark[] = [];
    for (const [row, placed] of rows.entries()) {
      placed.sort((left, right) => left.start - right.start);
      for (const [index, { start, end, drawn }] of placed.entries()) {
        if ("section" in drawn) {
          marks.push({ ...drawn.section, row });
        } else {
          const roomEnd = index + 1 < placed.length ? placed[index + 1].start : Number.POSITIVE_INFINITY;
          bars.push({ ...drawn, start, end, row, roomEnd });
        }
      }
    }

    return { bars, marks, rowCount: rows.length, columns, line };
  }, [files, sortBy]);

  const plotHeightPx = Math.min(rowCount * ROW_HEIGHT_PX, TALLEST_PLOT_PX);
  const isSqueezed = rowCount * ROW_HEIGHT_PX > TALLEST_PLOT_PX;
  const zoom = useZoom(isSqueezed ? (rowCount * ROW_HEIGHT_PX) / TALLEST_PLOT_PX : WIDTH_ONLY);

  if (!line) {
    return (
      <p className="page-note">
        None of these files records bounds on <code>{sortBy[0]}</code>, so they cannot be placed by key.
      </p>
    );
  }

  // The rows that are shown, counted from the top row, and the height each
  // is drawn with.
  const firstRowShown = zoom.shownDown.start * rowCount;
  const rowsShown = (zoom.shownDown.end - zoom.shownDown.start) * rowCount;
  const rowHeightPx = plotHeightPx / rowsShown;
  // The least a bar is widened to, in the line's own measure: a bar is
  // widened into the room before the next file of its row, and no further.
  const narrowestBar = NARROWEST_BAR * (zoom.shown.end - zoom.shown.start);
  const visibleBars = bars
    .map((bar) => ({
      bar,
      place: placeInView(zoom.shown, {
        start: bar.start,
        end: Math.max(bar.end, Math.min(bar.start + narrowestBar, bar.roomEnd)),
      }),
      topPx: (bar.row - firstRowShown) * rowHeightPx,
    }))
    .filter(({ place, topPx }) => place.width > 0 && topPx + rowHeightPx > 0 && topPx < plotHeightPx);
  const visibleMarks = marks
    .map((mark) => ({
      mark,
      place: placeInView(zoom.shown, mark),
      topPx: (mark.row - firstRowShown) * rowHeightPx,
    }))
    .filter(({ place, topPx }) => place.width > 0 && topPx + rowHeightPx > 0 && topPx < plotHeightPx);
  const hovered = visibleBars.find(({ bar }) => bar.file.path === hoveredPath);
  const [firstKeys, lastKeys] = line.nameEnds(zoom.shown);

  return (
    <div className="key-map">
      <ZoomControls zoom={zoom} />
      <ZoomSurface
        zoom={zoom}
        className="key-map-plot"
        style={{ height: plotHeightPx + 1 }}
        label={`${bars.length} large files in ${rowCount} ${rowCount === 1 ? "row" : "rows"} along the sort order, from ${firstKeys} to ${lastKeys}`}
      >
        {(widthPx) => (
          <>
            <div className="key-map-files">
              {visibleMarks.map(({ mark, place, topPx }) => (
                <div
                  key={`${mark.level} ${mark.start}`}
                  className={[
                    "key-map-section",
                    place.isCutLeft && "is-cut-left",
                    place.isCutRight && "is-cut-right",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={{
                    left: `${place.left * 100}%`,
                    width: `${place.width * 100}%`,
                    top: topPx,
                    height: rowHeightPx,
                  }}
                  title={`${mark.column} ${String(mark.value)}: ${mark.fileCount} ${mark.fileCount === 1 ? "file" : "files"}`}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    zoom.showStretch(mark, (mark.row + 0.5) / rowCount);
                  }}
                >
                  {rowHeightPx >= LEAST_LABELLED_ROW_PX && (
                    <span className="key-map-section-name">{String(mark.value)}</span>
                  )}
                </div>
              ))}
              {visibleBars.map(({ bar, place, topPx }) => {
                const isTooSmallForOutline =
                  rowHeightPx < LEAST_OUTLINED_ROW_PX || place.width * widthPx < LEAST_OUTLINED_BAR_PX;
                const label =
                  rowHeightPx >= LEAST_LABELLED_ROW_PX
                    ? chooseBlockLabel(
                        readFileKeys(bar.file, sortBy).slice(bar.sectionDepth),
                        place.width * widthPx,
                      )
                    : undefined;
                return (
                  <div
                    key={bar.file.path}
                    className={[
                      "key-map-file",
                      bar.file.path === hoveredPath && "is-outlined",
                      isTooSmallForOutline && "is-thin",
                      place.isCutLeft && "is-cut-left",
                      place.isCutRight && "is-cut-right",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    style={{
                      left: `${place.left * 100}%`,
                      width: `${place.width * 100}%`,
                      top: topPx + rowHeightPx * BAR_TOP_SHARE,
                      height: Math.max(rowHeightPx * BAR_HEIGHT_SHARE, 1),
                    }}
                    onMouseEnter={() => setHoveredPath(bar.file.path)}
                    onMouseLeave={() => setHoveredPath(undefined)}
                    onDoubleClick={(event) => {
                      event.stopPropagation();
                      zoom.showStretch(bar, (bar.row + 0.5) / rowCount);
                    }}
                  >
                    {label && <span className="file-label">{label}</span>}
                  </div>
                );
              })}
            </div>
            {rowsShown < rowCount && (
              <div
                className="key-map-rows-shown"
                style={{
                  top: `${zoom.shownDown.start * 100}%`,
                  height: `${(zoom.shownDown.end - zoom.shownDown.start) * 100}%`,
                }}
              />
            )}
            {hovered && (
              <div
                className="tooltip"
                role="tooltip"
                style={{
                  left: `${(hovered.place.left + hovered.place.width / 2) * 100}%`,
                  bottom: `calc(100% - ${Math.max(hovered.topPx, 0)}px + 8px)`,
                  transform: "translateX(-50%)",
                }}
              >
                <strong>{readFileName(hovered.bar.file.path)}</strong>
                {formatMemory(hovered.bar.file.bytes)}
                {readFileKeys(hovered.bar.file, sortBy).map(({ column, min, max }) => (
                  <span key={column} className="tooltip-line">
                    {compareBounds(min, max) === 0
                      ? `${column} ${String(min)}`
                      : `${column} ${String(min)} to ${String(max)}`}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </ZoomSurface>
      <div className="key-map-axis">
        <span>{firstKeys}</span>
        <span className="key-map-axis-name">
          {columns.length === 1
            ? columns[0]
            : `${columns[0]}, then ${columns.slice(1, NAMED_SORT_COLUMNS).join(", ")}${columns.length > NAMED_SORT_COLUMNS ? ", …" : ""}`}
        </span>
        <span>{lastKeys}</span>
      </div>
      {isSqueezed && (
        <p className="page-note">
          {rowCount} rows, drawn thinner than usual to fit. Zooming in opens them up, and dragging then moves
          up and down the rows as well.
        </p>
      )}
      {bars.length < files.length && (
        <p className="page-note">
          {files.length - bars.length} files record no bounds on {sortBy[0]} and are not drawn.
        </p>
      )}
    </div>
  );
}
