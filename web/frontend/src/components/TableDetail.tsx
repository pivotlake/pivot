import { forwardRef, useCallback, useEffect, useState, type ReactNode } from "react";
import { TableVirtuoso, type TableComponents, type TableProps } from "react-virtuoso";
import {
  fetchFiles,
  fetchRowGroups,
  type FileInfo,
  type QueryColumn,
  type TableOverview,
} from "../api";
import { bytes, count, rate } from "../format";
import { useInfiniteScroll } from "../useInfiniteScroll";

const PAGE = 50;

const VTable = forwardRef<HTMLTableElement, TableProps>((props, ref) => (
  <table {...props} ref={ref} className="grid vgrid" />
));

// Row component: when the panel is selectable (files), the row is clickable and
// highlights the selected item by a stable key (its path). State flows via
// Virtuoso `context` so the components object stays stable (no remount).
interface RowContext<T> {
  selectedKey?: string | null;
  selectKey?: (item: T) => string;
  onSelect?: (item: T) => void;
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const VRow = ({ context, item, ...rest }: any) => {
  const ctx = context as RowContext<unknown> | undefined;
  const selectable = !!ctx?.selectKey;
  const key = selectable ? ctx!.selectKey!(item) : null;
  const selected = selectable && key != null && key === ctx?.selectedKey;
  return (
    <tr
      {...rest}
      className={selectable ? `vrow ${selected ? "sel" : ""}` : undefined}
      onClick={selectable ? () => ctx!.onSelect!(item) : undefined}
    />
  );
};
const VIRTUOSO_COMPONENTS = { Table: VTable, TableRow: VRow };

interface Props {
  table: TableOverview | null;
  store: string;
}

export default function TableDetail({ table, store }: Props) {
  const name = table?.name ?? null;
  const [rgColumns, setRgColumns] = useState<QueryColumn[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  // Clear the file filter when switching tables.
  useEffect(() => {
    setSelectedFile(null);
  }, [name]);

  const loadFiles = useCallback(
    async (offset: number, limit: number) => {
      const page = await fetchFiles(name!, offset, limit);
      return { items: page.items, hasMore: offset + page.items.length < page.total };
    },
    [name],
  );

  const loadRowGroups = useCallback(
    async (offset: number, limit: number) => {
      const page = await fetchRowGroups(name!, offset, limit, selectedFile ?? undefined);
      if (offset === 0) setRgColumns(page.columns);
      return { items: page.rows, hasMore: page.has_more };
    },
    [name, selectedFile],
  );

  const files = useInfiniteScroll<FileInfo>(name, PAGE, loadFiles);
  // Reset row groups when the table OR the selected file changes.
  const rgKey = name == null ? null : `${name}::${selectedFile ?? "all"}`;
  const rowGroups = useInfiniteScroll<(string | null)[]>(rgKey, PAGE, loadRowGroups);

  const fileName = (f: FileInfo) => f.path;
  const toggleFile = (f: FileInfo) => setSelectedFile((cur) => (cur === f.path ? null : f.path));

  if (!table) {
    return (
      <div className="empty">
        <div>Select a table</div>
        <div className="sub">click a table in the graph to see its storage layout</div>
      </div>
    );
  }

  return (
    <div className="detail">
      <div className="head">
        <h3>{table.name}</h3>
        <span className="loc">
          {store} <span style={{ color: "var(--faint)" }}>/</span> {table.location}
        </span>
      </div>

      <div className="kv">
        <div className="item">
          <div className="k">Rows</div>
          <div className="v tnum">{count(table.row_count)}</div>
        </div>
        <div className="item">
          <div className="k">Insert rate</div>
          <div className={`v ${(table.rows_per_sec ?? 0) > 0 ? "live" : ""}`}>
            {rate(table.rows_per_sec)}
          </div>
        </div>
        <div className="item">
          <div className="k">Files</div>
          <div className="v tnum">{table.file_count}</div>
        </div>
        <div className="item">
          <div className="k">Size on disk</div>
          <div className="v tnum">{bytes(table.total_bytes)}</div>
        </div>
        <div className="item">
          <div className="k">Partition by</div>
          <div className="v">{table.partition_by.join(", ") || "-"}</div>
        </div>
        <div className="item">
          <div className="k">Sort by</div>
          <div className="v">{table.sort_by.join(", ") || "-"}</div>
        </div>
      </div>

      <div className="detail-grid">
        <div className="panel-col">
          <div className="section-label" style={{ marginBottom: 10 }}>
            Files &amp; placement · {table.file_count}
            <span className="hint" style={{ marginLeft: 8, textTransform: "none", letterSpacing: 0 }}>
              click a file to filter row groups
            </span>
          </div>
          <VirtualTable
            data={files.items}
            loading={files.loading}
            loadMore={files.loadMore}
            empty="no data files yet"
            selectedKey={selectedFile}
            selectKey={fileName}
            onSelect={toggleFile}
            header={
              <tr>
                <th>Path</th>
                <th style={{ width: 92, textAlign: "right" }}>Size</th>
              </tr>
            }
            row={(f) => (
              <>
                <td className="mono">{f.path}</td>
                <td className="num">{bytes(f.size)}</td>
              </>
            )}
          />
        </div>

        <div className="panel-col">
          <div className="section-label" style={{ marginBottom: 10 }}>
            Storage layout · row groups
            {selectedFile != null && (
              <button className="filter-chip" onClick={() => setSelectedFile(null)} title={selectedFile}>
                {selectedFile.split("/").pop()} ✕
              </button>
            )}
          </div>
          <VirtualTable
            data={rowGroups.items}
            loading={rowGroups.loading}
            loadMore={rowGroups.loadMore}
            empty="no row groups yet"
            header={
              <tr>
                {rgColumns.map((c, i) => (
                  <th key={i} style={{ textAlign: i === 0 ? "left" : "right" }}>
                    {c.name}
                  </th>
                ))}
              </tr>
            }
            row={(rowCells) => (
              <>
                {rowCells.map((cell, c) => (
                  <td key={c} className={c === 0 ? "mono" : "mono num"}>
                    {cell ?? ""}
                  </td>
                ))}
              </>
            )}
          />
        </div>
      </div>
    </div>
  );
}

interface VirtualTableProps<T> {
  data: T[];
  loading: boolean;
  loadMore: () => void;
  header: ReactNode;
  row: (item: T) => ReactNode;
  empty: string;
  selectedKey?: string | null;
  selectKey?: (item: T) => string;
  onSelect?: (item: T) => void;
}

function VirtualTable<T>({
  data,
  loading,
  loadMore,
  header,
  row,
  empty,
  selectedKey,
  selectKey,
  onSelect,
}: VirtualTableProps<T>) {
  return (
    <div className="scroll-panel">
      <TableVirtuoso
        style={{ height: "100%" }}
        data={data}
        context={{ selectedKey, selectKey, onSelect }}
        endReached={loadMore}
        increaseViewportBy={300}
        components={VIRTUOSO_COMPONENTS as TableComponents<T, RowContext<T>>}
        fixedHeaderContent={() => header}
        itemContent={(_index, item) => row(item)}
      />
      {data.length === 0 && !loading && <div className="scroll-empty">{empty}</div>}
      {loading && (
        <div className="scroll-loading">
          <span className="spinner" /> loading…
        </div>
      )}
    </div>
  );
}
