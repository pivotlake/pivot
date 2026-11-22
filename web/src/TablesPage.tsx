import { useMemo, useState } from "react";
import { KeyRangeMap } from "./components/KeyRangeMap";
import { SmallFilePile } from "./components/SmallFilePile";
import { isSmallFile, planSmallFileMerges } from "./compaction";
import { formatMemory } from "./format";
import {
  fetchTableFiles,
  fetchTables,
  type TableFile,
  type TableFilesReading,
  type TableSummary,
} from "./tables";
import { usePolledReading } from "./usePolledReading";

// How many files one of a table's two lists shows before it stops.
const LISTED_FILES_LIMIT = 300;

// One partition's files, split the way the server's compacter splits them,
// with the merges of small files a routine round would start.
type PartitionPlan = {
  partition: string;
  smallFiles: TableFile[];
  largeFiles: TableFile[];
  smallMerges: TableFile[][];
};

function planTable(reading: TableFilesReading): PartitionPlan[] {
  const filesByPartition = new Map<string, TableFile[]>();
  for (const file of reading.files) {
    const files = filesByPartition.get(file.partition) ?? [];
    files.push(file);
    filesByPartition.set(file.partition, files);
  }
  return [...filesByPartition.entries()]
    .map(([partition, files]) => ({
      partition,
      smallFiles: files.filter((file) => isSmallFile(file, reading.compaction)),
      largeFiles: files.filter((file) => !isSmallFile(file, reading.compaction)),
      smallMerges: planSmallFileMerges(files, reading.compaction),
    }))
    .sort(
      (left, right) =>
        right.smallMerges.length - left.smallMerges.length ||
        right.smallFiles.length +
          right.largeFiles.length -
          (left.smallFiles.length + left.largeFiles.length) ||
        (left.partition < right.partition ? -1 : 1),
    );
}

function sumBytes(files: TableFile[]): number {
  return files.reduce((bytes, file) => bytes + file.bytes, 0);
}

// One kind of a partition's files as a list: each file's size and its range
// on the sort columns, and for small files which merge, if any, would
// rewrite it.
function FileList({
  files,
  merges,
  sortBy,
}: {
  files: TableFile[];
  // The merges a routine round would start among `files`, for the list of a
  // kind of file the page can tell that of.
  merges?: TableFile[][];
  sortBy: string[];
}) {
  const mergeNumberByFile = new Map(
    (merges ?? []).flatMap((merge, mergeIndex) => merge.map((file) => [file, mergeIndex + 1] as const)),
  );
  const rangeColumns = sortBy.slice(0, 2);
  const listed = files.slice(0, LISTED_FILES_LIMIT);
  return (
    <>
      <div className="file-list-scroller">
        <table className="file-list">
          <thead>
            <tr>
              <th>File</th>
              <th className="is-number">Size</th>
              {rangeColumns.map((column) => (
                <th key={column}>{column}</th>
              ))}
              {merges && <th>Compaction</th>}
            </tr>
          </thead>
          <tbody>
            {listed.map((file) => {
              const mergeNumber = mergeNumberByFile.get(file);
              return (
                <tr key={file.path}>
                  <td className="file-path" title={file.path}>
                    {file.path.slice(file.path.lastIndexOf("/") + 1)}
                  </td>
                  <td className="is-number">{formatMemory(file.bytes)}</td>
                  {rangeColumns.map((column) => {
                    const bounds = file.bounds[column];
                    return (
                      <td key={column}>
                        {bounds ? `${String(bounds.min)} to ${String(bounds.max)}` : "no bounds"}
                      </td>
                    );
                  })}
                  {merges && <td>{mergeNumber === undefined ? "waiting" : `merge ${mergeNumber}`}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {files.length > listed.length && (
        <p className="page-note">
          Showing the first {listed.length} of {files.length} files.
        </p>
      )}
    </>
  );
}

function TableDetail({ table }: { table: TableSummary }) {
  const loaded = usePolledReading(table.id, () => fetchTableFiles(table.id));
  const reading = loaded.status === "live" ? loaded.reading : undefined;
  const plans = useMemo(() => (reading ? planTable(reading) : []), [reading]);

  // The partition shown: the one picked, while the table still has it, and
  // otherwise the one with the most merges waiting.
  const [pickedPartition, setPickedPartition] = useState<string>();
  const plan = plans.find((candidate) => candidate.partition === pickedPartition) ?? plans[0];

  const smallMergeCount = plans.reduce((count, candidate) => count + candidate.smallMerges.length, 0);

  return (
    <section className="table-detail">
      <header className="table-heading">
        <h1>
          <span className="table-qualifier">
            {table.datastore}.{table.schema}.
          </span>
          {table.name}
        </h1>
        <ul className="server-facts">
          <li>{table.totalRows.toLocaleString()} rows</li>
          <li>{formatMemory(table.bytes)}</li>
          <li>{table.fileCount} files</li>
          <li>{table.sortingKeys.length > 0 ? `sorted by ${table.sortingKeys.join(", ")}` : "not sorted"}</li>
          {table.partitionKey.length > 0 && <li>partitioned by {table.partitionKey.join(", ")}</li>}
          {reading && (
            <li>
              <strong>{smallMergeCount}</strong> small-file {smallMergeCount === 1 ? "merge" : "merges"} ready
            </li>
          )}
        </ul>
      </header>

      {loaded.status === "loading" && <p className="page-note">Reading the table's files…</p>}
      {loaded.status === "failed" && (
        <p className="connections-notice is-failed" role="alert">
          Cannot read the table's files: {loaded.reason}
        </p>
      )}

      {plans.length > 1 && (
        <div className="partition-picker">
          <span className="section-label">Partition</span>
          {plans.map((candidate) => (
            <button
              key={candidate.partition}
              type="button"
              aria-pressed={candidate === plan}
              onClick={() => setPickedPartition(candidate.partition)}
            >
              {candidate.partition || "unpartitioned"}
              <span className="partition-count">
                {candidate.smallFiles.length + candidate.largeFiles.length} files
                {candidate.smallMerges.length > 0 && ` · ${candidate.smallMerges.length} to merge`}
              </span>
            </button>
          ))}
        </div>
      )}

      {reading && !plan && <p className="page-note">This table has no files yet.</p>}

      {reading && plan && (
        <div className="file-panes">
          <section className="file-pane">
            <header className="pane-heading">
              <h2>Small files</h2>
              <span>
                {plan.smallFiles.length} below {formatMemory(Math.ceil(reading.compaction.targetBytes / 2))}{" "}
                · {formatMemory(sumBytes(plan.smallFiles))}
              </span>
            </header>
            <p className="pane-figure">
              <strong>{plan.smallMerges.length}</strong>{" "}
              {plan.smallMerges.length === 1 ? "merge" : "merges"} ready
            </p>
            <p className="pane-rule">
              Merged by size alone, whatever keys they hold: once enough of them add up to a full file.
            </p>
            {plan.smallFiles.length > 0 ? (
              <>
                <SmallFilePile
                  key={plan.partition}
                  files={plan.smallFiles}
                  merges={plan.smallMerges}
                  settings={reading.compaction}
                  sortBy={table.sortingKeys}
                />
                <FileList files={plan.smallFiles} merges={plan.smallMerges} sortBy={table.sortingKeys} />
              </>
            ) : (
              <p className="page-note">No small files.</p>
            )}
          </section>

          <section className="file-pane">
            <header className="pane-heading">
              <h2>Large files</h2>
              <span>
                {plan.largeFiles.length} at least {formatMemory(Math.ceil(reading.compaction.targetBytes / 2))}{" "}
                · {formatMemory(sumBytes(plan.largeFiles))}
              </span>
            </header>
            <p className="pane-rule is-leading">
              Merged for their layout, not their size: {reading.compaction.layoutCliqueSize} files
              whose keys all interleave are rewritten in key order into files that do not overlap.
              The server decides that from each file's row groups, which this page cannot see, so
              it shows where the files overlap and does not predict the merges.
            </p>
            {plan.largeFiles.length === 0 && <p className="page-note">No large files.</p>}
            {plan.largeFiles.length > 0 && table.sortingKeys.length === 0 && (
              <p className="page-note">
                This table has no sort key, so its large files have no key order to draw and are
                never merged again.
              </p>
            )}
            {plan.largeFiles.length > 0 && table.sortingKeys.length > 0 && (
              <KeyRangeMap key={plan.partition} files={plan.largeFiles} sortBy={table.sortingKeys} />
            )}
            {plan.largeFiles.length > 0 && <FileList files={plan.largeFiles} sortBy={table.sortingKeys} />}
          </section>
        </div>
      )}
    </section>
  );
}

export function TablesPage({ selectedTableId }: { selectedTableId?: string }) {
  const loaded = usePolledReading("tables", fetchTables);
  const tables = loaded.status === "live" ? loaded.reading.tables : [];
  // The table the address names, or the first in the list when it names none.
  const selected =
    selectedTableId === undefined ? tables[0] : tables.find((table) => table.id === selectedTableId);

  return (
    <div className="tables-page">
      <aside className="table-list">
        <div className="section-label">Tables</div>
        {loaded.status === "loading" && <p className="page-note">Reading the server's tables…</p>}
        {loaded.status === "failed" && (
          <p className="connections-notice is-failed" role="alert">
            Cannot read the server's tables: {loaded.reason}
          </p>
        )}
        {loaded.status === "live" && tables.length === 0 && (
          <p className="page-note">The server has no tables.</p>
        )}
        {tables.map((table) => (
          <a
            key={table.id}
            href={`#/tables/${table.id}`}
            className="table-link"
            aria-current={table === selected ? "page" : undefined}
          >
            <span className="table-link-name">{table.name}</span>
            <span className="table-link-facts">
              {table.datastore}.{table.schema} · {table.fileCount} files · {formatMemory(table.bytes)}
            </span>
          </a>
        ))}
      </aside>
      {selected ? (
        <TableDetail key={selected.id} table={selected} />
      ) : (
        <section className="table-detail">
          {selectedTableId !== undefined && loaded.status === "live" && (
            <p className="page-note">The server has no table with that id.</p>
          )}
        </section>
      )}
    </div>
  );
}
