import { useMemo, useState } from "react";
import { postBackend, readBackend } from "./backend";
import { SqlEditor, type SqlColumn } from "./components/SqlEditor";
import { fetchTables } from "./tables";
import { usePolledReading } from "./usePolledReading";

// Where the browser keeps the statement being written, so it is still there
// after a visit to another page.
const STORED_STATEMENT_KEY = "pivot-web.statement";
const FIRST_STATEMENT =
  "SELECT datastore, schema, name, total_rows, bytes\nFROM system.tables\nORDER BY bytes DESC;";

// The server's own tables, which the backend's table list leaves out.
const SYSTEM_TABLES = ["datastores", "tables", "columns", "table_files", "memory_blocks", "connections"];

// What a statement run on the server produced.
type StatementResult = {
  // The result's column names, none for a statement that returns no rows.
  columns: string[];
  // The result's rows as the server renders them in text.
  rows: (string | null)[][];
  // Whether the result had more rows than the backend hands over.
  isCutShort: boolean;
  elapsedMs: number;
};

type Run =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: StatementResult }
  | { status: "failed"; reason: string };

// The browser's storage can be switched off or full; the editor then just
// starts from the first statement and forgets what is written in it.
function readStoredStatement(): string {
  try {
    return window.localStorage.getItem(STORED_STATEMENT_KEY) ?? FIRST_STATEMENT;
  } catch {
    return FIRST_STATEMENT;
  }
}

function storeStatement(statement: string): void {
  try {
    window.localStorage.setItem(STORED_STATEMENT_KEY, statement);
  } catch {
    // Nowhere to keep it: see `readStoredStatement`.
  }
}

// A name as SQL takes it: bare when it is plain lower-case and no keyword the
// server reserves, and quoted otherwise.
function quoteName(name: string): string {
  return /^[a-z_][a-z0-9_]*$/.test(name) && name !== "default" ? name : `"${name.replaceAll('"', '""')}"`;
}

export function QueryPage() {
  const [statement, setStatement] = useState(readStoredStatement);
  const [run, setRun] = useState<Run>({ status: "idle" });
  const tables = usePolledReading("tables", fetchTables);
  // The names the editor completes from. Without them it still completes
  // keywords, so a failed read is not reported here.
  const loadedColumns = usePolledReading("columns", () =>
    readBackend<{ columns: SqlColumn[] }>("/api/columns"),
  );
  // Every reading is a new list even when nothing changed; the editor is only
  // handed a new one when its contents differ, so a reading does not shut the
  // list of completions on someone using it.
  const columnsJson = loadedColumns.status === "live" ? JSON.stringify(loadedColumns.reading.columns) : "[]";
  const columns = useMemo(() => JSON.parse(columnsJson) as SqlColumn[], [columnsJson]);

  const runStatement = async (sql: string) => {
    setRun({ status: "running" });
    try {
      const result = await postBackend<StatementResult>("/api/query", { sql });
      setRun({ status: "done", result });
    } catch (error) {
      setRun({ status: "failed", reason: error instanceof Error ? error.message : String(error) });
    }
  };

  const writeStatement = (sql: string) => {
    setStatement(sql);
    storeStatement(sql);
  };

  // Picking a table writes a statement that reads its first rows, and runs it.
  const openTable = (qualifiedName: string) => {
    const sql = `SELECT * FROM ${qualifiedName} LIMIT 100;`;
    writeStatement(sql);
    runStatement(sql);
  };

  return (
    <div className="tables-page">
      <aside className="table-list">
        <div className="section-label">Tables</div>
        {tables.status === "failed" && (
          <p className="connections-notice is-failed" role="alert">
            Cannot read the server's tables: {tables.reason}
          </p>
        )}
        {tables.status === "live" &&
          tables.reading.tables.map((table) => (
            <button
              key={table.id}
              type="button"
              className="table-link"
              onClick={() =>
                openTable([table.datastore, table.schema, table.name].map(quoteName).join("."))
              }
            >
              <span className="table-link-name">{table.name}</span>
              <span className="table-link-facts">
                {table.datastore}.{table.schema} · {table.totalRows.toLocaleString()} rows
              </span>
            </button>
          ))}
        <div className="section-label table-list-divider">System</div>
        {SYSTEM_TABLES.map((name) => (
          <button key={name} type="button" className="table-link" onClick={() => openTable(`system.${name}`)}>
            <span className="table-link-name">{name}</span>
          </button>
        ))}
      </aside>

      <section className="query-pane">
        <SqlEditor
          value={statement}
          columns={columns}
          onChange={writeStatement}
          onRun={() => runStatement(statement)}
        />
        <div className="query-bar">
          <button
            type="button"
            className="query-run"
            disabled={run.status === "running" || statement.trim() === ""}
            onClick={() => runStatement(statement)}
          >
            Run
          </button>
          <span className="query-hint">Ctrl or ⌘ + Enter</span>
          <span className="query-status">
            {run.status === "running" && "Running…"}
            {run.status === "done" &&
              `${run.result.rows.length.toLocaleString()}${run.result.isCutShort ? "+" : ""} ${
                run.result.rows.length === 1 ? "row" : "rows"
              } · ${run.result.elapsedMs.toLocaleString()} ms`}
          </span>
        </div>

        {run.status === "failed" && (
          <p className="connections-notice is-failed query-failure" role="alert">
            {run.reason}
          </p>
        )}
        {run.status === "done" && run.result.isCutShort && (
          <p className="page-note">
            Showing the first {run.result.rows.length.toLocaleString()} rows; the result has more.
          </p>
        )}
        {run.status === "done" && run.result.columns.length === 0 && (
          <p className="page-note">The statement ran and returned no rows.</p>
        )}
        {run.status === "done" && run.result.columns.length > 0 && (
          <div className="query-result">
            <table className="file-list">
              <thead>
                <tr>
                  {run.result.columns.map((column, index) => (
                    <th key={index}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {run.result.rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((value, columnIndex) =>
                      value === null ? (
                        <td key={columnIndex} className="is-null">
                          NULL
                        </td>
                      ) : (
                        <td key={columnIndex} className="query-value" title={value}>
                          {value}
                        </td>
                      ),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
