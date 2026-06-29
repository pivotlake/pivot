import { useMemo, useRef, useState } from "react";
import CodeMirror, { EditorView, Prec, keymap } from "@uiw/react-codemirror";
import { sql } from "@codemirror/lang-sql";
import { runQuery, type QueryResult, type TableOverview } from "../api";

export default function Console({ tables }: { tables: TableOverview[] }) {
  const [sqlText, setSqlText] = useState("SELECT count(*) FROM otel_logs");
  const [result, setResult] = useState<QueryResult | null>(null);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const executeRef = useRef<() => void>(() => {});

  const execute = async () => {
    if (running || !sqlText.trim()) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    try {
      setResult(await runQuery(sqlText, controller.signal));
    } catch (e) {
      setResult({
        columns: [],
        rows: [],
        row_count: 0,
        elapsed_ms: 0,
        error: controller.signal.aborted
          ? "Query stopped"
          : e instanceof Error
            ? e.message
            : String(e),
      });
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };
  executeRef.current = execute;

  const stop = () => abortRef.current?.abort();

  // Feed the schema to the SQL completer: table names, and their columns after
  // `table.`. Rebuilds when the catalog changes.
  const schema = useMemo(() => {
    const s: Record<string, string[]> = {};
    for (const t of tables) s[t.name] = t.columns.map((c) => c.name);
    return s;
  }, [tables]);

  const extensions = useMemo(
    () => [
      sql({ schema, upperCaseKeywords: true }),
      EditorView.lineWrapping,
      // ⌘↵ / Ctrl↵ runs the query from inside the editor.
      Prec.highest(
        keymap.of([
          {
            key: "Mod-Enter",
            run: () => {
              executeRef.current();
              return true;
            },
          },
        ]),
      ),
    ],
    [schema],
  );

  return (
    <div className="console">
      <div className="editor-wrap">
        <CodeMirror
          value={sqlText}
          onChange={setSqlText}
          extensions={extensions}
          height="120px"
          basicSetup={{
            lineNumbers: false,
            foldGutter: false,
            highlightActiveLine: false,
            autocompletion: true,
          }}
        />
        <div className="editor-bar">
          {running ? (
            <button className="run stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="run" onClick={execute} disabled={!sqlText.trim()}>
              Run <kbd>⌘↵</kbd>
            </button>
          )}
          {running && <span className="run-meta">running…</span>}
          {!running && result && !result.error && (
            <span className="run-meta">
              {result.rows.length.toLocaleString()} rows · {result.elapsed_ms.toFixed(1)} ms
            </span>
          )}
        </div>
        {tables.length > 0 && (
          <div className="suggestions">
            {tables.slice(0, 8).map((t) => (
              <button
                key={t.name}
                className="suggestion"
                onClick={() => setSqlText(`SELECT * FROM "${t.name}" LIMIT 100`)}
              >
                {t.name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="results">
        {result?.error ? (
          <div className="query-error">{result.error}</div>
        ) : result ? (
          <ResultGrid result={result} />
        ) : (
          <div className="empty">
            <div>Run a query</div>
            <div className="sub">results render here · ⌘↵ to execute</div>
          </div>
        )}
      </div>
    </div>
  );
}

function ResultGrid({ result }: { result: QueryResult }) {
  if (result.columns.length === 0) {
    return (
      <div className="empty">
        <div>Done</div>
        <div className="sub">{result.row_count} rows affected · no result set</div>
      </div>
    );
  }
  return (
    <table className="grid">
      <thead>
        <tr>
          <th style={{ color: "var(--faint)" }}>#</th>
          {result.columns.map((c, i) => (
            <th key={i}>
              {c.name} <span className="type-tag">{c.col_type}</span>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {result.rows.map((row, r) => (
          <tr key={r}>
            <td style={{ color: "var(--faint)" }}>{r + 1}</td>
            {row.map((cell, c) => (
              <td key={c} className={cell !== null && isNumeric(cell) ? "mono num" : "mono"}>
                {cell === null ? <span style={{ color: "var(--faint)" }}>NULL</span> : cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function isNumeric(value: string): boolean {
  return value !== "" && !Number.isNaN(Number(value));
}
