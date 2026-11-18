import { useEffect, useRef } from "react";
import {
  acceptCompletion,
  autocompletion,
  closeBrackets,
  closeBracketsKeymap,
  completionKeymap,
  type Completion,
  type CompletionSource,
} from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { sql, SQLDialect, type SQLNamespace } from "@codemirror/lang-sql";
import { bracketMatching, HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { Compartment, EditorState, Prec } from "@codemirror/state";
import { drawSelection, EditorView, keymap, lineNumbers } from "@codemirror/view";
import { tags } from "@lezer/highlight";

// A table's column as the editor completes it.
export type SqlColumn = {
  datastore: string;
  schema: string;
  table: string;
  name: string;
  sqlType: string;
};

// The SQL the editor colours and completes: the words of the statements a
// pivot server takes, not every word some dialect of SQL reserves, so the
// list of completions stays short enough to read.
const pivotSql = SQLDialect.define({
  keywords:
    "select from where group by having order limit offset distinct as on using join inner left right " +
    "full outer cross union all except intersect with and or not in is null like ilike between exists " +
    "case when then else end asc desc nulls first last over partition insert into values create table " +
    "schema user password drop if copy stdin format compact final vacuum set reset explain analyze " +
    "begin commit rollback cast true false",
  types:
    "tinyint smallint integer int bigint utinyint usmallint uinteger ubigint hugeint float real double " +
    "decimal numeric boolean bool varchar text string blob date time timestamp timestamptz interval " +
    "uuid variant json",
  builtin:
    "count sum min max avg floor round abs coalesce nullif greatest least lower upper length substring " +
    "concat generate_series read_parquet now",
  identifierQuotes: '"',
});

// Completes a column's bare name for every table the statement names
// anywhere in it, whichever way it names it. The statement is searched as
// text, so a column is also offered where the name is only mentioned, which
// errs towards offering.
function completeColumnsOfNamedTables(columns: SqlColumn[]): CompletionSource {
  const columnsByTable = new Map<string, SqlColumn[]>();
  for (const column of columns) {
    const tableColumns = columnsByTable.get(column.table) ?? [];
    tableColumns.push(column);
    columnsByTable.set(column.table, tableColumns);
  }
  return (context) => {
    const word = context.matchBefore(/\w+$/);
    // A name after a dot is completed from what is before the dot, not here.
    if (!word || context.state.sliceDoc(word.from - 1, word.from) === ".") {
      return null;
    }
    const namedInStatement = new Set(context.state.doc.toString().match(/\w+/g));
    const options: Completion[] = [...columnsByTable.entries()]
      .filter(([table]) => namedInStatement.has(table))
      .flatMap(([table, tableColumns]) =>
        tableColumns.map((column) => ({
          label: column.name,
          type: "property",
          detail: `${column.sqlType} · ${table}`,
          // Ahead of a keyword that starts the same way.
          boost: 50,
        })),
      );
    return options.length === 0 ? null : { from: word.from, options, validFor: /^\w*$/ };
  };
}

// The names the editor completes from, the way a statement can write them: a
// table by its full `datastore.schema.table` name, by `schema.table`, or
// alone, and a `system` table as `system.table` too. A table reached by a
// shorter name that two tables share completes to whichever was read last.
function buildNamespace(columns: SqlColumn[]): SQLNamespace {
  const namespace: { [name: string]: SQLNamespace } = {};
  const enter = (parent: { [name: string]: SQLNamespace }, name: string) => {
    const child = (parent[name] ?? {}) as { [name: string]: SQLNamespace };
    parent[name] = child;
    return child;
  };
  const columnsByTable = new Map<string, { column: SqlColumn; completions: Completion[] }>();
  for (const column of columns) {
    const key = `${column.datastore}\n${column.schema}\n${column.table}`;
    const table = columnsByTable.get(key) ?? { column, completions: [] };
    table.completions.push({ label: column.name, type: "property", detail: column.sqlType });
    columnsByTable.set(key, table);
  }
  for (const { column, completions } of columnsByTable.values()) {
    enter(enter(namespace, column.datastore), column.schema)[column.table] = completions;
    enter(namespace, column.schema)[column.table] = completions;
    if (column.datastore === "system") {
      enter(namespace, "system")[column.table] = completions;
    } else {
      namespace[column.table] = completions;
    }
  }
  return namespace;
}

// The editor's colours come from the page's own, so it follows the theme.
const editorTheme = EditorView.theme({
  "&": {
    height: "100%",
    color: "var(--text)",
    backgroundColor: "var(--surface)",
    fontSize: "13px",
  },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.6" },
  ".cm-content": { padding: "10px 0", caretColor: "var(--text)" },
  ".cm-cursor": { borderLeftColor: "var(--text)" },
  ".cm-gutters": {
    backgroundColor: "var(--surface)",
    color: "var(--text-muted)",
    border: "none",
  },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": {
    backgroundColor: "var(--sql-selection)",
  },
  ".cm-matchingBracket, &.cm-focused .cm-matchingBracket": {
    backgroundColor: "var(--sql-selection)",
    outline: "none",
  },
  ".cm-tooltip": {
    border: "1px solid var(--hairline-strong)",
    borderRadius: "6px",
    backgroundColor: "var(--surface)",
    color: "var(--text)",
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul": { fontFamily: "var(--font-mono)", fontSize: "12px" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li": { padding: "3px 10px 3px 6px" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]": {
    backgroundColor: "var(--accent)",
    color: "var(--accent-ink)",
  },
  ".cm-completionDetail": { marginLeft: "12px", color: "var(--text-muted)", fontStyle: "normal" },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected] .cm-completionDetail": {
    color: "inherit",
  },
});

const highlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: "var(--sql-keyword)", fontWeight: "700" },
  { tag: [tags.string, tags.special(tags.string)], color: "var(--sql-string)" },
  { tag: [tags.number, tags.bool, tags.null], color: "var(--sql-number)" },
  { tag: [tags.typeName, tags.standard(tags.name)], color: "var(--sql-type)" },
  { tag: [tags.lineComment, tags.blockComment], color: "var(--text-muted)", fontStyle: "italic" },
  { tag: [tags.operator, tags.punctuation], color: "var(--text-secondary)" },
]);

// A SQL editor: coloured as SQL, completing keywords, the names of the
// server's tables, and the columns of the tables a statement names, and
// running the statement on Ctrl or ⌘ + Enter.
export function SqlEditor({
  value,
  columns,
  onChange,
  onRun,
}: {
  value: string;
  // Every table's columns, for completing names from.
  columns: SqlColumn[];
  onChange: (value: string) => void;
  onRun: () => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView>();
  // The editor is built once and outlives renders, so it reaches the latest
  // callbacks through these.
  const latestOnChange = useRef(onChange);
  const latestOnRun = useRef(onRun);
  latestOnChange.current = onChange;
  latestOnRun.current = onRun;
  // The language is set apart so it can be given the tables' names once they
  // are read, and again whenever they change.
  const language = useRef(new Compartment());

  useEffect(() => {
    if (!host.current) {
      throw new Error("the SQL editor has nowhere to mount");
    }
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(),
          history(),
          drawSelection(),
          bracketMatching(),
          closeBrackets(),
          autocompletion(),
          Prec.highest(
            keymap.of([
              {
                key: "Mod-Enter",
                run: () => {
                  latestOnRun.current();
                  return true;
                },
              },
              // Tab takes the completion picked in the list while the list is
              // open. With no list open it is left alone, so it still moves
              // on from the editor to the next control.
              { key: "Tab", run: acceptCompletion },
            ]),
          ),
          keymap.of([...closeBracketsKeymap, ...completionKeymap, ...historyKeymap, ...defaultKeymap]),
          language.current.of(sql({ dialect: pivotSql, upperCaseKeywords: true })),
          syntaxHighlighting(highlightStyle),
          editorTheme,
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ "aria-label": "SQL statement" }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) {
              latestOnChange.current(update.state.doc.toString());
            }
          }),
        ],
      }),
    });
    view.current = editor;
    return () => editor.destroy();
    // Built once: later values and columns reach it through the effects below.
  }, []);

  // A statement written from outside the editor, such as by picking a table,
  // replaces what the editor holds.
  useEffect(() => {
    const editor = view.current;
    if (editor && editor.state.doc.toString() !== value) {
      editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
    }
  }, [value]);

  useEffect(() => {
    view.current?.dispatch({
      effects: language.current.reconfigure([
        sql({ dialect: pivotSql, upperCaseKeywords: true, schema: buildNamespace(columns) }),
        pivotSql.language.data.of({ autocomplete: completeColumnsOfNamedTables(columns) }),
      ]),
    });
  }, [columns]);

  return <div ref={host} className="query-editor" />;
}
