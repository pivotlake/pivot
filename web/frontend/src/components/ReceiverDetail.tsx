import type { Ingest, TableOverview } from "../api";
import { ago, bytes, count, rate } from "../format";

interface Props {
  ingest: Ingest | null;
  tables: TableOverview[];
}

export default function ReceiverDetail({ ingest, tables }: Props) {
  if (!ingest) {
    return (
      <div className="empty">
        <div>Receiver</div>
        <div className="sub">no longer running</div>
      </div>
    );
  }

  const tableByName = new Map(tables.map((t) => [t.name, t]));

  return (
    <div className="detail">
      <div className="head">
        <h3>{ingest.addr}</h3>
        <span className="loc">{ingest.kind.toUpperCase()} receiver</span>
      </div>

      <div className="kv">
        <div className="item">
          <div className="k">Throughput</div>
          <div className={`v ${ingest.rows_per_sec > 0 ? "live" : ""}`}>
            {rate(ingest.rows_per_sec)}
          </div>
        </div>
        <div className="item">
          <div className="k">Rows received</div>
          <div className="v tnum">{count(ingest.rows)}</div>
        </div>
        <div className="item">
          <div className="k">Written</div>
          <div className="v tnum">{bytes(ingest.bytes)}</div>
        </div>
        <div className="item">
          <div className="k">Files flushed</div>
          <div className="v tnum">{count(ingest.files)}</div>
        </div>
        <div className="item">
          <div className="k">Flushes</div>
          <div className="v tnum">{count(ingest.flushes)}</div>
        </div>
        <div className="item">
          <div className="k">Flush trigger</div>
          <div className="v">
            {count(ingest.flush_rows)} rows / {ingest.flush_secs}s
          </div>
        </div>
        <div className="item">
          <div className="k">Last flush</div>
          <div className="v">{ago(ingest.last_flush_unix_ms)}</div>
        </div>
      </div>

      <div className="section-label" style={{ marginBottom: 10 }}>
        Signals · {ingest.signals.length}
      </div>
      <table className="grid">
        <thead>
          <tr>
            <th>Signal</th>
            <th>Table</th>
            <th style={{ textAlign: "right" }}>Rows</th>
            <th style={{ textAlign: "right" }}>Rate</th>
          </tr>
        </thead>
        <tbody>
          {ingest.signals.map((s, i) => {
            const t = tableByName.get(s.table);
            const live = (t?.rows_per_sec ?? 0) > 0;
            return (
              <tr key={i}>
                <td>{s.signal}</td>
                <td className="mono">{s.table}</td>
                <td className="num">{count(t?.row_count ?? null)}</td>
                <td className="num" style={live ? { color: "var(--teal)" } : undefined}>
                  {rate(t?.rows_per_sec ?? null)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
