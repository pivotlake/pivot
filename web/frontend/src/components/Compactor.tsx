import type { Compaction } from "../api";
import { ago, bytes, count } from "../format";

// `compaction` is non-null whenever the server has a compactor running (it's
// omitted only when started with `--compact-bytes 0`).
export default function Compactor({ compaction }: { compaction: Compaction | null }) {
  const c = compaction;
  return (
    <div className="card">
      <div className="card-head">
        <h3>
          <span className={`dot ${c ? "live" : ""}`} />
          Compactor
        </h3>
        <span className={`status ${c ? "on" : "off"}`}>{c ? "on" : "off"}</span>
      </div>
      <div className="card-body">
        {!c ? (
          <div className="hint">not running</div>
        ) : (
          <>
            <div className="stats">
              <div className="stat">
                <div className="k">Compactions</div>
                <div className="v tnum">{count(c.compactions)}</div>
              </div>
              <div className="stat">
                <div className="k">Merged → written</div>
                <div className="v tnum">
                  {count(c.files_merged_in)} → {count(c.files_written)}
                </div>
              </div>
              <div className="stat">
                <div className="k">Rewritten</div>
                <div className="v tnum">{bytes(c.bytes_written)}</div>
              </div>
              <div className="stat">
                <div className="k">Last sweep</div>
                <div className="v">{ago(c.last_run_unix_ms)}</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
