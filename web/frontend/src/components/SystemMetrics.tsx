import type { SystemInfo } from "../api";
import { bytes, percent } from "../format";

export default function SystemMetrics({ system }: { system: SystemInfo | null }) {
  return (
    <div className="card">
      <div className="card-head">
        <h3>System</h3>
        <span className="section-label">live</span>
      </div>
      <div className="card-body">
        {!system ? (
          <div className="hint">metrics unavailable</div>
        ) : (
          <>
            <div className="metric">
              <div className="metric-top">
                <span className="label">CPU</span>
                <span className="value tnum">{percent(system.cpu_avg)}</span>
              </div>
              <div className="bar">
                <span style={{ width: `${Math.min(100, system.cpu_avg)}%` }} />
              </div>
              <div className="cores">
                {system.cpu_per_core.map((c, i) => (
                  <div className="core" key={i} title={`core ${i}: ${percent(c)}`}>
                    <span style={{ height: `${Math.min(100, c)}%` }} />
                  </div>
                ))}
              </div>
            </div>

            <div className="metric">
              <div className="metric-top">
                <span className="label">Memory</span>
                <span className="value tnum">
                  {bytes(system.mem_used)} <small>/ {bytes(system.mem_total)}</small>
                </span>
              </div>
              <div className="bar">
                <span
                  style={{
                    width: `${system.mem_total ? Math.min(100, (system.mem_used / system.mem_total) * 100) : 0}%`,
                  }}
                />
              </div>
              <div className="metric-top" style={{ marginTop: 8, marginBottom: 0 }}>
                <span className="label">pivotdb process</span>
                <span className="value tnum" style={{ fontSize: 14 }}>
                  {bytes(system.process_mem)}
                </span>
              </div>
            </div>

            {system.disks[0] && (
              <div className="disk-line">
                <div className="meta">
                  <span className="name">Disk · {system.disks[0].mount}</span>
                  <span className="tnum">
                    {bytes(system.disks[0].total - system.disks[0].available)} /{" "}
                    {bytes(system.disks[0].total)}
                  </span>
                </div>
                <div className="bar">
                  <span
                    style={{
                      width: `${system.disks[0].total ? Math.min(100, ((system.disks[0].total - system.disks[0].available) / system.disks[0].total) * 100) : 0}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
