// Functions list — under the timeline, above the code panel.
// Filterable. Click a function to "enter" it (loads code below).

const { FUNCTIONS } = window.PERF_DATA;

function FunctionList({ filter, setFilter, selected, onSelect, showFlame }) {
  const filtered = FUNCTIONS.filter(f =>
    !filter || f.name.toLowerCase().includes(filter.toLowerCase()) ||
    f.file.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0,
      background: 'oklch(0.995 0.002 250)' }}>
      <div style={{
        display: 'flex', alignItems: 'center', height: 36, padding: '0 14px',
        borderBottom: '1px solid oklch(0.93 0.005 250)', gap: 12,
      }}>
        <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, color: 'oklch(0.45 0.01 250)' }}>
          Call frames
        </div>
        <div style={{ flex: 1 }} />
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter functions…"
          style={{
            border: '1px solid oklch(0.9 0.005 250)', borderRadius: 5,
            padding: '4px 8px', fontSize: 11.5, width: 200,
            fontFamily: 'inherit', outline: 'none',
            background: 'white', color: 'oklch(0.2 0.01 250)',
          }}
        />
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '70px 70px 1fr 220px',
          padding: '6px 14px', fontSize: 10, color: 'oklch(0.5 0.01 250)',
          textTransform: 'uppercase', letterSpacing: 0.4,
          borderBottom: '1px solid oklch(0.95 0.005 250)',
          fontFamily: 'ui-monospace, monospace',
        }}>
          <div style={{ textAlign: 'right', paddingRight: 12 }}>Self %</div>
          <div style={{ textAlign: 'right', paddingRight: 12 }}>Total %</div>
          <div>Function</div>
          <div>{showFlame ? 'Time distribution' : 'File'}</div>
        </div>

        {filtered.map(f => {
          const isSel = selected === f.name;
          return (
            <div
              key={f.name}
              onClick={() => onSelect(f.name)}
              style={{
                display: 'grid', gridTemplateColumns: '70px 70px 1fr 220px',
                padding: '5px 14px', alignItems: 'center', cursor: 'pointer',
                background: isSel ? 'oklch(0.96 0.04 250)' : 'transparent',
                borderLeft: isSel ? '2px solid oklch(0.55 0.18 250)' : '2px solid transparent',
                fontSize: 12,
              }}
            >
              <div style={{
                textAlign: 'right', paddingRight: 12,
                fontFamily: 'ui-monospace, monospace', fontVariantNumeric: 'tabular-nums',
                color: f.self > 10 ? 'oklch(0.4 0.16 38)' : 'oklch(0.3 0.01 250)',
                fontWeight: f.self > 10 ? 600 : 400,
              }}>{f.self.toFixed(1)}</div>
              <div style={{
                textAlign: 'right', paddingRight: 12,
                fontFamily: 'ui-monospace, monospace', fontVariantNumeric: 'tabular-nums',
                color: 'oklch(0.4 0.01 250)',
              }}>{f.total.toFixed(1)}</div>
              <div style={{
                fontFamily: 'ui-monospace, monospace', fontSize: 12,
                color: 'oklch(0.2 0.01 250)', whiteSpace: 'nowrap',
                overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{f.name}</div>
              {showFlame ? (
                <FlameBar self={f.self} total={f.total} />
              ) : (
                <div style={{
                  fontFamily: 'ui-monospace, monospace', fontSize: 11,
                  color: 'oklch(0.55 0.01 250)', whiteSpace: 'nowrap',
                  overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{f.file}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FlameBar({ self, total }) {
  // Compact horizontal stacked bar: self time on top of children time.
  const max = 40;
  const selfPct = Math.min(1, self / max);
  const totalPct = Math.min(1, total / max);
  return (
    <div style={{ position: 'relative', height: 16, borderRadius: 3,
      background: 'oklch(0.97 0.003 250)', overflow: 'hidden' }}>
      <div style={{
        position: 'absolute', inset: 0, width: totalPct * 100 + '%',
        background: 'oklch(0.9 0.05 38)',
      }} />
      <div style={{
        position: 'absolute', inset: 0, width: selfPct * 100 + '%',
        background: 'oklch(0.7 0.16 38)',
      }} />
    </div>
  );
}

window.FunctionList = FunctionList;
