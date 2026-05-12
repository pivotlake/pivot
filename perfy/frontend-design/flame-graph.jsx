// Flame graph in the screenshot's style: dense stacked yellow bars, tiny text.
// Top row is the root; each row below is one stack frame deeper.
// Click a frame -> opens code panel for that function.

const { flameRows } = window.PERF_DATA;

function FlameGraph({ onSelectFrame, selectedFrame }) {
  const ROW_H = 16;
  return (
    <div style={{
      background: 'white',
      borderTop: '1px solid oklch(0.92 0.005 250)',
      borderBottom: '1px solid oklch(0.92 0.005 250)',
      padding: '8px 0',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '0 14px 6px',
      }}>
        <span style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
          color: 'oklch(0.45 0.01 250)' }}>Flame graph</span>
        <span style={{ fontSize: 11, color: 'oklch(0.55 0.01 250)' }}>
          click a frame to open its code
        </span>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: 10, color: 'oklch(0.55 0.01 250)',
          fontFamily: 'ui-monospace, monospace' }}>
          {flameRows.length} levels · root → leaves
        </span>
      </div>
      <div style={{ position: 'relative', padding: '0 8px' }}>
        {flameRows.map((row, ri) => (
          <div key={ri} style={{ position: 'relative', height: ROW_H, marginBottom: 1 }}>
            {row.map((cell, ci) => {
              const isHot = cell.hot;
              const isTarget = cell.target;
              const isSel = selectedFrame === cell.label;
              return (
                <div
                  key={ci}
                  onClick={() => onSelectFrame(cell)}
                  title={cell.label}
                  style={{
                    position: 'absolute',
                    left: cell.x + '%', width: cell.w + '%',
                    top: 0, bottom: 0,
                    background: isSel
                      ? 'oklch(0.65 0.2 95)'
                      : isHot
                        ? 'oklch(0.85 0.18 95)'
                        : 'oklch(0.92 0.13 95)',
                    border: isTarget ? '1.5px solid oklch(0.45 0.18 38)' : '1px solid oklch(0.78 0.18 90)',
                    borderRadius: 1,
                    fontSize: 9.5,
                    fontFamily: 'ui-monospace, SFMono-Regular, monospace',
                    color: isSel ? 'white' : 'oklch(0.25 0.06 80)',
                    display: 'flex', alignItems: 'center', paddingLeft: 4,
                    overflow: 'hidden', whiteSpace: 'nowrap',
                    cursor: 'pointer',
                    boxSizing: 'border-box',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{cell.label}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

window.FlameGraph = FlameGraph;
