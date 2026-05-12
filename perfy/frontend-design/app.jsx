// Main app — toolbar (no checkboxes), timeline with section dividers + per-cat
// disclosure carets, click-name → flame, click-flame → code.

const { TIME_BINS } = window.PERF_DATA;

const DEFAULTS = /*EDITMODE-BEGIN*/{
  "showFlame": true
}/*EDITMODE-END*/;

function Toolbar({ range, setRange }) {
  const span = range[1] - range[0];
  const isZoomed = span < TIME_BINS - 1;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '10px 16px',
      borderBottom: '1px solid oklch(0.93 0.005 250)', background: 'white',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 22, height: 22, borderRadius: 5,
          background: 'linear-gradient(135deg, oklch(0.78 0.18 95), oklch(0.55 0.18 295))',
        }} />
        <div style={{ fontWeight: 600, fontSize: 13, color: 'oklch(0.2 0.01 250)' }}>Perfy</div>
        <div style={{ fontSize: 11, color: 'oklch(0.55 0.01 250)', marginLeft: 6,
          fontFamily: 'ui-monospace, monospace' }}>
          query_engine.profile · 12.4 s · 16 cpus
        </div>
      </div>
      <div style={{ flex: 1 }} />
      <span style={{ fontSize: 11, color: 'oklch(0.55 0.01 250)' }}>
        Drag in the timeline to zoom · click ▸ to expand per-CPU · click a track name for flame
      </span>
      <button onClick={() => setRange([0, TIME_BINS - 1])} disabled={!isZoomed}
        style={{
          padding: '5px 10px', borderRadius: 5,
          border: '1px solid oklch(0.92 0.005 250)', background: 'white',
          color: isZoomed ? 'oklch(0.4 0.12 250)' : 'oklch(0.7 0.01 250)',
          fontSize: 12, cursor: isZoomed ? 'pointer' : 'default', fontFamily: 'inherit',
        }}>
        {isZoomed ? `↺ Reset zoom (${(span / TIME_BINS * 12.4).toFixed(2)}s)` : 'Full range'}
      </button>
    </div>
  );
}

function SegmentedToggle({ value, onChange, options }) {
  return (
    <div style={{
      display: 'inline-flex', borderRadius: 5,
      border: '1px solid oklch(0.9 0.005 250)', background: 'oklch(0.985 0.003 250)',
      padding: 2, gap: 2,
    }}>
      {options.map(opt => {
        const isSel = value === opt.id;
        return (
          <button
            key={opt.id}
            onClick={() => onChange(opt.id)}
            style={{
              border: 'none', background: isSel ? 'white' : 'transparent',
              boxShadow: isSel ? '0 1px 2px oklch(0.2 0.02 250 / 0.08)' : 'none',
              color: isSel ? 'oklch(0.25 0.08 250)' : 'oklch(0.5 0.01 250)',
              fontWeight: isSel ? 600 : 400,
              padding: '4px 10px', borderRadius: 3,
              fontSize: 11, cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function App() {
  const [tweaks, setTweak] = useTweaks(DEFAULTS);
  const [range, setRange] = React.useState([0, TIME_BINS - 1]);
  const [expandedCats, setExpandedCats] = React.useState(new Set());
  const [selectedCat, setSelectedCat] = React.useState(null);
  const [selectedFrame, setSelectedFrame] = React.useState(null);
  const [activeSrc, setActiveSrc] = React.useState(154);
  const [selectedAsm, setSelectedAsm] = React.useState(null);
  const [counterMode, setCounterMode] = React.useState('percent');

  function toggleExpand(catId) {
    setExpandedCats(prev => {
      const next = new Set(prev);
      if (next.has(catId)) next.delete(catId);
      else next.add(catId);
      return next;
    });
  }

  function onSelectFrame(cell) {
    setSelectedFrame(cell.label);
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
      color: 'oklch(0.2 0.01 250)',
      background: 'oklch(0.99 0.003 250)',
      position: 'relative',
    }}>
      <Toolbar range={range} setRange={setRange} />

      {/* Timeline */}
      <div data-screen-label="01 Timeline" style={{
        flex: selectedCat ? '0 0 auto' : 1,
        maxHeight: selectedCat ? '50vh' : 'none',
        overflow: 'auto',
        borderBottom: '1px solid oklch(0.92 0.005 250)',
      }}>
        <Timeline
          selectedCat={selectedCat}
          setSelectedCat={(c) => { setSelectedCat(c); }}
          expandedCats={expandedCats}
          toggleExpand={toggleExpand}
          range={range}
          setRange={setRange}
        />
      </div>

      {/* Flame graph appears once a category is selected */}
      {selectedCat && (
        <div data-screen-label="02 Flame" style={{ flex: selectedFrame ? '0 0 auto' : 1, overflow: 'auto' }}>
          <FlameGraph onSelectFrame={onSelectFrame} selectedFrame={selectedFrame} />
        </div>
      )}

      {/* Code panel — appears once a flame frame is clicked */}
      {selectedCat && selectedFrame && (
        <div data-screen-label="03 Code" style={{ flex: 1, minHeight: 0,
          borderTop: '1px solid oklch(0.92 0.005 250)',
          display: 'flex', flexDirection: 'column' }}>
          <div style={{
            padding: '8px 14px', borderBottom: '1px solid oklch(0.93 0.005 250)',
            display: 'flex', alignItems: 'center', gap: 12, background: 'white',
          }}>
            <span style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, color: 'oklch(0.45 0.01 250)' }}>
              Inside
            </span>
            <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 12.5, color: 'oklch(0.2 0.01 250)' }}>
              {selectedFrame}
            </span>
            <span style={{ fontSize: 11, color: 'oklch(0.55 0.01 250)', fontFamily: 'ui-monospace, monospace' }}>
              src/exec/hash_join.cpp
            </span>
            <div style={{ flex: 1 }} />
            <SegmentedToggle
              value={counterMode}
              onChange={setCounterMode}
              options={[
                { id: 'percent',  label: 'Percentage' },
                { id: 'raw',      label: 'Raw counts' },
                { id: 'weighted', label: 'Weighted' },
              ]}
            />
            <button onClick={() => setSelectedFrame(null)} style={{
              border: '1px solid oklch(0.9 0.005 250)', background: 'white',
              borderRadius: 4, padding: '3px 9px', fontSize: 11,
              color: 'oklch(0.45 0.01 250)', cursor: 'pointer',
            }}>Close</button>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <CodePanel
              activeSrc={activeSrc} setActiveSrc={setActiveSrc}
              selectedAsm={selectedAsm} setSelectedAsm={setSelectedAsm}
              fileLabel="src/exec/hash_join.cpp"
              mode={counterMode}
            />
          </div>
        </div>
      )}

      <DeepView asm={selectedAsm} onClose={() => setSelectedAsm(null)} />

      <TweaksPanel title="Tweaks">
        <TweakSection title="Layout">
          <TweakToggle label="Auto-show flame on track click"
            value={tweaks.showFlame} onChange={v => setTweak('showFlame', v)} />
        </TweakSection>
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
