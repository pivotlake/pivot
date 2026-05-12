// Code panel: source on the left, asm on the right.
// Counter columns on the RIGHT of the text. Numbers only — color/weight scale
// with intensity (subtle bg → hot red, normal → bold).

const { CATEGORIES, SOURCE_LINES, ASM_LINES, LINE_COUNTERS, asmCounters } = window.PERF_DATA;

const CAT_FOR_BAR = ['cycles', 'dram', 'l1', 'l2', 'l3'];
const CAT_HUES = Object.fromEntries(CATEGORIES.map(c => [c.id, c.hue]));

// MODE: 'percent' | 'raw' | 'weighted'
function formatValue(v, mode, cid) {
  if (v <= 0) return '';
  if (mode === 'percent')  return v.toFixed(1) + '%';
  if (mode === 'raw')      return Math.round(v * (cid === 'cycles' ? 12000 : 800)).toLocaleString();
  if (mode === 'weighted') return (v * (cid === 'cycles' ? 1.0 : (cid === 'dram' ? 4.5 : (cid === 'l3' ? 2.6 : (cid === 'l2' ? 1.6 : 1.1))))).toFixed(2);
  return v.toFixed(1);
}

function CounterCell({ value, cid, mode, max = 30 }) {
  const pct = Math.min(1, value / max);
  // Color: cool → red as it gets hotter. Weight: normal → 700.
  const bg = pct > 0
    ? `oklch(${0.99 - pct * 0.10} ${0.01 + pct * 0.13} ${28})`
    : 'oklch(0.985 0.002 250)';
  const fg = pct < 0.15
    ? 'oklch(0.45 0.01 250)'
    : `oklch(${0.4 - pct * 0.16} ${0.06 + pct * 0.16} 28)`;
  const weight = 400 + Math.round(pct * 300); // 400 → 700
  return (
    <div style={{
      width: '100%', height: 18, borderRadius: 3, background: bg,
      display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
      paddingRight: 6, fontSize: 10.5,
      fontFamily: 'ui-monospace, SFMono-Regular, monospace',
      color: fg, fontVariantNumeric: 'tabular-nums', fontWeight: weight,
    }}>
      {formatValue(value, mode, cid)}
    </div>
  );
}

const COUNTER_COL_W = 64;

function HeaderRow({ leftLabel, leftWidth, bodyLabel }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', height: 26,
      borderBottom: '1px solid oklch(0.92 0.005 250)',
      background: 'oklch(0.985 0.003 250)',
      fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
      color: 'oklch(0.5 0.01 250)', fontFamily: 'ui-monospace, monospace',
    }}>
      <div style={{ width: leftWidth, paddingLeft: 12, color: 'oklch(0.6 0.01 250)' }}>{leftLabel}</div>
      <div style={{ flex: 1, paddingLeft: 12 }}>{bodyLabel}</div>
      {CAT_FOR_BAR.map(cid => (
        <div key={cid} style={{
          width: COUNTER_COL_W, padding: '0 4px', textAlign: 'right',
          color: `oklch(0.5 0.06 ${CAT_HUES[cid]})`,
        }}>
          {CATEGORIES.find(c => c.id === cid).short}
        </div>
      ))}
    </div>
  );
}

function SourcePane({ activeSrc, onSelectSrc, fileLabel, mode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0 }}>
      <HeaderRow leftLabel="line" leftWidth={44} bodyLabel={fileLabel} />
      <div style={{ flex: 1, overflow: 'auto', fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 12 }}>
        {SOURCE_LINES.map(ln => {
          const c = LINE_COUNTERS[ln.n];
          const active = activeSrc === ln.n;
          const isHot = c && c.cycles > 12;
          return (
            <div
              key={ln.n}
              onClick={() => onSelectSrc(ln.n)}
              style={{
                display: 'flex', alignItems: 'center', minHeight: 22,
                background: active ? 'oklch(0.96 0.03 250)' : (isHot ? 'oklch(0.985 0.012 38)' : 'transparent'),
                borderLeft: active ? '2px solid oklch(0.6 0.18 250)' : '2px solid transparent',
                cursor: c ? 'pointer' : 'default',
              }}
            >
              <div style={{ width: 44, paddingLeft: 12, color: 'oklch(0.6 0.01 250)',
                fontVariantNumeric: 'tabular-nums', fontSize: 11 }}>{ln.n}</div>
              <pre style={{
                margin: 0, paddingLeft: 12, flex: 1, whiteSpace: 'pre',
                color: ln.text.trim().startsWith('//') ? 'oklch(0.55 0.04 140)' : 'oklch(0.2 0.01 250)',
              }}>{ln.text}</pre>
              {CAT_FOR_BAR.map(cid => (
                <div key={cid} style={{ width: COUNTER_COL_W, padding: '2px 4px' }}>
                  <CounterCell value={c?.[cid] || 0} cid={cid} mode={mode} />
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AsmPane({ activeSrc, onSelectAsm, selectedAsmKey, mode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0,
      borderLeft: '1px solid oklch(0.92 0.005 250)' }}>
      <HeaderRow leftLabel="addr" leftWidth={80} bodyLabel="disassembly" />
      <div style={{ flex: 1, overflow: 'auto', fontFamily: 'ui-monospace, SFMono-Regular, monospace', fontSize: 12 }}>
        {ASM_LINES.map((a, i) => {
          const key = a.addr + '|' + a.text;
          const c = asmCounters[key];
          const dim = activeSrc && a.src !== activeSrc;
          const isLabel = a.text.endsWith(':');
          const isSelected = selectedAsmKey === key;
          const isHot = c && c.cycles > 6;
          return (
            <div
              key={key + '-' + i}
              onClick={() => !isLabel && onSelectAsm(a)}
              style={{
                display: 'flex', alignItems: 'center', minHeight: 22,
                background: isSelected ? 'oklch(0.94 0.05 250)' :
                            (isHot && !dim ? 'oklch(0.985 0.012 38)' : 'transparent'),
                opacity: dim ? 0.32 : 1,
                borderLeft: isSelected ? '2px solid oklch(0.55 0.18 250)' : '2px solid transparent',
                cursor: isLabel ? 'default' : 'pointer',
              }}
            >
              <div style={{ width: 80, paddingLeft: 12,
                color: 'oklch(0.55 0.05 280)', fontVariantNumeric: 'tabular-nums', fontSize: 11,
              }}>{isLabel ? '' : a.addr}</div>
              <pre style={{
                margin: 0, paddingLeft: 12, flex: 1, whiteSpace: 'pre',
                color: isLabel ? 'oklch(0.45 0.06 295)' : 'oklch(0.22 0.01 250)',
                fontWeight: isLabel ? 600 : 400,
              }}>{a.text}</pre>
              {CAT_FOR_BAR.map(cid => (
                <div key={cid} style={{ width: COUNTER_COL_W, padding: '2px 4px' }}>
                  <CounterCell value={c?.[cid] || 0} cid={cid} mode={mode} max={20} />
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CodePanel({ activeSrc, setActiveSrc, selectedAsm, setSelectedAsm, fileLabel, mode }) {
  return (
    <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <SourcePane activeSrc={activeSrc} onSelectSrc={setActiveSrc} fileLabel={fileLabel} mode={mode} />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <AsmPane
          activeSrc={activeSrc}
          onSelectAsm={(a) => setSelectedAsm(a)}
          selectedAsmKey={selectedAsm ? selectedAsm.addr + '|' + selectedAsm.text : null}
          mode={mode}
        />
      </div>
    </div>
  );
}

window.CodePanel = CodePanel;
