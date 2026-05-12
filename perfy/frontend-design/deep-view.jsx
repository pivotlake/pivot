// Deep view: side panel for a single instruction.
// Slides in from the right. Shows MABs, TLB misses, branch hit, stalls, etc.

const { deepMetricsFor, CATEGORIES, asmCounters } = window.PERF_DATA;

function StatBar({ label, value, max, unit, hue = 250, hint }) {
  const pct = Math.min(1, value / max);
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'oklch(0.45 0.01 250)', textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</span>
        <span style={{
          fontFamily: 'ui-monospace, monospace', fontSize: 13, fontWeight: 500,
          color: 'oklch(0.2 0.01 250)', fontVariantNumeric: 'tabular-nums',
        }}>
          {value}<span style={{ color: 'oklch(0.55 0.01 250)', fontWeight: 400, marginLeft: 2 }}>{unit}</span>
        </span>
      </div>
      <div style={{ position: 'relative', height: 6, borderRadius: 3, background: 'oklch(0.96 0.005 250)', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', inset: 0, width: pct * 100 + '%',
          background: `oklch(${0.85 - pct * 0.3} ${0.04 + pct * 0.16} ${hue})`,
          borderRadius: 3,
        }} />
      </div>
      {hint && <div style={{ fontSize: 10.5, color: 'oklch(0.55 0.01 250)', marginTop: 3 }}>{hint}</div>}
    </div>
  );
}

function DeepView({ asm, onClose }) {
  if (!asm) return null;
  const m = deepMetricsFor(asm);
  const counters = asmCounters[asm.addr + '|' + asm.text] || {};
  const isPtrLoad = /mov\s+\w+, \[/.test(asm.text);

  return (
    <div style={{
      position: 'absolute', top: 0, right: 0, bottom: 0, width: 360,
      background: 'oklch(0.995 0.002 250)',
      borderLeft: '1px solid oklch(0.9 0.005 250)',
      boxShadow: '-12px 0 32px oklch(0.2 0.02 250 / 0.06)',
      display: 'flex', flexDirection: 'column', zIndex: 30,
    }}>
      <div style={{
        padding: '14px 16px', borderBottom: '1px solid oklch(0.93 0.005 250)',
        display: 'flex', alignItems: 'flex-start', gap: 10,
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
            color: 'oklch(0.5 0.01 250)', marginBottom: 6 }}>
            Deep view · instruction
          </div>
          <div style={{
            fontFamily: 'ui-monospace, monospace', fontSize: 13,
            color: 'oklch(0.2 0.01 250)', display: 'flex', gap: 12, alignItems: 'baseline',
          }}>
            <span style={{ color: 'oklch(0.5 0.05 280)' }}>{asm.addr}</span>
            <span style={{ fontWeight: 500 }}>{asm.text}</span>
          </div>
          <div style={{ fontSize: 11, color: 'oklch(0.55 0.01 250)', marginTop: 4 }}>
            HashJoin::probeBatch · src/exec/hash_join.cpp:{asm.src}
          </div>
        </div>
        <button onClick={onClose} style={{
          border: '1px solid oklch(0.9 0.005 250)', background: 'white', cursor: 'pointer',
          width: 24, height: 24, borderRadius: 4, color: 'oklch(0.45 0.01 250)', fontSize: 14,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>×</button>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: '16px 18px' }}>
        {isPtrLoad && (
          <div style={{
            background: 'oklch(0.97 0.04 38)', border: '1px solid oklch(0.9 0.06 38)',
            padding: '8px 10px', borderRadius: 6, fontSize: 11.5,
            color: 'oklch(0.35 0.08 38)', marginBottom: 18,
            display: 'flex', gap: 8, alignItems: 'flex-start',
          }}>
            <span style={{ fontSize: 14, lineHeight: 1 }}>◆</span>
            <span>Pointer-chase load. High latency, hard for the prefetcher to anticipate. Consider linear probing or open-addressing.</span>
          </div>
        )}

        <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
          color: 'oklch(0.5 0.01 250)', marginBottom: 10 }}>Memory subsystem</div>
        <StatBar label="Avg MABs in flight" value={m.avgMabs} max={10} unit="" hue={12}
          hint="Out of 10 line-fill buffers" />
        <StatBar label="Avg memory latency" value={m.avgLatency} max={300} unit="cyc" hue={12} />
        <StatBar label="dTLB L1 miss rate" value={m.tlbL1} max={5} unit="%" hue={295} />
        <StatBar label="dTLB L2 miss rate" value={m.tlbL2} max={2} unit="%" hue={295} />
        <StatBar label="Store buffer occupancy" value={m.storeBufOcc} max={1} unit="" hue={220}
          hint="Fraction of cycles with ≥1 pending store" />

        <div style={{ height: 1, background: 'oklch(0.93 0.005 250)', margin: '8px 0 18px' }} />

        <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
          color: 'oklch(0.5 0.01 250)', marginBottom: 10 }}>Pipeline</div>
        <StatBar label="Backend stall" value={m.backendStall} max={1} unit="" hue={260}
          hint="Fraction of cycles backend cannot accept µops" />
        <StatBar label="Frontend stall" value={m.frontendStall} max={0.4} unit="" hue={220} />
        <StatBar label="Branch hit rate" value={m.branchHit} max={1} unit="" hue={140} />

        <div style={{ height: 1, background: 'oklch(0.93 0.005 250)', margin: '8px 0 18px' }} />

        <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
          color: 'oklch(0.5 0.01 250)', marginBottom: 10 }}>Counter share at this PC</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
          {['cycles','dram','l1','l2','l3'].map(cid => {
            const cat = CATEGORIES.find(c => c.id === cid);
            const v = counters[cid] || 0;
            return (
              <div key={cid} style={{
                padding: '8px 6px', borderRadius: 5,
                background: `oklch(0.98 ${0.02 + Math.min(0.06, v/40)} ${cat.hue})`,
                border: `1px solid oklch(0.92 0.04 ${cat.hue})`,
              }}>
                <div style={{ fontSize: 9.5, textTransform: 'uppercase', color: `oklch(0.45 0.08 ${cat.hue})`, letterSpacing: 0.4 }}>{cat.short}</div>
                <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13, color: 'oklch(0.2 0.01 250)', fontVariantNumeric: 'tabular-nums', marginTop: 2 }}>
                  {v.toFixed(1)}<span style={{ color: 'oklch(0.55 0.01 250)', fontSize: 10, marginLeft: 1 }}>%</span>
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ height: 1, background: 'oklch(0.93 0.005 250)', margin: '18px 0' }} />

        <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.5,
          color: 'oklch(0.5 0.01 250)', marginBottom: 10 }}>Sampling</div>
        <div style={{ display: 'flex', gap: 18, fontSize: 12, color: 'oklch(0.3 0.01 250)' }}>
          <div>
            <div style={{ color: 'oklch(0.55 0.01 250)', fontSize: 10.5, marginBottom: 2 }}>Samples</div>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontVariantNumeric: 'tabular-nums' }}>{m.samples.toLocaleString()}</div>
          </div>
          <div>
            <div style={{ color: 'oklch(0.55 0.01 250)', fontSize: 10.5, marginBottom: 2 }}>Iterations</div>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontVariantNumeric: 'tabular-nums' }}>{m.iterations.toLocaleString()}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.DeepView = DeepView;
