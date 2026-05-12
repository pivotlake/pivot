// Timeline: dense yellow line-graph tracks with dark-blue baseline tick marks.
// Click the disclosure caret on the left -> expand per-CPU rows.
// Click the row label/track body -> select that category (used by app to show flame).

const { CATEGORIES, CPU_COUNT, TIME_BINS, perCpu, consolidated } = window.PERF_DATA;

// Render a dense, jagged yellow signal with dark-blue baseline ticks.
// Style chosen to match the reference: thin vertical needles for peaks,
// flat near zero, lots of small variation, prominent baseline tick row.
function drawLineGraph(cvs, values, range) {
  const dpr = window.devicePixelRatio || 1;
  const w = cvs.clientWidth, h = cvs.clientHeight;
  if (w === 0 || h === 0) return;
  cvs.width = w * dpr; cvs.height = h * dpr;
  const ctx = cvs.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const start = range[0], end = range[1];
  const span = Math.max(1, end - start);
  const padTop = 1;
  const baselineY = h - 3;       // leave 3px below baseline for ticks
  const usable = baselineY - padTop;

  // Sample density: aim for ~1.2 px between samples (dense like screenshot)
  const samples = Math.max(span + 1, Math.floor(w / 1.1));
  const pts = new Array(samples);
  for (let i = 0; i < samples; i++) {
    const f = i / (samples - 1);
    const idx = start + f * span;
    const i0 = Math.floor(idx), i1 = Math.min(end, i0 + 1);
    const t = idx - i0;
    const v = (values[i0] || 0) * (1 - t) + (values[i1] || 0) * t;
    // Add tiny deterministic micro-jitter for the dense needle look
    const jitter = (Math.sin(idx * 12.9898) * 43758.5453) % 1;
    const noise = ((jitter + 1) % 1) * 0.06 - 0.03;
    const e = Math.pow(Math.max(0, Math.min(1, v + noise * Math.max(0.2, v))), 1.1);
    pts[i] = [f * w, padTop + (1 - e) * usable];
  }

  // Yellow fill under curve
  ctx.fillStyle = 'oklch(0.86 0.19 95)';
  ctx.beginPath();
  ctx.moveTo(0, baselineY);
  for (const [x, y] of pts) ctx.lineTo(x, y);
  ctx.lineTo(w, baselineY);
  ctx.closePath();
  ctx.fill();

  // Slightly darker outline on top edge for definition
  ctx.strokeStyle = 'oklch(0.78 0.19 92)';
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Dense dark-blue baseline ticks: one tick per bin where there's any value
  ctx.fillStyle = 'oklch(0.45 0.2 255)';
  for (let b = start; b <= end; b++) {
    const v = values[b] || 0;
    if (v < 0.03) continue;
    const f = (b - start) / span;
    const x = f * w;
    // tick height proportional to value, capped
    const th = Math.min(3, 1 + v * 2);
    ctx.fillRect(x - 0.5, baselineY, 1, th);
  }
}

function LineRow({ values, range }) {
  const ref = React.useRef(null);
  const draw = React.useCallback(() => {
    if (ref.current) drawLineGraph(ref.current, values, range);
  }, [values, range[0], range[1]]);
  React.useEffect(() => { draw(); }, [draw]);
  React.useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [draw]);
  return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />;
}

function Timeline({
  selectedCat, setSelectedCat,
  expandedCats, toggleExpand,
  range, setRange,
}) {
  const [hover, setHover] = React.useState(null);
  const [drag, setDrag] = React.useState(null);
  const hostRef = React.useRef(null);

  const labelWidth = 160;
  const trackHeight = 44;
  const subTrackHeight = 16;

  const perfCats = CATEGORIES.filter(c => !c.group);
  const memCats  = CATEGORIES.filter(c => c.group === 'memory');

  // Build flat row list with section dividers
  const rows = [];
  function pushSection(title, cats) {
    rows.push({ kind: 'section', title });
    for (const cat of cats) {
      rows.push({ kind: 'cat', cat, values: consolidated[cat.id], height: trackHeight });
      if (!cat.group && expandedCats.has(cat.id)) {
        for (let cpu = 0; cpu < CPU_COUNT; cpu++) {
          rows.push({ kind: 'cpu', cat, cpu, values: perCpu[cat.id][cpu], height: subTrackHeight });
        }
      }
    }
  }
  pushSection('Tracks with graphs', perfCats);
  pushSection('Memory', memCats);

  const wrapWidth = hostRef.current ? hostRef.current.clientWidth : 1200;
  const trackPxWidth = Math.max(50, wrapWidth - labelWidth);
  const span = range[1] - range[0];

  function pxToBin(px, rect) {
    const f = (px - rect.left - labelWidth) / Math.max(1, rect.width - labelWidth);
    return Math.round(range[0] + f * span);
  }
  function onMouseDown(e) {
    const rect = hostRef.current.getBoundingClientRect();
    if (e.clientX - rect.left < labelWidth) return;
    const startBin = pxToBin(e.clientX, rect);
    setDrag({ startBin, currentBin: startBin });
  }
  function onMouseMove(e) {
    const rect = hostRef.current.getBoundingClientRect();
    if (drag) { setDrag({ ...drag, currentBin: pxToBin(e.clientX, rect) }); return; }
    if (e.clientX - rect.left < labelWidth) { setHover(null); return; }
    const bin = pxToBin(e.clientX, rect);
    if (bin < range[0] || bin > range[1]) { setHover(null); return; }
    let y = e.clientY - rect.top - 28;
    let idx = -1;
    for (let i = 0; i < rows.length; i++) {
      const rh = rows[i].kind === 'section' ? 24 : rows[i].height;
      if (y < rh) { idx = i; break; }
      y -= rh;
    }
    if (idx < 0 || rows[idx].kind === 'section') { setHover(null); return; }
    setHover({ x: e.clientX - rect.left, y: e.clientY - rect.top, bin, row: idx });
  }
  function onMouseUp() {
    if (drag) {
      const a = Math.min(drag.startBin, drag.currentBin);
      const b = Math.max(drag.startBin, drag.currentBin);
      if (b - a > 2) setRange([Math.max(0, a), Math.min(TIME_BINS - 1, b)]);
      setDrag(null);
    }
  }
  function onDoubleClick() { setRange([0, TIME_BINS - 1]); }

  let dragOverlay = null;
  if (drag) {
    const lo = Math.min(drag.startBin, drag.currentBin);
    const hi = Math.max(drag.startBin, drag.currentBin);
    const x1 = labelWidth + ((lo - range[0]) / span) * trackPxWidth;
    const x2 = labelWidth + ((hi - range[0]) / span) * trackPxWidth;
    dragOverlay = (
      <div style={{
        position: 'absolute', top: 28, bottom: 0, left: x1, width: x2 - x1,
        background: 'oklch(0.65 0.16 250 / 0.14)',
        borderLeft: '1.5px solid oklch(0.55 0.18 250 / 0.7)',
        borderRight: '1.5px solid oklch(0.55 0.18 250 / 0.7)',
        pointerEvents: 'none',
      }} />
    );
  }

  let tip = null;
  if (hover && !drag && rows[hover.row]) {
    const row = rows[hover.row];
    const v = row.values[hover.bin] || 0;
    tip = (
      <div style={{
        position: 'absolute',
        left: Math.min(hover.x + 14, wrapWidth - 220),
        top: hover.y + 14,
        background: 'oklch(0.18 0.01 250)',
        color: 'oklch(0.97 0.005 250)',
        padding: '8px 10px', fontSize: 11,
        fontFamily: 'ui-monospace, monospace', borderRadius: 6,
        pointerEvents: 'none', zIndex: 10, minWidth: 180,
        boxShadow: '0 6px 24px oklch(0.2 0.02 250 / 0.18)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ color: 'oklch(0.7 0.02 250)' }}>
            {row.cat.label}{row.kind === 'cpu' ? ` · cpu${row.cpu.toString().padStart(2, '0')}` : ''}
          </span>
          <span style={{ color: 'oklch(0.85 0.16 95)' }}>●</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: 'oklch(0.6 0.01 250)' }}>t</span>
          <span>{(hover.bin / TIME_BINS * 12.4).toFixed(2)} s</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: 'oklch(0.6 0.01 250)' }}>intensity</span>
          <span>{(v * 100).toFixed(1)}%</span>
        </div>
      </div>
    );
  }

  const tickStep = Math.max(10, Math.round(span / 8));
  const ticks = [];
  for (let b = Math.ceil(range[0] / tickStep) * tickStep; b <= range[1]; b += tickStep) {
    const x = labelWidth + ((b - range[0]) / span) * trackPxWidth;
    ticks.push(
      <div key={b} style={{
        position: 'absolute', left: x, top: 0, bottom: 0,
        borderLeft: '1px solid oklch(0.94 0.005 250)',
      }}>
        <span style={{
          position: 'absolute', top: 6, left: 5, fontSize: 10,
          color: 'oklch(0.5 0.01 250)', fontFamily: 'ui-monospace, monospace',
        }}>{(b / TIME_BINS * 12.4).toFixed(1)}s</span>
      </div>
    );
  }

  return (
    <div
      ref={hostRef}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => setHover(null)}
      onDoubleClick={onDoubleClick}
      style={{
        position: 'relative', userSelect: 'none',
        cursor: drag ? 'ew-resize' : 'crosshair', overflow: 'hidden',
        background: 'white',
      }}
    >
      {/* Ruler */}
      <div style={{
        position: 'relative', height: 28,
        borderBottom: '1px solid oklch(0.92 0.005 250)',
        background: 'oklch(0.99 0.003 250)',
      }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, width: labelWidth, bottom: 0,
          borderRight: '1px solid oklch(0.92 0.005 250)',
          display: 'flex', alignItems: 'center', paddingLeft: 14,
          fontSize: 10, color: 'oklch(0.5 0.01 250)', textTransform: 'uppercase', letterSpacing: 0.5,
        }}>Tracks</div>
        {ticks}
      </div>

      {rows.map((row, i) => {
        if (row.kind === 'section') {
          return (
            <div key={`s-${row.title}`} style={{
              height: 24, padding: '0 14px', display: 'flex', alignItems: 'center',
              fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6,
              color: 'oklch(0.5 0.01 250)', background: 'oklch(0.97 0.005 250)',
              borderTop: '1px solid oklch(0.93 0.005 250)',
              borderBottom: '1px solid oklch(0.93 0.005 250)',
              fontWeight: 500,
            }}>
              {row.title}
            </div>
          );
        }
        const isCat = row.kind === 'cat';
        const isExpanded = isCat && expandedCats.has(row.cat.id);
        const isSelected = isCat && selectedCat === row.cat.id;
        return (
          <div key={isCat ? row.cat.id : `${row.cat.id}-${row.cpu}`} style={{
            display: 'flex', height: row.height, position: 'relative',
            borderBottom: isCat ? '1px solid oklch(0.95 0.005 250)' : 'none',
            background: isSelected ? 'oklch(0.97 0.04 250 / 0.5)' : 'transparent',
          }}>
            <div style={{
              width: labelWidth, flex: '0 0 auto',
              borderRight: '1px solid oklch(0.92 0.005 250)',
              display: 'flex', alignItems: 'center',
              paddingLeft: isCat ? 6 : 28,
              fontSize: isCat ? 12 : 10,
              fontFamily: isCat ? 'inherit' : 'ui-monospace, monospace',
              background: isSelected ? 'oklch(0.96 0.05 250 / 0.6)' : 'oklch(0.99 0.003 250)',
            }}>
              {isCat && !row.cat.group && (
                <button
                  onClick={(e) => { e.stopPropagation(); toggleExpand(row.cat.id); }}
                  title={isExpanded ? 'Collapse per-CPU' : 'Expand per-CPU'}
                  style={{
                    width: 18, height: 18, border: 'none', background: 'transparent',
                    cursor: 'pointer', color: 'oklch(0.45 0.01 250)',
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    padding: 0, marginRight: 2,
                  }}
                >
                  <span style={{
                    display: 'inline-block', width: 0, height: 0,
                    borderLeft: '4px solid transparent',
                    borderRight: '4px solid transparent',
                    borderTop: '5px solid currentColor',
                    transform: isExpanded ? 'rotate(0deg)' : 'rotate(-90deg)',
                    transition: 'transform 0.15s',
                  }} />
                </button>
              )}
              {isCat && row.cat.group && (
                <span style={{ display: 'inline-block', width: 20 }} />
              )}
              {isCat ? (
                <button
                  onClick={(e) => { e.stopPropagation(); setSelectedCat(row.cat.id); }}
                  style={{
                    flex: 1, display: 'inline-flex', alignItems: 'center', gap: 8,
                    border: 'none', background: 'transparent', cursor: 'pointer',
                    color: isSelected ? `oklch(0.3 0.16 ${row.cat.hue})` : 'oklch(0.25 0.01 250)',
                    fontFamily: 'inherit', fontSize: 12,
                    fontWeight: isSelected ? 600 : 500,
                    padding: '4px 6px', borderRadius: 4, textAlign: 'left',
                  }}
                >
                  <span style={{ width: 8, height: 8, borderRadius: 2,
                    background: `oklch(0.78 0.16 ${row.cat.hue})` }} />
                  <span>{row.cat.label}</span>
                </button>
              ) : (
                <span style={{ color: 'oklch(0.5 0.01 250)' }}>cpu{row.cpu.toString().padStart(2, '0')}</span>
              )}
            </div>
            <div style={{ flex: 1, position: 'relative',
              borderBottom: '1px solid oklch(0.96 0.005 250)' }}>
              <LineRow values={row.values} range={range} />
              {hover && hover.row === i && !drag && (
                <div style={{
                  position: 'absolute', top: 0, bottom: 0,
                  left: ((hover.bin - range[0]) / span) * 100 + '%',
                  width: 1, background: 'oklch(0.25 0.01 250 / 0.5)',
                  pointerEvents: 'none',
                }} />
              )}
            </div>
          </div>
        );
      })}

      {dragOverlay}
      {tip}
    </div>
  );
}

window.Timeline = Timeline;
