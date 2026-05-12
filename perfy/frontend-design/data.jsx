// Synthetic perf data for a database query engine workload.
// Deterministic so renders match across reloads.

const CATEGORIES = [
  { id: 'cycles', label: 'Cycles',     short: 'CYC',  hue: 38,   max: 1.0 },
  { id: 'dram',   label: 'DRAM',       short: 'DRAM', hue: 12,   max: 1.0 },
  { id: 'l1',     label: 'L1 miss',    short: 'L1',   hue: 220,  max: 1.0 },
  { id: 'l2',     label: 'L2 miss',    short: 'L2',   hue: 260,  max: 1.0 },
  { id: 'l3',     label: 'L3 miss',    short: 'L3',   hue: 295,  max: 1.0 },
  { id: 'memlat', label: 'Mem latency',short: 'LAT',  hue: 25,   max: 1.0, group: 'memory' },
  { id: 'memthr', label: 'Mem throughput', short: 'BW', hue: 160, max: 1.0, group: 'memory' },
];

const CPU_COUNT = 16;
const TIME_BINS = 240; // horizontal resolution of timeline

// Mulberry32-style seeded PRNG for determinism
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Smooth bumpy signal with hotspots around given centers
function makeSignal(seed, hotspots, baseline = 0.08) {
  const r = rng(seed);
  const out = new Float32Array(TIME_BINS);
  for (let i = 0; i < TIME_BINS; i++) {
    let v = baseline + r() * 0.05;
    for (const h of hotspots) {
      const d = (i - h.center) / h.width;
      v += h.amp * Math.exp(-d * d);
    }
    // tiny micro-spikes
    if (r() < 0.04) v += 0.15 + r() * 0.3;
    out[i] = Math.min(1, Math.max(0, v));
  }
  // light box-blur to smooth
  const sm = new Float32Array(TIME_BINS);
  for (let i = 0; i < TIME_BINS; i++) {
    let s = 0, n = 0;
    for (let k = -2; k <= 2; k++) {
      const j = i + k;
      if (j >= 0 && j < TIME_BINS) { s += out[j]; n++; }
    }
    sm[i] = s / n;
  }
  return sm;
}

// Build per-CPU per-category signals. Different CPUs get slightly shifted hotspots
// to feel like real parallel DB execution: scan -> hash build -> probe -> aggregate.
function buildPerCpuData() {
  const data = {}; // data[catId] = Array<Float32Array length=TIME_BINS> indexed by cpu
  const phaseHotspots = [
    { center: 30,  width: 18, amp: 0.55 }, // scan
    { center: 95,  width: 14, amp: 0.85 }, // hash build (DRAM heavy)
    { center: 150, width: 22, amp: 0.7 },  // probe (L2/L3 heavy)
    { center: 205, width: 12, amp: 0.5 },  // aggregate
  ];
  for (const cat of CATEGORIES) {
    data[cat.id] = [];
    for (let cpu = 0; cpu < CPU_COUNT; cpu++) {
      const jitter = ((cpu * 7) % 11) - 5;
      const hs = phaseHotspots.map((h, i) => {
        let amp = h.amp;
        // Category-specific emphasis
        if (cat.id === 'dram' && i === 1) amp *= 1.4;
        if (cat.id === 'l3'   && i === 2) amp *= 1.3;
        if (cat.id === 'l2'   && i === 2) amp *= 1.15;
        if (cat.id === 'l1'   && i === 0) amp *= 1.2;
        if (cat.id === 'cycles') amp *= 1.0;
        // some CPUs are stragglers
        if (cpu % 5 === 0) amp *= 1.15;
        if (cpu === 7) amp *= 0.6; // idle-ish core
        return { ...h, center: h.center + jitter, amp };
      });
      const baseline = cat.id === 'cycles' ? 0.18 : 0.06;
      data[cat.id].push(makeSignal(cpu * 31 + cat.id.charCodeAt(0), hs, baseline));
    }
  }
  return data;
}

// Consolidated = average across CPUs per bin
function consolidate(data) {
  const out = {};
  for (const cat of CATEGORIES) {
    const arr = new Float32Array(TIME_BINS);
    for (let i = 0; i < TIME_BINS; i++) {
      let s = 0;
      for (let c = 0; c < CPU_COUNT; c++) s += data[cat.id][c][i];
      arr[i] = s / CPU_COUNT;
    }
    out[cat.id] = arr;
  }
  return out;
}

// Functions list (call frame samples) — for the panel under the timeline
const FUNCTIONS = [
  { name: 'HashJoin::probeBatch',          file: 'exec/hash_join.cpp',    self: 24.6, total: 38.2 },
  { name: 'HashTable::lookup',              file: 'exec/hash_table.cpp',   self: 18.1, total: 22.4 },
  { name: 'TableScan::nextBatch',           file: 'exec/table_scan.cpp',   self: 11.7, total: 15.3 },
  { name: 'AggregateOp::merge',             file: 'exec/aggregate.cpp',    self:  8.4, total: 11.0 },
  { name: 'BloomFilter::contains',          file: 'exec/bloom.cpp',        self:  6.2, total:  6.2 },
  { name: 'ColumnReader::decodeRLE',        file: 'storage/column.cpp',    self:  5.9, total:  9.1 },
  { name: 'memcmp',                          file: 'libc',                  self:  4.1, total:  4.1 },
  { name: 'PageCache::pin',                 file: 'storage/page_cache.cpp',self:  3.3, total:  4.7 },
  { name: 'std::__hash_bytes',              file: 'libstdc++',             self:  2.8, total:  2.8 },
  { name: 'Predicate::evaluate',            file: 'exec/predicate.cpp',    self:  2.1, total:  3.6 },
  { name: 'Allocator::allocateAligned',     file: 'mem/alloc.cpp',         self:  1.8, total:  2.4 },
  { name: 'Sort::partition',                file: 'exec/sort.cpp',         self:  1.4, total:  2.1 },
];

// Source + asm for HashJoin::probeBatch — the "current" function
const SOURCE_LINES = [
  { n: 142, text: '// Probe a batch of build-side keys against the hash table.' },
  { n: 143, text: '// Returns a vector of matching row offsets per probe key.' },
  { n: 144, text: 'void HashJoin::probeBatch(const KeyBatch& keys,' },
  { n: 145, text: '                          MatchVector& out) {' },
  { n: 146, text: '  const size_t n = keys.size();' },
  { n: 147, text: '  out.reserve(n);' },
  { n: 148, text: '' },
  { n: 149, text: '  for (size_t i = 0; i < n; ++i) {' },
  { n: 150, text: '    const uint64_t h = hash64(keys[i]);' },
  { n: 151, text: '    Bucket* b = &table_[h & mask_];' },
  { n: 152, text: '' },
  { n: 153, text: '    while (b) {' },
  { n: 154, text: '      if (b->hash == h && keysEqual(b->key, keys[i])) {' },
  { n: 155, text: '        out.push_back(b->rowId);' },
  { n: 156, text: '        break;' },
  { n: 157, text: '      }' },
  { n: 158, text: '      b = b->next;' },
  { n: 159, text: '    }' },
  { n: 160, text: '  }' },
  { n: 161, text: '}' },
];

// Per-source-line counter percentages (cycles/dram/l1/l2/l3) — chosen to tell a story:
// the pointer-chase line is the hot DRAM/L3 line.
const LINE_COUNTERS = {
  146: { cycles: 0.4, dram: 0.0, l1: 0.0, l2: 0.0, l3: 0.0 },
  147: { cycles: 1.2, dram: 0.1, l1: 0.4, l2: 0.0, l3: 0.0 },
  149: { cycles: 3.6, dram: 0.0, l1: 0.2, l2: 0.0, l3: 0.0 },
  150: { cycles: 8.4, dram: 0.3, l1: 4.1, l2: 1.2, l3: 0.4 },
  151: { cycles: 11.7, dram: 2.1, l1: 9.4, l2: 4.6, l3: 1.8 },
  153: { cycles: 4.2, dram: 0.4, l1: 1.8, l2: 0.6, l3: 0.2 },
  154: { cycles: 28.3, dram: 18.7, l1: 22.1, l2: 14.3, l3: 9.8 }, // pointer-chase
  155: { cycles: 6.1, dram: 0.6, l1: 2.4, l2: 0.8, l3: 0.3 },
  156: { cycles: 0.9, dram: 0.0, l1: 0.0, l2: 0.0, l3: 0.0 },
  158: { cycles: 14.2, dram: 9.3, l1: 11.6, l2: 7.2, l3: 4.1 }, // b = b->next
  160: { cycles: 1.8, dram: 0.0, l1: 0.0, l2: 0.0, l3: 0.0 },
};

// Mapped assembly. Each row optionally maps to a source line via .src.
const ASM_LINES = [
  { addr: '0x4a32c0', text: 'push   rbp',                          src: 144 },
  { addr: '0x4a32c1', text: 'mov    rbp, rsp',                     src: 144 },
  { addr: '0x4a32c4', text: 'push   r15',                          src: 144 },
  { addr: '0x4a32c6', text: 'push   r14',                          src: 144 },
  { addr: '0x4a32c8', text: 'mov    r14, rsi',                     src: 145 },
  { addr: '0x4a32cb', text: 'mov    r15, rdi',                     src: 145 },
  { addr: '0x4a32ce', text: 'mov    rax, [rsi+0x10]',              src: 146 },
  { addr: '0x4a32d2', text: 'mov    rcx, [rsi+0x8]',               src: 146 },
  { addr: '0x4a32d6', text: 'sub    rax, rcx',                     src: 146 },
  { addr: '0x4a32d9', text: 'sar    rax, 3',                       src: 146 },
  { addr: '0x4a32dd', text: 'mov    [rdi+0x18], rax',              src: 147 },
  { addr: '0x4a32e1', text: 'xor    r12d, r12d',                   src: 149 },
  { addr: '0x4a32e4', text: 'cmp    r12, rax',                     src: 149 },
  { addr: '0x4a32e7', text: 'jae    .Lexit',                       src: 149 },
  { addr: '0x4a32ed', text: '.Lloop:',                              src: 149 },
  { addr: '0x4a32ed', text: 'mov    rdi, [rcx+r12*8]',             src: 150 },
  { addr: '0x4a32f1', text: 'call   hash64',                       src: 150 },
  { addr: '0x4a32f6', text: 'mov    rdx, rax',                     src: 151 },
  { addr: '0x4a32f9', text: 'and    rax, [r15+0x28]',              src: 151 },
  { addr: '0x4a32fd', text: 'shl    rax, 5',                       src: 151 },
  { addr: '0x4a3301', text: 'add    rax, [r15+0x20]',              src: 151 },
  { addr: '0x4a3305', text: '.Lwhile:',                             src: 153 },
  { addr: '0x4a3305', text: 'test   rax, rax',                     src: 153 },
  { addr: '0x4a3308', text: 'je     .Lnext',                       src: 153 },
  { addr: '0x4a330e', text: 'cmp    [rax], rdx',                   src: 154 },
  { addr: '0x4a3311', text: 'jne    .Lchain',                      src: 154 },
  { addr: '0x4a3317', text: 'mov    rsi, [rax+0x8]',               src: 154 },
  { addr: '0x4a331b', text: 'mov    rdi, [rcx+r12*8]',             src: 154 },
  { addr: '0x4a331f', text: 'call   keysEqual',                    src: 154 },
  { addr: '0x4a3324', text: 'test   al, al',                       src: 154 },
  { addr: '0x4a3326', text: 'je     .Lchain',                      src: 154 },
  { addr: '0x4a332c', text: 'mov    rdi, [rax+0x10]',              src: 155 },
  { addr: '0x4a3330', text: 'mov    rsi, r14',                     src: 155 },
  { addr: '0x4a3333', text: 'call   MatchVector::push_back',       src: 155 },
  { addr: '0x4a3338', text: 'jmp    .Lcont',                       src: 156 },
  { addr: '0x4a333d', text: '.Lchain:',                             src: 158 },
  { addr: '0x4a333d', text: 'mov    rax, [rax+0x18]',              src: 158 },
  { addr: '0x4a3341', text: 'jmp    .Lwhile',                      src: 158 },
  { addr: '0x4a3346', text: '.Lcont:',                              src: 160 },
  { addr: '0x4a3346', text: 'inc    r12',                          src: 160 },
  { addr: '0x4a3349', text: 'cmp    r12, [r15+0x18]',              src: 160 },
  { addr: '0x4a334d', text: 'jb     .Lloop',                       src: 160 },
  { addr: '0x4a3353', text: '.Lexit:',                              src: 161 },
  { addr: '0x4a3353', text: 'pop    r14',                          src: 161 },
  { addr: '0x4a3355', text: 'pop    r15',                          src: 161 },
  { addr: '0x4a3357', text: 'pop    rbp',                          src: 161 },
  { addr: '0x4a3358', text: 'ret',                                  src: 161 },
];

// Per-instruction counters — by addr. Lines without entries are ~0.
function buildAsmCounters() {
  const m = {};
  for (const a of ASM_LINES) {
    const lc = LINE_COUNTERS[a.src];
    if (!lc) continue;
    // distribute the source-line counter across its asm rows, weighted by how
    // many rows the source line maps to. The mov-from-pointer instructions get
    // the brunt of the misses.
    const rowsForSrc = ASM_LINES.filter(x => x.src === a.src);
    const isPtrLoad = /mov\s+\w+, \[/.test(a.text);
    const weight = isPtrLoad ? 2.0 : 0.5;
    const totalWeight = rowsForSrc.reduce((s, r) => s + (/mov\s+\w+, \[/.test(r.text) ? 2.0 : 0.5), 0);
    const f = weight / totalWeight;
    m[a.addr + '|' + a.text] = {
      cycles: lc.cycles * f,
      dram: lc.dram * f,
      l1: lc.l1 * f,
      l2: lc.l2 * f,
      l3: lc.l3 * f,
    };
  }
  return m;
}

// Deep-view metrics for a given asm instruction. Pointer-chase loads are scary.
function deepMetricsFor(asm) {
  const isPtrLoad = /mov\s+\w+, \[/.test(asm.text);
  const isCall = /^call\b/.test(asm.text);
  const isBranch = /^(jne|je|jb|jae|jmp)\b/.test(asm.text);
  const r = rng(asm.addr.split('').reduce((s, c) => s + c.charCodeAt(0), 0));
  const heat = isPtrLoad ? 0.95 : (isCall ? 0.4 : (isBranch ? 0.3 : 0.15));
  return {
    avgMabs:        +(heat * 9.4 + r() * 0.6).toFixed(2),
    tlbL1:          +(heat * 4.2 + r() * 0.4).toFixed(2),
    tlbL2:          +(heat * 1.1 + r() * 0.2).toFixed(2),
    storeBufOcc:    +(0.18 + r() * 0.12).toFixed(2),
    branchHit:      isBranch ? +(0.62 + r() * 0.3).toFixed(2) : 0.97,
    backendStall:   +(heat * 0.78 + r() * 0.05).toFixed(2),
    frontendStall:  +(0.04 + r() * 0.05).toFixed(2),
    avgLatency:     +(heat * 220 + 12 + r() * 8).toFixed(0),
    iterations:     Math.round(8000 + r() * 4000),
    samples:        Math.round(heat * 9000 + r() * 800),
  };
}

// Flame graph data — stacked frames per row. Each row is a depth level (top = root).
// Each frame: { x, w, label, hue?, hot? }. x/w are percentages of a row's width.
// Style: dense small text, yellow fill. Mirrors the screenshot vibe.
function buildFlameRows() {
  const rows = [];
  // Root row
  rows.push([
    { x: 0,  w: 22, label: 'std::thread::start' },
    { x: 22, w: 14, label: 'tokio::worker::run' },
    { x: 36, w: 38, label: 'Worker::run', hot: true },
    { x: 74, w: 18, label: 'Scheduler::poll' },
    { x: 92, w: 8,  label: 'epoll_wait' },
  ]);
  rows.push([
    { x: 0,  w: 8,  label: 'clock_nanosleep' },
    { x: 8,  w: 14, label: 'std::sys::thread::sleep' },
    { x: 22, w: 14, label: 'futex_wait' },
    { x: 36, w: 38, label: 'Worker::step_run', hot: true },
    { x: 74, w: 18, label: 'Operator::next' },
    { x: 92, w: 8,  label: 'fd_event_dispatch' },
  ]);
  rows.push([
    { x: 22, w: 14, label: 'park_thread' },
    { x: 36, w: 38, label: 'OperatorGraph::traverse_backwards', hot: true },
    { x: 74, w: 18, label: 'DataflowOp::ready_pull' },
    { x: 92, w: 8,  label: 'IO::poll' },
  ]);
  rows.push([
    { x: 22, w: 14, label: 'parking_lot::condvar' },
    { x: 36, w: 30, label: 'HashJoin::execute', hot: true },
    { x: 66, w: 8,  label: 'Aggregate::merge' },
    { x: 74, w: 12, label: 'TableScan::next' },
    { x: 86, w: 6,  label: 'Filter::eval' },
    { x: 92, w: 8,  label: 'IO::read' },
  ]);
  rows.push([
    { x: 36, w: 22, label: 'HashJoin::probeBatch', hot: true, target: true },
    { x: 58, w: 8,  label: 'HashJoin::buildPhase' },
    { x: 66, w: 8,  label: 'Aggregate::flush' },
    { x: 74, w: 7,  label: 'ColumnReader::nextBatch' },
    { x: 81, w: 5,  label: 'BloomFilter::contains' },
    { x: 86, w: 6,  label: 'Predicate::eval' },
    { x: 92, w: 8,  label: 'PageCache::pin' },
  ]);
  rows.push([
    { x: 36, w: 9,  label: 'hash64' },
    { x: 45, w: 13, label: 'HashTable::lookup', hot: true },
    { x: 58, w: 8,  label: 'HashTable::insert' },
    { x: 66, w: 4,  label: 'StateMap::merge' },
    { x: 70, w: 4,  label: 'Histogram::add' },
    { x: 74, w: 7,  label: 'ColumnReader::decodeRLE' },
    { x: 81, w: 5,  label: 'std::__hash_bytes' },
    { x: 86, w: 6,  label: 'Predicate::evaluate' },
    { x: 92, w: 4,  label: 'mmap_read' },
    { x: 96, w: 4,  label: 'memcpy' },
  ]);
  rows.push([
    { x: 36, w: 9,  label: 'std::__hash_bytes' },
    { x: 45, w: 9,  label: 'keysEqual' },
    { x: 54, w: 4,  label: 'memcmp' },
    { x: 58, w: 4,  label: 'Allocator::alloc' },
    { x: 62, w: 4,  label: 'Bucket::link' },
    { x: 66, w: 4,  label: 'memcpy' },
    { x: 70, w: 4,  label: 'memmove' },
    { x: 74, w: 4,  label: 'lz4_decompress' },
    { x: 78, w: 3,  label: 'memcpy' },
    { x: 81, w: 5,  label: 'CRC32' },
    { x: 86, w: 6,  label: 'cmp_inline' },
    { x: 92, w: 4,  label: 'pread' },
    { x: 96, w: 4,  label: '__memcpy_avx' },
  ]);
  return rows;
}

window.PERF_DATA = {
  flameRows: buildFlameRows(),
  CATEGORIES,
  CPU_COUNT,
  TIME_BINS,
  perCpu: buildPerCpuData(),
  consolidated: null, // filled below
  FUNCTIONS,
  SOURCE_LINES,
  ASM_LINES,
  LINE_COUNTERS,
  asmCounters: null,
  deepMetricsFor,
};
window.PERF_DATA.consolidated = consolidate(window.PERF_DATA.perCpu);
window.PERF_DATA.asmCounters = buildAsmCounters();
