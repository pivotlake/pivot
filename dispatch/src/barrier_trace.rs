//! Scratch instrumentation: records when each worker arrives at each stage
//! barrier and when each barrier opens, and prints a per-stage summary once
//! the pool is idle again. Enabled by `PIVOT_BARRIER_TRACE=1`.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Event {
    Built,
    Arrive,
    Open,
    Finished,
    Dropped,
    Mark,
}

/// Record a point event on the current worker.
pub fn mark(label: &'static str) {
    record(label, Event::Mark);
}

/// Pool-wide accumulated nanoseconds and counts for a few timed sections.
pub static TIMED_NS: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub static TIMED_COUNT: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub const TIMED_LABELS: [&str; 4] = ["cut_range", "range send", "cut consume", "decode consume"];

/// Add one timed section's duration to the pool-wide tally.
pub fn add_time(slot: usize, started: Instant) {
    if !enabled() {
        return;
    }
    let elapsed = started.elapsed().as_nanos() as u64;
    LOCAL_TIMED.with_borrow_mut(|local| {
        local[slot].0 += elapsed;
        local[slot].1 += 1;
    });
}

struct Record {
    at: Instant,
    worker: usize,
    stage: &'static str,
    event: Event,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
/// One uncontended record list per worker slot.
static RECORDS: OnceLock<Vec<Mutex<Vec<Record>>>> = OnceLock::new();
static ARMED: AtomicBool = AtomicBool::new(false);

fn records() -> &'static Vec<Mutex<Vec<Record>>> {
    RECORDS.get_or_init(|| (0..2048).map(|_| Mutex::new(Vec::new())).collect())
}

pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("PIVOT_BARRIER_TRACE").is_ok())
}

pub fn record(stage: &'static str, event: Event) {
    if !enabled() {
        return;
    }
    let worker = crate::worker::WORKER_IDX.get();
    records()[worker % 2048].lock().unwrap().push(Record {
        at: Instant::now(),
        worker,
        stage,
        event,
    });
    // Written once per query, not per event: a store per event would bounce
    // the line between every worker's core.
    if !ARMED.load(Ordering::Relaxed) {
        ARMED.store(true, Ordering::Relaxed);
    }
    if matches!(event, Event::Dropped) {
        flush_local_times();
    }
}

thread_local! {
    static LOCAL_TIMED: std::cell::RefCell<[(u64, u64); 4]> = const { std::cell::RefCell::new([(0, 0); 4]) };
}

fn flush_local_times() {
    LOCAL_TIMED.with_borrow_mut(|local| {
        for (slot, (ns, count)) in local.iter_mut().enumerate() {
            if *count > 0 {
                TIMED_NS[slot].fetch_add(*ns, Ordering::Relaxed);
                TIMED_COUNT[slot].fetch_add(*count, Ordering::Relaxed);
                *ns = 0;
                *count = 0;
            }
        }
    });
}

fn short(stage: &str) -> String {
    let s = stage.split('<').nth(1).unwrap_or(stage);
    let s = s.split(',').next().unwrap_or(s);
    let s = s.rsplit("::").next().unwrap_or(s);
    s.trim_end_matches('>').chars().take(28).collect()
}

/// Print the summary of everything recorded and clear it.
pub fn dump() {
    if !enabled() || !ARMED.swap(false, Ordering::Relaxed) {
        return;
    }
    let mut records: Vec<Record> = records()
        .iter()
        .flat_map(|slot| std::mem::take(&mut *slot.lock().unwrap()))
        .collect();
    if records.is_empty() {
        return;
    }
    records.sort_by_key(|r| r.at);
    let t0 = records[0].at;
    let us = |t: Instant| (t - t0).as_secs_f64() * 1e6;
    let mut stages: Vec<&'static str> = Vec::new();
    for r in &records {
        if matches!(r.event, Event::Arrive) && !stages.contains(&r.stage) {
            stages.push(r.stage);
        }
    }
    let built: Vec<f64> = records
        .iter()
        .filter(|r| matches!(r.event, Event::Built))
        .map(|r| us(r.at))
        .collect();
    let dropped: Vec<f64> = records
        .iter()
        .filter(|r| matches!(r.event, Event::Dropped))
        .map(|r| us(r.at))
        .collect();
    let mut out = String::new();
    out.push_str(&format!(
        "BARRIER TRACE: built first {:.0}us last {:.0}us (n={}); dropped first {:.0} last {:.0} (n={})\n",
        built.first().copied().unwrap_or(0.0),
        built.last().copied().unwrap_or(0.0),
        built.len(),
        dropped.first().copied().unwrap_or(0.0),
        dropped.last().copied().unwrap_or(0.0),
        dropped.len()
    ));
    out.push_str(&format!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}  late workers\n",
        "stage", "arr_p10", "arr_p50", "arr_p90", "arr_max", "open", "fin_p50", "fin_max"
    ));
    let mut last_open = 0.0;
    for stage in stages {
        let mut arrivals: Vec<(f64, usize)> = records
            .iter()
            .filter(|r| r.stage == stage && matches!(r.event, Event::Arrive))
            .map(|r| (us(r.at), r.worker))
            .collect();
        arrivals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let open = records
            .iter()
            .filter(|r| r.stage == stage && matches!(r.event, Event::Open))
            .map(|r| us(r.at))
            .next()
            .unwrap_or(0.0);
        let mut finished: Vec<f64> = records
            .iter()
            .filter(|r| r.stage == stage && matches!(r.event, Event::Finished))
            .map(|r| us(r.at))
            .collect();
        finished.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = arrivals.len().max(1);
        let pct = |p: usize| {
            arrivals
                .get((n * p / 100).min(n - 1))
                .map(|a| a.0)
                .unwrap_or(0.0)
        };
        let finished_count = finished.len().max(1);
        let late: Vec<String> = arrivals
            .iter()
            .rev()
            .take(4)
            .map(|(t, w)| format!("w{}@{:.0}", w, t))
            .collect();
        out.push_str(&format!(
            "{:<28} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0}  {}  (+{:.0} since prev open)\n",
            short(stage),
            pct(10),
            pct(50),
            pct(90),
            arrivals.last().map(|a| a.0).unwrap_or(0.0),
            open,
            finished.get(finished_count / 2).copied().unwrap_or(0.0),
            finished.last().copied().unwrap_or(0.0),
            late.join(" "),
            open - last_open
        ));
        // The straggler's point events since the previous stage opened.
        if let Some(&(arrival, straggler)) = arrivals.last() {
            let mut counts: Vec<(&'static str, usize, f64, f64)> = Vec::new();
            for r in records.iter().filter(|r| {
                r.worker == straggler
                    && matches!(r.event, Event::Mark)
                    && us(r.at) >= last_open
                    && us(r.at) <= arrival
            }) {
                let t = us(r.at);
                match counts.iter_mut().find(|c| c.0 == r.stage) {
                    Some(c) => {
                        c.1 += 1;
                        c.3 = t;
                    }
                    None => counts.push((r.stage, 1, t, t)),
                }
            }
            if !counts.is_empty() {
                let items: Vec<String> = counts
                    .iter()
                    .map(|(label, n, first, last)| format!("{label} x{n} [{first:.0}..{last:.0}]"))
                    .collect();
                out.push_str(&format!(
                    "    straggler w{straggler}: {}\n",
                    items.join(", ")
                ));
            }
        }
        last_open = open;
    }
    for (slot, label) in TIMED_LABELS.iter().enumerate() {
        let ns = TIMED_NS[slot].swap(0, Ordering::Relaxed);
        let count = TIMED_COUNT[slot].swap(0, Ordering::Relaxed);
        if count > 0 {
            out.push_str(&format!(
                "timed {label}: n={count} total {:.0}us mean {:.2}us\n",
                ns as f64 / 1e3,
                ns as f64 / 1e3 / count as f64
            ));
        }
    }
    eprint!("{out}");
}
