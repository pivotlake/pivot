use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static START: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

fn perf_write(cmd: &str) -> bool {
    OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open("/tmp/perf_ctl")
        .and_then(|mut f| write!(f, "{cmd}\n"))
        .is_ok()
}

pub fn perf_enable() {
    ENABLED.with(|e| {
        if !e.get() {
            e.set(true);
            if ACTIVE.fetch_add(1, Ordering::Relaxed) == 0 {
                if perf_write("enable") {
                    START.store(now_ns(), Ordering::Relaxed);
                }
            }
        }
    });
}

pub fn perf_disable() {
    ENABLED.with(|e| {
        if e.get() {
            e.set(false);
            if ACTIVE.fetch_sub(1, Ordering::Relaxed) == 1 {
                if perf_write("disable") {
                    let elapsed_ms = (now_ns() - START.load(Ordering::Relaxed)) as f64 / 1_000_000.0;
                    eprintln!("perf window: {elapsed_ms:.1}ms");
                }
            }
        }
    });
}