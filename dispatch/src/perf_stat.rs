use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

static ACTIVE: AtomicBool = AtomicBool::new(false);
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
            if ACTIVE.swap(true, Ordering::Relaxed) == false {
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
            if ACTIVE.swap(false, Ordering::Relaxed) == true {
                if perf_write("disable") {
                    let elapsed_ms = (now_ns() - START.load(Ordering::Relaxed)) as f64 / 1_000_000.0;
                    eprintln!("perf window: {elapsed_ms:.1}ms");
                }
            }
        }
    });
}