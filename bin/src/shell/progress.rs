//! Terminal feedback while the shell synchronously opens its datastore.

use std::io::{self, IsTerminal, Write};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

const FRAMES: [char; 4] = ['|', '/', '-', '\\'];
const TICK_INTERVAL: Duration = Duration::from_millis(100);

pub(super) fn show_opening_progress<T>(open: impl FnOnce() -> T) -> T {
    if !io::stderr().is_terminal() || std::env::var_os("TERM").is_some_and(|term| term == "dumb") {
        return open();
    }

    let started = Instant::now();
    draw_frame(FRAMES[0], Duration::ZERO);

    thread::scope(|scope| {
        let (stop, ticks) = mpsc::channel::<()>();
        // Opening blocks on metadata I/O, so animation needs its own thread.
        // Dropping the sender wakes it immediately, including during unwinding.
        scope.spawn(move || {
            let mut frame = 1;
            while matches!(
                ticks.recv_timeout(TICK_INTERVAL),
                Err(RecvTimeoutError::Timeout)
            ) {
                draw_frame(FRAMES[frame % FRAMES.len()], started.elapsed());
                frame += 1;
            }

            let mut stderr = io::stderr().lock();
            let _ = write!(stderr, "\r\x1b[2K");
            let _ = stderr.flush();
        });

        let result = open();
        drop(stop);
        result
    })
}

fn draw_frame(frame: char, elapsed: Duration) {
    let mut stderr = io::stderr().lock();
    // Cosmetic output must not turn a successful datastore open into an error.
    let _ = write!(
        stderr,
        "\r\x1b[2K{frame} Loading datastore... ({}s)",
        elapsed.as_secs()
    );
    let _ = stderr.flush();
}
