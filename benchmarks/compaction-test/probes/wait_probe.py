"""Machine-only: add up how long the compaction spends in each state, as the
planner sees it, and print the totals once a second.

usage: wait_probe.py <repository root>
"""
import os
import sys

c = os.path.expanduser(sys.argv[1]) + '/parquet-engine/src/writing/compaction/'


def edit(name, pairs):
    s = open(c + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(c + name, 'w').write(s)


edit('memory_budget.rs', [
    ('''    /// Whether an output row group is being encoded or written.''',
     '''    /// Machine-only probe.
    pub(super) fn probe_state(&self) -> (usize, usize) {
        (
            self.reserved_bytes.load(Ordering::Acquire) >> 20,
            self.row_groups_in_flight.load(Ordering::Acquire),
        )
    }

    /// Whether an output row group is being encoded or written.'''),
])

edit('merge/planner.rs', [
    ('''    /// Send out the planned jobs, planning the next round whenever they are
    /// all out.''', '''    /// Machine-only probe: add the time since the last call to the state the
    /// compaction was in then, and print the totals once a second.
    fn probe_wait(&self) {
        use std::collections::BTreeMap;
        use std::sync::Mutex;
        use std::time::Instant;
        struct Totals {
            last: Instant,
            last_print: Instant,
            state: String,
            seconds: BTreeMap<String, f64>,
        }
        static TOTALS: Mutex<Option<Totals>> = Mutex::new(None);
        if self.files.is_empty() {
            return;
        }
        let bytes_per_row = if self.decoded_rows == 0 {
            self.first_bytes_per_row
        } else {
            self.decoded_bytes.div_ceil(self.decoded_rows)
        };
        let next_file = self
            .files
            .iter()
            .filter(|file| file.count_next_range_rows() > 0)
            .min_by_key(|file| file.count_rows_ahead());
        let no_room = next_file.is_some_and(|file| {
            !self
                .memory_budget
                .has_room_for(file.count_next_range_rows() * bytes_per_row)
        });
        let merge_waits = self
            .files
            .iter()
            .any(|file| !file.has_all_rows_arrived() && !file.has_ready_rows());
        let (reserved_mb, in_flight) = self.memory_budget.probe_state();
        let state = format!(
            "merge {}; budget {}; in flight {}",
            if merge_waits { "waits for rows" } else { "has rows of every file" },
            if no_room { "has no room" } else { "has room" },
            in_flight.min(6),
        );
        let now = Instant::now();
        let mut totals = TOTALS.lock().unwrap();
        let totals = totals.get_or_insert_with(|| Totals {
            last: now,
            last_print: now,
            state: state.clone(),
            seconds: BTreeMap::new(),
        });
        *totals.seconds.entry(totals.state.clone()).or_default() +=
            (now - totals.last).as_secs_f64();
        totals.last = now;
        totals.state = state;
        if (now - totals.last_print).as_secs_f64() >= 1.0 {
            totals.last_print = now;
            let listed: Vec<String> = totals
                .seconds
                .iter()
                .map(|(state, seconds)| format!("{seconds:.2}\\t{state}"))
                .collect();
            eprintln!("WAIT reserved_mb={reserved_mb} ;; {}", listed.join(" ;; "));
        }
    }

    /// Send out the planned jobs, planning the next round whenever they are
    /// all out.'''),
    ('''        self.plan_and_send_merge_jobs(sender)?;
        Ok(())
    }
''', '''        self.plan_and_send_merge_jobs(sender)?;
        self.probe_wait();
        Ok(())
    }
'''),
    ('''        // Memory given back since the last time may have made room.
        self.read_ahead();
        Ok(if self.plan_and_send_merge_jobs(sender)? {''', '''        // Memory given back since the last time may have made room.
        self.read_ahead();
        let sent = self.plan_and_send_merge_jobs(sender)?;
        self.probe_wait();
        Ok(if sent {'''),
])
print('wait probe added')
