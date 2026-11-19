"""Machine-only: tag every block of the memory ring with the operator that took
it, and report twice a second how many pinned blocks each operator took.

usage: owner_probe.py <repository root>
"""
import os
import sys

root = os.path.expanduser(sys.argv[1]) + '/dispatch/src/'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


open(root + 'owner_probe.rs', 'w').write(r'''//! Machine-only probe: which operator took each block of the memory ring.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::memory::{MemoryBlockState, memory_ctx};

const OUTSIDE: &str = "outside an operator call";

thread_local! {
    static CURRENT: Cell<&'static str> = const { Cell::new(OUTSIDE) };
}

static OWNERS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static LAST_REPORT_MS: AtomicU64 = AtomicU64::new(0);

/// Restores the operator that was current before this one was entered.
pub struct Entered(&'static str);

impl Drop for Entered {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.0));
    }
}

/// Name `operator` as the one whose call is running on this thread.
pub fn enter(operator: &'static str) -> Entered {
    Entered(CURRENT.with(|current| current.replace(operator)))
}

static CALL_SITES: Mutex<Vec<(&'static str, u32, &'static str)>> = Mutex::new(Vec::new());

/// Record that the operator running on this thread took block `slot`. A
/// block taken outside an operator call is recorded by the code that took it.
#[track_caller]
pub fn tag(slot: usize) {
    let mut owner = CURRENT.with(|current| current.get());
    if owner == OUTSIDE {
        let location = std::panic::Location::caller();
        let mut call_sites = CALL_SITES.lock().unwrap();
        owner = match call_sites
            .iter()
            .find(|(file, line, _)| *file == location.file() && *line == location.line())
        {
            Some((_, _, name)) => name,
            None => {
                let name: &'static str = Box::leak(
                    format!("outside, taken at {}:{}", location.file(), location.line())
                        .into_boxed_str(),
                );
                call_sites.push((location.file(), location.line(), name));
                name
            }
        };
    }
    {
        let mut owners = OWNERS.lock().unwrap();
        if owners.len() <= slot {
            owners.resize(slot + 1, OUTSIDE);
        }
        owners[slot] = owner;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let last = LAST_REPORT_MS.load(Ordering::Relaxed);
    if now - last < 500
        || LAST_REPORT_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let blocks = memory_ctx().read_all_blocks();
    let owners = OWNERS.lock().unwrap();
    let mut pinned_by_owner: HashMap<&'static str, usize> = HashMap::new();
    let mut pinned = 0;
    for block in &blocks {
        if matches!(block.state, MemoryBlockState::Pinned) {
            pinned += 1;
            let owner = owners.get(block.slot).copied().unwrap_or(OUTSIDE);
            *pinned_by_owner.entry(owner).or_default() += 1;
        }
    }
    let mut counts: Vec<(&'static str, usize)> = pinned_by_owner.into_iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let listed: Vec<String> = counts
        .iter()
        .map(|(owner, count)| format!("{count}\t{owner}"))
        .collect();
    eprintln!("OWNERS t={now} pinned={pinned} ;; {}", listed.join(" ;; "));
}
''')

edit('lib.rs', [
    ('pub mod memory;\n', 'pub mod memory;\npub mod owner_probe;\n'),
])

enter = '        let _owner = crate::owner_probe::enter(std::any::type_name::<U>());\n'
edit('operations/unary/mod.rs', [
    (signature, signature + enter)
    for signature in [
        '    fn run_cpu_work(&mut self, io: &mut OperatorIO) -> super::Result<WorkStatus> {\n',
        '''    fn process_read_response(
        &mut self,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> super::Result<()> {
''',
        '    fn process_fs_write_response(&mut self, request: FsWriteRequest) -> super::Result<()> {\n',
        '    fn try_steal_work(&mut self, io: &mut OperatorIO) -> super::Result<WorkStatus> {\n',
        '    fn try_finish(&mut self) -> super::Result<FinishStatus> {\n',
        '    fn process_http_upload_response(&mut self, request: HttpUploadRequest) -> super::Result<()> {\n',
    ]
])

edit('memory/context.rs', [
    ('''    pub fn get_write_buffer(&self, prefer_zeroed: bool) -> WriteBuffer {
''', '''    #[track_caller]
    pub fn get_write_buffer(&self, prefer_zeroed: bool) -> WriteBuffer {
        let buffer = self.get_write_buffer_untagged(prefer_zeroed);
        crate::owner_probe::tag(buffer.slot_idx);
        buffer
    }

    fn get_write_buffer_untagged(&self, prefer_zeroed: bool) -> WriteBuffer {
'''),
])
print('owner probe added')
