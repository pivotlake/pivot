"""Machine-only: count how many items each operator has taken in, and print
the counts once a second.

usage: count_probe.py <repository root>   (after time_operators.py)
"""
import os
import sys

p = os.path.expanduser(sys.argv[1]) + '/dispatch/src/operations/unary/mod.rs'
s = open(p).read()

helper = '''
/// Machine-only probe: count the items each operator takes in.
fn count_consumed<U>() {
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTS: Mutex<Option<HashMap<&'static str, usize>>> = Mutex::new(None);
    static LAST: AtomicU64 = AtomicU64::new(0);
    let mut counts = COUNTS.lock().unwrap();
    let counts = counts.get_or_insert_with(HashMap::new);
    *counts.entry(std::any::type_name::<U>()).or_default() += 1;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if LAST.swap(now, Ordering::Relaxed) != now {
        let mut listed: Vec<String> = counts
            .iter()
            .map(|(name, count)| format!("{count}\\t{name}"))
            .collect();
        listed.sort();
        eprintln!("CONSUMED ;; {}", listed.join(" ;; "));
    }
}
'''
anchor = '/// Wires a [`Unary`] transform to a receiver (input channel) and sender (output channel),'
assert s.count(anchor) == 1
s = s.replace(anchor, helper + '\n' + anchor)
for call in ['self.unary.consume(item, &mut *self.sender, io)?;', 'self.unary.consume(s, &mut *self.sender, io)?;']:
    assert s.count(call) == 1, call
    s = s.replace(call, 'count_consumed::<U>();\n        ' + call)
open(p, 'w').write(s)
print('count probe added')
