//! Worker parking and wake-up coordination.
//!
//! Each NUMA node has a [`WorkerWaker`] with one parking slot per worker.
//! Worker-local sends normally wake one same-node worker, targeted sends wake
//! their destination, and events that affect the whole pool use [`WakerSet`]
//! to reach every node. Worker threads access both through thread-local handles
//! installed during startup.

use crate::worker::WORKER_IDX;
#[cfg(any(test, feature = "test-util"))]
use crate::worker::{NUM_WORKERS, set_current_node};
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};
#[cfg(not(any(debug_assertions, feature = "unbounded-park")))]
use std::time::Duration;

thread_local! {
    /// This worker's node-local waker. Non-worker threads instead use an owned
    /// [`WakerSet`], such as the one held by a dispatched dataflow's handle.
    static WORKER_WAKER: Cell<*const WorkerWaker> = const { Cell::new(std::ptr::null()) };
    /// Owns the set referenced by `WAKER_SET` for the lifetime of this thread.
    static WAKER_SET_OWNER: RefCell<Option<Box<WakerSet>>> = const { RefCell::new(None) };
    static WAKER_SET: Cell<*const WakerSet> = const { Cell::new(std::ptr::null()) };
}

/// One worker's parking state within a node-local [`WorkerWaker`].
struct ParkSlot {
    /// Whether this worker is currently parked. Whoever swaps it `true -> false`
    /// (the parker on wake-up, or a notifier claiming the slot) decrements
    /// `parked_workers`.
    parked: AtomicBool,
    /// The worker's thread handle, registered once at worker startup.
    thread: OnceLock<Thread>,
    /// Whether this worker is blocked inside its IO ring's completion wait
    /// (see [`begin_ring_wait`](WorkerWaker::begin_ring_wait)). Claimed with
    /// the same `true -> false` discipline as `parked`; the claimer wakes the
    /// ring through `ring_wake` instead of unparking the thread.
    ring_parked: AtomicBool,
    /// How to interrupt this worker's ring wait, registered once at worker
    /// startup.
    ring_wake: OnceLock<crate::io::RingWakeHandle>,
}

/// How long a release build lets a worker sleep without a notification before
/// it wakes up and runs a pass anyway.
#[cfg(not(any(debug_assertions, feature = "unbounded-park")))]
pub const PARK_TIMEOUT: Duration = Duration::from_millis(100);

/// Block the calling worker until a notifier unparks it.
///
/// Every wake-up is supposed to come from a notifier, and a worker that never
/// hears one is a bug: some send forgot to notify. Debug builds, and any build
/// with the `unbounded-park` feature, keep the park unbounded so such a bug
/// shows up as a hang instead of hiding. A plain release build caps the sleep
/// at [`PARK_TIMEOUT`]: the same bug then costs one short stall per missed
/// wake rather than a query stuck for good. The caller already tolerates early
/// returns, so a timed-out park is handled the same way as a spurious unpark.
#[cfg(not(any(debug_assertions, feature = "unbounded-park")))]
fn park() {
    thread::park_timeout(PARK_TIMEOUT);
}

/// Block the calling worker until a notifier unparks it. See the release
/// variant for why this build never times out.
#[cfg(any(debug_assertions, feature = "unbounded-park"))]
fn park() {
    thread::park();
}

/// Wake-up coordination for the workers in one NUMA node.
///
/// A monotonic counter lets active workers detect notifications without a
/// syscall. Once idle, each worker parks on its own thread token. Per-worker
/// slots avoid the thundering herd produced by a shared condition variable:
/// ordinary sends can wake exactly one worker, targeted messages can wake their
/// recipient, and true broadcast events can wake the whole node.
///
/// A worker snapshots the wake count after each park. On its next attempt to
/// sleep, [`wait_if_unchanged`](Self::wait_if_unchanged) parks only if no
/// notification arrived during the intervening work pass.
pub struct WorkerWaker {
    /// Monotonic wrapping counter bumped on every notification. Workers poll it
    /// while spinning before they park.
    wake_count: AtomicU64,
    /// Monotonic counter bumped only by notifications that may interrupt a
    /// ring wait ([`notify`](Self::notify), [`notify_slot`](Self::notify_slot),
    /// [`notify_delegated`](Self::notify_delegated)). Data-availability sends
    /// ([`notify_one`](Self::notify_one)) leave it alone, so a worker with IO
    /// in flight can still block: it is guaranteed a kernel wakeup when its
    /// own IO completes and picks the new work up then.
    ring_wake_count: AtomicU64,
    /// Number of workers currently thread-parked. The no-sleeper send path
    /// checks this before scanning slots.
    parked_workers: AtomicUsize,
    /// Number of workers currently blocked in their IO ring wait.
    ring_parked_workers: AtomicUsize,
    slots: Box<[ParkSlot]>,
    /// Rotates [`notify_one`](Self::notify_one)'s scan start so wake-ups spread
    /// over parked workers instead of always choosing the lowest index.
    next_wake: AtomicUsize,
    /// Bumped on every delegated broadcast. The first worker woken observes the
    /// change and wakes its remaining parked siblings.
    broadcast_epoch: AtomicU64,
    /// One counter per worker, bumped by that worker's own data sends
    /// ([`notify_one`](Self::notify_one)). A send from every core of a node
    /// used to bump `wake_count`, and the read-modify-writes of 96 producers
    /// on one line serialised, costing a busy scan a fifth of its time. Each
    /// worker owns its line instead; whoever needs the total sums them.
    sent: Box<[Padded<AtomicU64>]>,
    /// Spinning workers' view of the total count, see [`SpinTicket`].
    relayed_wake_count: Padded<AtomicU64>,
    /// Whether some spinning worker currently relays `wake_count`.
    relay_claimed: Padded<AtomicBool>,
}

/// Keeps its content on a cache line of its own, so polling it shares no
/// line with anything a notifier writes.
#[repr(align(128))]
struct Padded<T>(T);

/// How many polls a spinner makes between checks for a vacant relay.
const RELAY_CLAIM_INTERVAL: u32 = 256;

/// A spinning worker's way of watching the wake count.
///
/// Every idle worker polls the wake count, so each one holds its cache line
/// in its own cache, and a notifier's increment has to invalidate every one
/// of those copies before it completes: on a node with many idle cores a
/// plain send costs microseconds precisely while the pool is mostly idle and
/// the send's latency matters most. So one spinner per node, the holder,
/// polls the real count and copies each change into a relay line that only
/// it writes, while every other spinner polls the relay. The holder gives the
/// role up when it stops spinning, and a spinner that then finds it vacant
/// takes it over at its next check, so a change is never relayed later than
/// one check interval. The relay only serves spinning: parking still checks
/// the real count, so no wake is lost through it.
pub struct SpinTicket<'a> {
    waker: &'a WorkerWaker,
    holds_relay: bool,
    polls_since_claim_check: u32,
}

impl SpinTicket<'_> {
    /// The wake count as this spinner sees it: exact for the holder, the
    /// holder's last relayed value for everyone else.
    pub fn wake_count(&mut self) -> u64 {
        if self.holds_relay {
            let now = self.waker.wake_count();
            if self.waker.relayed_wake_count.0.load(Ordering::Relaxed) != now {
                self.waker
                    .relayed_wake_count
                    .0
                    .store(now, Ordering::Release);
            }
            return now;
        }
        self.polls_since_claim_check += 1;
        if self.polls_since_claim_check >= RELAY_CLAIM_INTERVAL {
            self.polls_since_claim_check = 0;
            if self.waker.try_claim_relay() {
                self.holds_relay = true;
                return self.wake_count();
            }
        }
        self.waker.relayed_wake_count.0.load(Ordering::Acquire)
    }
}

impl Drop for SpinTicket<'_> {
    fn drop(&mut self) {
        if self.holds_relay {
            self.waker.relay_claimed.0.store(false, Ordering::Release);
        }
    }
}

impl WorkerWaker {
    /// Create a waker with one parking slot for each worker in the node.
    pub fn new(worker_count: usize) -> Self {
        Self {
            wake_count: AtomicU64::new(0),
            ring_wake_count: AtomicU64::new(0),
            parked_workers: AtomicUsize::new(0),
            ring_parked_workers: AtomicUsize::new(0),
            slots: (0..worker_count)
                .map(|_| ParkSlot {
                    parked: AtomicBool::new(false),
                    thread: OnceLock::new(),
                    ring_parked: AtomicBool::new(false),
                    ring_wake: OnceLock::new(),
                })
                .collect(),
            next_wake: AtomicUsize::new(0),
            broadcast_epoch: AtomicU64::new(0),
            sent: (0..worker_count)
                .map(|_| Padded(AtomicU64::new(0)))
                .collect(),
            relayed_wake_count: Padded(AtomicU64::new(0)),
            relay_claimed: Padded(AtomicBool::new(false)),
        }
    }

    /// Start watching the wake count as a spinner, taking the relay if it is
    /// vacant. See [`SpinTicket`].
    pub fn begin_spin(&self) -> SpinTicket<'_> {
        let holds_relay = self.try_claim_relay();
        SpinTicket {
            waker: self,
            holds_relay,
            polls_since_claim_check: 0,
        }
    }

    /// Claim the relay if it is vacant. The plain load first keeps the many
    /// spinners that find it taken from contending for the line.
    fn try_claim_relay(&self) -> bool {
        !self.relay_claimed.0.load(Ordering::Relaxed)
            && self
                .relay_claimed
                .0
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
    }

    /// Register the calling thread as worker `local_idx` in this node. Must be
    /// called once, on the worker thread, before its first park.
    pub fn register(&self, local_idx: usize) {
        self.slots[local_idx]
            .thread
            .set(thread::current())
            .expect("worker slot registered twice");
    }

    /// Register how to interrupt worker `local_idx`'s IO ring wait. Must be
    /// called once, before its first [`begin_ring_wait`](Self::begin_ring_wait).
    pub fn register_ring_waker(&self, local_idx: usize, handle: crate::io::RingWakeHandle) {
        self.slots[local_idx]
            .ring_wake
            .set(handle)
            .unwrap_or_else(|_| panic!("worker ring waker registered twice"));
    }

    /// Claim `slot` if it is parked and wake it: a worker blocked in its IO
    /// ring is woken through the ring, a thread-parked worker by unpark. The
    /// atomic claim makes concurrent notifiers choose different sleepers
    /// whenever possible.
    fn wake_slot(&self, slot: &ParkSlot) -> bool {
        if slot.ring_parked.swap(false, Ordering::SeqCst) {
            self.ring_parked_workers.fetch_sub(1, Ordering::SeqCst);
            slot.ring_wake
                .get()
                .expect("a ring-parked worker registered its ring waker")
                .wake();
            return true;
        }
        self.wake_thread_parked_slot(slot)
    }

    /// Claim `slot` only if its worker is thread-parked and unpark it. A worker
    /// blocked in its IO ring is left alone: it wakes on its own IO completion,
    /// and interrupting the ring costs an eventfd round-trip per wake, which
    /// data-availability sends fire far too often to afford.
    fn wake_thread_parked_slot(&self, slot: &ParkSlot) -> bool {
        if slot.parked.swap(false, Ordering::SeqCst) {
            self.parked_workers.fetch_sub(1, Ordering::SeqCst);
            if let Some(thread) = slot.thread.get() {
                thread.unpark();
            }
            return true;
        }
        false
    }

    /// Record a data-availability notification and wake one thread-parked
    /// worker, if any.
    ///
    /// This is used when one new stealable item needs one same-node consumer.
    /// Workers blocked in their IO ring wait are deliberately not interrupted:
    /// their own IO completion wakes them promptly, and they scan for new work
    /// then. Returns whether a parked worker was woken so callers can try
    /// another node when they need to grow the pool-wide working set.
    pub fn notify_one(&self) -> bool {
        // A worker records the send on its own counter; a thread outside the
        // pool has no counter and bumps the shared count. A worker on another
        // node lands on some counter of this node, which is only ever summed.
        let worker = WORKER_IDX.get();
        if worker == usize::MAX {
            self.wake_count.fetch_add(1, Ordering::SeqCst);
        } else {
            self.sent[worker % self.slots.len()]
                .0
                .fetch_add(1, Ordering::SeqCst);
        }
        if self.parked_workers.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let start = self.next_wake.fetch_add(1, Ordering::Relaxed);
        for i in 0..self.slots.len() {
            if self.wake_thread_parked_slot(&self.slots[(start + i) % self.slots.len()]) {
                return true;
            }
        }
        false
    }

    /// Record a notification and wake worker `local_idx` if it is parked,
    /// interrupting its ring wait when necessary. Used for messages addressed
    /// to a specific worker, which no other worker can handle for it.
    pub fn notify_slot(&self, local_idx: usize) {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        self.ring_wake_count.fetch_add(1, Ordering::SeqCst);
        if self.any_parked() {
            self.wake_slot(&self.slots[local_idx]);
        }
    }

    /// Record a notification and wake every parked worker in this node,
    /// interrupting ring waits.
    ///
    /// Used for events any local worker may be waiting on, such as cancellation
    /// or a sibling counter reaching zero.
    pub fn notify(&self) {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        self.ring_wake_count.fetch_add(1, Ordering::SeqCst);
        self.wake_all_parked();
    }

    /// Whether any worker is thread-parked or blocked in its ring wait.
    fn any_parked(&self) -> bool {
        self.parked_workers.load(Ordering::SeqCst) != 0
            || self.ring_parked_workers.load(Ordering::SeqCst) != 0
    }

    /// Unpark every currently parked worker without changing the wake counts.
    fn wake_all_parked(&self) {
        if !self.any_parked() {
            return;
        }
        for slot in &self.slots {
            self.wake_slot(slot);
        }
    }

    /// Record a broadcast while directly waking only one worker.
    ///
    /// Coordinator threads use this when dispatching a new dataflow. The first
    /// worker woken calls [`finish_delegated_wake`](Self::finish_delegated_wake)
    /// and unparks the rest of the node, avoiding a serial run of unpark syscalls
    /// on the unpinned coordinator.
    pub fn notify_delegated(&self) {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        self.ring_wake_count.fetch_add(1, Ordering::SeqCst);
        self.broadcast_epoch.fetch_add(1, Ordering::SeqCst);
        if !self.any_parked() {
            return;
        }
        // The delegate also scans in slot order, preserving the wake order of a
        // direct broadcast and therefore the downstream claim pattern.
        for slot in &self.slots {
            if self.wake_slot(slot) {
                return;
            }
        }
    }

    /// Current delegated-broadcast epoch, used to initialize a worker's memo.
    pub fn broadcast_epoch(&self) -> u64 {
        self.broadcast_epoch.load(Ordering::SeqCst)
    }

    /// If a delegated broadcast arrived since `last_seen`, update the memo and
    /// wake every remaining parked worker in the node.
    pub fn finish_delegated_wake(&self, last_seen: &mut u64) {
        let epoch = self.broadcast_epoch.load(Ordering::SeqCst);
        if epoch != *last_seen {
            *last_seen = epoch;
            self.notify();
        }
    }

    /// Current wake count: the shared count plus every worker's send count.
    /// Workers snapshot it around work passes and the relay holder polls it
    /// while spinning before a park.
    pub fn wake_count(&self) -> u64 {
        self.sent
            .iter()
            .map(|count| count.0.load(Ordering::SeqCst))
            .sum::<u64>()
            .wrapping_add(self.wake_count.load(Ordering::SeqCst))
    }

    /// Park worker `local_idx` if the wake count still equals `last_seen`.
    /// Returns the current count for use on the next park attempt.
    ///
    /// No wake is lost: notifiers increment a count before scanning slots,
    /// while this method publishes the parked slot before rechecking the sum.
    /// A concurrent notification therefore either changes the count observed
    /// here or claims the slot and unparks the thread. A leftover unpark token
    /// can only make a later park return early.
    ///
    /// In a plain release build the park itself returns after `PARK_TIMEOUT`
    /// even without a notification; debug builds and builds with the
    /// `unbounded-park` feature park until notified.
    pub fn wait_if_unchanged(&self, last_seen: u64, local_idx: usize) -> u64 {
        let slot = &self.slots[local_idx];
        slot.parked.store(true, Ordering::SeqCst);
        self.parked_workers.fetch_add(1, Ordering::SeqCst);
        if self.wake_count() != last_seen {
            // Withdraw the slot unless a notifier already claimed it and
            // decremented `parked_workers`.
            if slot.parked.swap(false, Ordering::SeqCst) {
                self.parked_workers.fetch_sub(1, Ordering::SeqCst);
            }
            return self.wake_count();
        }
        park();
        if slot.parked.swap(false, Ordering::SeqCst) {
            self.parked_workers.fetch_sub(1, Ordering::SeqCst);
        }
        self.wake_count()
    }

    /// Current ring wake count. Workers snapshot it around their ring waits the
    /// way [`wake_count`](Self::wake_count) is snapshotted around parks.
    pub fn ring_wake_count(&self) -> u64 {
        self.ring_wake_count.load(Ordering::SeqCst)
    }

    /// Publish worker `local_idx` as blocked in its IO ring wait, unless the
    /// ring wake count moved past `last_seen` first. Returns whether the caller
    /// should proceed into the blocking wait; `false` means a notification
    /// raced in and the worker should re-run its loop instead.
    ///
    /// Same no-lost-wake protocol as [`wait_if_unchanged`](Self::wait_if_unchanged),
    /// against [`ring_wake_count`](Self::ring_wake_count): ring-interrupting
    /// notifiers bump that count before scanning slots, this publishes the slot
    /// before rechecking it, so a concurrent notification either shows in the
    /// recheck or claims the slot and interrupts the ring (see
    /// [`wake_slot`](Self::wake_slot)). Data-availability sends do not bump the
    /// ring count and do not interrupt the wait: the caller only enters it with
    /// IO in flight, whose completion is a guaranteed wakeup. The caller must
    /// pair a `true` return with [`end_ring_wait`](Self::end_ring_wait) after
    /// the wait returns.
    pub fn begin_ring_wait(&self, local_idx: usize, last_seen: u64) -> bool {
        let slot = &self.slots[local_idx];
        slot.ring_parked.store(true, Ordering::SeqCst);
        self.ring_parked_workers.fetch_add(1, Ordering::SeqCst);
        if self.ring_wake_count.load(Ordering::SeqCst) != last_seen {
            // Withdraw the slot unless a notifier already claimed it (and woke
            // the ring; the spurious wake drains harmlessly).
            if slot.ring_parked.swap(false, Ordering::SeqCst) {
                self.ring_parked_workers.fetch_sub(1, Ordering::SeqCst);
            }
            return false;
        }
        true
    }

    /// Withdraw worker `local_idx`'s ring-wait publication after its wait
    /// returned, whether it was woken by an IO completion (the slot is still
    /// claimed here) or by a notifier (who already claimed it).
    pub fn end_ring_wait(&self, local_idx: usize) {
        let slot = &self.slots[local_idx];
        if slot.ring_parked.swap(false, Ordering::SeqCst) {
            self.ring_parked_workers.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// Routes wake-ups across the pool's NUMA-node wakers.
///
/// Node-local sends use [`worker_waker`]. Events that can unblock another node
/// use this set to target one global worker, grow the working set near a node,
/// or broadcast to the pool.
#[derive(Clone)]
pub struct WakerSet {
    node_wakers: Arc<[Arc<WorkerWaker>]>,
    workers_per_node: usize,
}

impl WakerSet {
    /// Build the routing table. `node_wakers` must contain one equally-sized
    /// worker group per NUMA node.
    pub fn new(node_wakers: impl Into<Arc<[Arc<WorkerWaker>]>>, workers_per_node: usize) -> Self {
        Self {
            node_wakers: node_wakers.into(),
            workers_per_node,
        }
    }

    pub(crate) fn workers_per_node(&self) -> usize {
        self.workers_per_node
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.node_wakers.len() * self.workers_per_node
    }

    /// Whether `other` wakes the same workers, i.e. both sets came from the
    /// same worker pool.
    pub fn wakes_same_pool(&self, other: &WakerSet) -> bool {
        Arc::ptr_eq(&self.node_wakers, &other.node_wakers)
    }

    /// Wake `worker`, expressed as a global worker index, if it is parked.
    pub fn notify_worker(&self, worker: usize) {
        self.node_wakers[worker / self.workers_per_node]
            .notify_slot(worker % self.workers_per_node);
    }

    /// Wake one parked worker, preferring `node` and checking subsequent nodes
    /// only when the preferred node has no sleeper.
    ///
    /// This grows the pool-wide working set without waking a whole node. It
    /// stays local while local sleepers exist, but can pull in a fully parked
    /// node once the producer's node is already awake.
    pub fn notify_one_near(&self, node: usize) {
        for offset in 0..self.node_wakers.len() {
            if self.node_wakers[(node + offset) % self.node_wakers.len()].notify_one() {
                return;
            }
        }
    }

    /// Wake every node group. Cheap when no workers are parked: one atomic
    /// increment and parked-count check per node.
    pub fn notify_all(&self) {
        for waker in self.node_wakers.iter() {
            waker.notify();
        }
    }

    /// Broadcast from a coordinator, paying one direct unpark per node and
    /// delegating each node's remaining wake-ups to that worker.
    pub fn notify_all_delegated(&self) {
        for waker in self.node_wakers.iter() {
            waker.notify_delegated();
        }
    }
}

/// Install this worker thread's node-local waker.
///
/// The caller must keep the `Arc` alive for the thread's lifetime. Production
/// workers own one and also install a [`WakerSet`] containing it; direct operator
/// tests keep it alive through their installed set.
pub fn init_worker_waker(waker: &Arc<WorkerWaker>) {
    WORKER_WAKER.set(Arc::as_ptr(waker));
}

/// Install this worker thread's cross-node waker set.
///
/// The set is boxed in thread-local storage so the cached pointer remains valid
/// until another set is installed or the thread exits.
pub fn init_waker_set(set: WakerSet) {
    WAKER_SET_OWNER.with_borrow_mut(|slot| {
        let boxed = Box::new(set);
        WAKER_SET.set(&*boxed as *const WakerSet);
        *slot = Some(boxed);
    });
}

/// Return the cross-node waker set installed on the current worker thread.
///
/// Like [`crate::memory::memory_ctx`], this assumes worker startup or test setup
/// initialized the thread-local pointer before dataflow code uses it.
pub fn waker_set() -> &'static WakerSet {
    unsafe { &*WAKER_SET.get() }
}

/// Return the node-local waker installed on the current worker thread.
///
/// Like [`crate::memory::memory_ctx`], this assumes worker startup or test setup
/// initialized the thread-local pointer before dataflow code uses it.
pub fn worker_waker() -> &'static WorkerWaker {
    unsafe { &*WORKER_WAKER.get() }
}

/// Install a test waker on the current thread.
///
/// Direct operator tests do not start a real [`crate::Dispatch`], but their send
/// paths still notify through the worker TLS. The installed [`WakerSet`] owns
/// the `Arc` that keeps the node-local pointer valid.
#[cfg(any(test, feature = "test-util"))]
pub(crate) fn install_test_worker_waker() {
    // Targeted notifications still index the slots even though these test
    // threads never park, so size the waker for the identity the test installed.
    let workers = match NUM_WORKERS.get() {
        usize::MAX => 1,
        n => n.max(1),
    };
    let waker = Arc::new(WorkerWaker::new(workers));
    init_worker_waker(&waker);
    set_current_node(0);
    init_waker_set(WakerSet::new(vec![waker], workers));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Park a thread in `local_idx` and return the wake count it observes.
    fn park_worker(waker: &Arc<WorkerWaker>, local_idx: usize) -> thread::JoinHandle<u64> {
        let waker = waker.clone();
        let last_seen = waker.wake_count();
        thread::spawn(move || {
            waker.register(local_idx);
            waker.wait_if_unchanged(last_seen, local_idx)
        })
    }

    /// Wait until `n` workers are parked, failing instead of hanging if they do
    /// not reach that state.
    fn await_parked(waker: &WorkerWaker, n: usize) {
        for _ in 0..2000 {
            if waker.parked_workers.load(Ordering::SeqCst) == n {
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("workers never parked");
    }

    #[test]
    fn delegated_broadcast_wakes_every_parked_worker() {
        let waker = Arc::new(WorkerWaker::new(8));
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let waker = waker.clone();
                thread::spawn(move || {
                    waker.register(i);
                    let mut broadcast_memo = waker.broadcast_epoch();
                    waker.wait_if_unchanged(waker.wake_count(), i);
                    waker.finish_delegated_wake(&mut broadcast_memo);
                })
            })
            .collect();
        await_parked(&waker, 8);

        waker.notify_delegated();

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(waker.parked_workers.load(Ordering::SeqCst), 0);
    }

    /// Only a release build without `unbounded-park` bounds the park, so this
    /// runs under `cargo test --release`.
    #[cfg(not(any(debug_assertions, feature = "unbounded-park")))]
    #[test]
    fn release_park_returns_without_a_notification() {
        let waker = Arc::new(WorkerWaker::new(1));
        let parked = park_worker(&waker, 0);
        await_parked(&waker, 1);

        let started = std::time::Instant::now();
        parked.join().unwrap();

        assert!(started.elapsed() < PARK_TIMEOUT * 10);
        assert_eq!(waker.parked_workers.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn notify_one_wakes_exactly_one_parked_worker() {
        let waker = Arc::new(WorkerWaker::new(2));
        let a = park_worker(&waker, 0);
        let b = park_worker(&waker, 1);
        await_parked(&waker, 2);

        waker.notify_one();

        for _ in 0..2000 {
            if waker.parked_workers.load(Ordering::SeqCst) == 1 {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(waker.parked_workers.load(Ordering::SeqCst), 1);
        waker.notify();
        assert_eq!(a.join().unwrap(), waker.wake_count());
        assert_eq!(b.join().unwrap(), waker.wake_count());
    }

    #[test]
    fn notify_slot_wakes_the_addressed_worker() {
        let waker = Arc::new(WorkerWaker::new(2));
        let a = park_worker(&waker, 0);
        let b = park_worker(&waker, 1);
        await_parked(&waker, 2);

        waker.notify_slot(1);

        b.join().unwrap();
        assert_eq!(waker.parked_workers.load(Ordering::SeqCst), 1);
        waker.notify();
        a.join().unwrap();
    }

    #[test]
    fn a_notify_before_the_park_is_not_lost() {
        let waker = Arc::new(WorkerWaker::new(1));
        waker.register(0);
        let last_seen = waker.wake_count();

        waker.notify_one();

        assert_eq!(waker.wait_if_unchanged(last_seen, 0), waker.wake_count());
        assert_eq!(waker.parked_workers.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn notify_one_near_stays_local_while_the_local_node_has_parked_workers() {
        let local = Arc::new(WorkerWaker::new(1));
        let remote = Arc::new(WorkerWaker::new(1));
        let set = WakerSet::new(vec![local.clone(), remote.clone()], 1);
        let a = park_worker(&local, 0);
        let b = park_worker(&remote, 0);
        await_parked(&local, 1);
        await_parked(&remote, 1);

        set.notify_one_near(0);

        a.join().unwrap();
        assert_eq!(remote.parked_workers.load(Ordering::SeqCst), 1);
        remote.notify();
        b.join().unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_data_send_does_not_interrupt_a_ring_wait() {
        let waker = Arc::new(WorkerWaker::new(1));
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        waker.register_ring_waker(0, crate::io::RingWakeHandle::EventFd(fd));
        assert!(waker.begin_ring_wait(0, waker.ring_wake_count()));

        let woke = waker.notify_one();

        let mut counter = 0u64;
        let drained = unsafe { libc::read(fd, (&mut counter as *mut u64).cast(), 8) };
        assert!(!woke);
        assert_eq!(drained, -1, "a data send must not write the ring eventfd");
        waker.notify();
        let drained = unsafe { libc::read(fd, (&mut counter as *mut u64).cast(), 8) };
        assert_eq!(drained, 8, "a broadcast still interrupts the ring wait");
        waker.end_ring_wait(0);
        unsafe { libc::close(fd) };
    }

    #[test]
    fn only_ring_capable_notifies_block_entering_a_ring_wait() {
        let waker = Arc::new(WorkerWaker::new(1));
        let snapshot = waker.ring_wake_count();

        waker.notify_one();

        assert!(
            waker.begin_ring_wait(0, snapshot),
            "a data send must not keep a worker with IO in flight from sleeping"
        );
        waker.end_ring_wait(0);
        waker.notify_slot(0);
        assert!(
            !waker.begin_ring_wait(0, snapshot),
            "a targeted notify raced in and the worker must re-run its loop"
        );
        waker.end_ring_wait(0);
    }

    #[test]
    fn notify_one_near_spills_to_the_next_node_when_the_local_node_is_awake() {
        let local = Arc::new(WorkerWaker::new(1));
        let remote = Arc::new(WorkerWaker::new(1));
        let set = WakerSet::new(vec![local, remote.clone()], 1);
        let remote_worker = park_worker(&remote, 0);
        await_parked(&remote, 1);

        set.notify_one_near(0);

        remote_worker.join().unwrap();
        assert_eq!(remote.parked_workers.load(Ordering::SeqCst), 0);
    }
}
