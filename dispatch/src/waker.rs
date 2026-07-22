//! Worker parking and wake-up coordination.
//!
//! Each NUMA node has a [`WorkerWaker`] with one parking slot per worker.
//! Worker-local sends normally wake one same-node worker, targeted sends wake
//! their destination, and events that affect the whole pool use [`WakerSet`]
//! to reach every node. Worker threads access both through thread-local handles
//! installed during startup.

#[cfg(any(test, feature = "test-util"))]
use crate::worker::{NUM_WORKERS, set_current_node};
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};

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
    /// Number of workers currently parked. The no-sleeper send path checks this
    /// before scanning slots.
    parked_workers: AtomicUsize,
    slots: Box<[ParkSlot]>,
    /// Rotates [`notify_one`](Self::notify_one)'s scan start so wake-ups spread
    /// over parked workers instead of always choosing the lowest index.
    next_wake: AtomicUsize,
    /// Bumped on every delegated broadcast. The first worker woken observes the
    /// change and wakes its remaining parked siblings.
    broadcast_epoch: AtomicU64,
}

impl WorkerWaker {
    /// Create a waker with one parking slot for each worker in the node.
    pub fn new(worker_count: usize) -> Self {
        Self {
            wake_count: AtomicU64::new(0),
            parked_workers: AtomicUsize::new(0),
            slots: (0..worker_count)
                .map(|_| ParkSlot {
                    parked: AtomicBool::new(false),
                    thread: OnceLock::new(),
                })
                .collect(),
            next_wake: AtomicUsize::new(0),
            broadcast_epoch: AtomicU64::new(0),
        }
    }

    /// Register the calling thread as worker `local_idx` in this node. Must be
    /// called once, on the worker thread, before its first park.
    pub fn register(&self, local_idx: usize) {
        self.slots[local_idx]
            .thread
            .set(thread::current())
            .expect("worker slot registered twice");
    }

    /// Claim `slot` if it is parked and wake its thread. The atomic claim makes
    /// concurrent notifiers choose different sleepers whenever possible.
    fn wake_slot(&self, slot: &ParkSlot) -> bool {
        if slot.parked.swap(false, Ordering::SeqCst) {
            self.parked_workers.fetch_sub(1, Ordering::SeqCst);
            if let Some(thread) = slot.thread.get() {
                thread.unpark();
            }
            return true;
        }
        false
    }

    /// Record a notification and wake one parked worker, if any.
    ///
    /// This is used when one new stealable item needs one same-node consumer.
    /// Returns whether a parked worker was woken so callers can try another node
    /// when they need to grow the pool-wide working set.
    pub fn notify_one(&self) -> bool {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        if self.parked_workers.load(Ordering::SeqCst) == 0 {
            return false;
        }
        let start = self.next_wake.fetch_add(1, Ordering::Relaxed);
        for i in 0..self.slots.len() {
            if self.wake_slot(&self.slots[(start + i) % self.slots.len()]) {
                return true;
            }
        }
        false
    }

    /// Record a notification and wake worker `local_idx` if it is parked.
    /// Used for messages addressed to a specific worker.
    pub fn notify_slot(&self, local_idx: usize) {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        if self.parked_workers.load(Ordering::SeqCst) == 0 {
            return;
        }
        self.wake_slot(&self.slots[local_idx]);
    }

    /// Record a notification and wake every parked worker in this node.
    ///
    /// Used for events any local worker may be waiting on, such as cancellation
    /// or a sibling counter reaching zero.
    pub fn notify(&self) {
        self.wake_count.fetch_add(1, Ordering::SeqCst);
        self.wake_all_parked();
    }

    /// Unpark every currently parked worker without changing the wake count.
    fn wake_all_parked(&self) {
        if self.parked_workers.load(Ordering::SeqCst) == 0 {
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
        self.broadcast_epoch.fetch_add(1, Ordering::SeqCst);
        if self.parked_workers.load(Ordering::SeqCst) == 0 {
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

    /// Current wake count. Workers snapshot it around work passes and poll it
    /// while spinning before a park.
    pub fn wake_count(&self) -> u64 {
        self.wake_count.load(Ordering::SeqCst)
    }

    /// Park worker `local_idx` if the wake count still equals `last_seen`.
    /// Returns the current count for use on the next park attempt.
    ///
    /// No wake is lost: notifiers increment `wake_count` before scanning slots,
    /// while this method publishes the parked slot before rechecking the count.
    /// A concurrent notification therefore either changes the count observed
    /// here or claims the slot and unparks the thread. A leftover unpark token
    /// can only make a later park return early.
    pub fn wait_if_unchanged(&self, last_seen: u64, local_idx: usize) -> u64 {
        let slot = &self.slots[local_idx];
        slot.parked.store(true, Ordering::SeqCst);
        self.parked_workers.fetch_add(1, Ordering::SeqCst);
        if self.wake_count.load(Ordering::SeqCst) != last_seen {
            // Withdraw the slot unless a notifier already claimed it and
            // decremented `parked_workers`.
            if slot.parked.swap(false, Ordering::SeqCst) {
                self.parked_workers.fetch_sub(1, Ordering::SeqCst);
            }
            return self.wake_count.load(Ordering::SeqCst);
        }
        thread::park();
        if slot.parked.swap(false, Ordering::SeqCst) {
            self.parked_workers.fetch_sub(1, Ordering::SeqCst);
        }
        self.wake_count.load(Ordering::SeqCst)
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
