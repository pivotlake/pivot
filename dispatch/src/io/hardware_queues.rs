//! Bounds in-flight local-file operations by the device's real queue capacity.
//!
//! A submission into a saturated device queue either sleeps in the kernel's
//! request allocation, with the submitting worker in it, or pays an io-wq
//! punt, so each worker holds its excess in a backlog instead of submitting
//! it (see [`super::requester::IORequester`]). The bound itself lives here: a
//! block device exposes its hardware queues in sysfs, each serving a fixed
//! set of CPUs with a fixed budget of requests (`queue/nr_requests`). The
//! workers whose CPUs submit into the same hardware queue share one atomic
//! count of the requests in flight, capped at that budget, so together they
//! can fill the queue exactly and never oversubscribe it.
//!
//! What the budget counts is device requests, not operations. The kernel cuts
//! an operation into requests no larger than the device takes
//! (`queue/max_sectors_kb`) and of no more separate memory runs than it takes
//! (`queue/max_segments`), so an operation holds one slot for each request it
//! becomes (see [`HardwareQueue::count_slots`]).
//!
//! A worker that finds no room for an operation parks with nothing of its own
//! in flight, so releasing slots must wake it: each queue records the waiting
//! workers in an atomic bitmap and targets them through the worker waker when
//! slots free. The woken workers race for the slots and the first to run
//! wins, so each release starts waking from a different worker: woken in the
//! same order every time, the same workers would win every time and do most
//! of the device's work.

#[cfg(target_os = "linux")]
use std::collections::HashMap;
#[cfg(target_os = "linux")]
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// In-flight cap for a device whose hardware-queue topology cannot be read
/// (non-Linux, or a device without blk-mq sysfs entries, e.g. tmpfs). Private
/// to one worker: without the topology there is no sharing structure to
/// mirror, so each worker gets a conservative cap of its own, deep enough to
/// keep a device busy but far below any real queue's budget.
const FALLBACK_WORKER_CAPACITY: usize = 8;

/// Process-wide index of a block device, assigned when a local file is opened.
/// Requests carry this instead of asking the file descriptor for its device on
/// every submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceIndex(usize);

/// One block-device hardware queue's in-flight budget, shared by every worker
/// whose CPU submits into it.
pub(crate) struct HardwareQueue {
    /// The queue's budget of requests (`queue/nr_requests`).
    capacity: usize,
    /// The most bytes the device takes as one request
    /// (`queue/max_sectors_kb`).
    largest_request_bytes: usize,
    /// The most separate memory runs the device takes in one request
    /// (`queue/max_segments`).
    most_runs_per_request: usize,
    /// Slots held by the operations in flight.
    in_flight: AtomicUsize,
    /// Workers that found no room in this queue, one bit per global worker
    /// index.
    /// Sized on first use from the worker pool; after that registration and
    /// release are atomic bitmap operations only.
    waiters: OnceLock<Box<[AtomicUsize]>>,
    /// Which waiting worker the next release wakes first: its place among the
    /// workers waiting then, wrapped round their number. Moves one place on
    /// with every release that wakes any.
    first_waiter_to_wake: AtomicUsize,
}

impl HardwareQueue {
    /// A queue of `capacity` slots on a device with no limit on a request's
    /// size, so that every operation takes one slot.
    pub(crate) fn with_capacity(capacity: usize) -> Arc<Self> {
        Self::with_request_limits(capacity, usize::MAX, usize::MAX)
    }

    pub(crate) fn with_request_limits(
        capacity: usize,
        largest_request_bytes: usize,
        most_runs_per_request: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            largest_request_bytes,
            most_runs_per_request,
            in_flight: AtomicUsize::new(0),
            waiters: OnceLock::new(),
            first_waiter_to_wake: AtomicUsize::new(0),
        })
    }

    /// How many slots an operation of `bytes` bytes in `runs` separate memory
    /// runs holds: one for each request the kernel cuts it into. An operation
    /// of more requests than the queue has slots holds all of them, and has
    /// the queue to itself.
    pub(crate) fn count_slots(&self, bytes: usize, runs: usize) -> usize {
        bytes
            .div_ceil(self.largest_request_bytes)
            .max(runs.div_ceil(self.most_runs_per_request))
            .clamp(1, self.capacity)
    }

    /// Reserve `slots` slots, or `None` if the queue has no room for them.
    pub(crate) fn try_acquire(self: &Arc<Self>, slots: usize) -> Option<InFlightPermit> {
        self.in_flight
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |in_flight| {
                (in_flight + slots <= self.capacity).then_some(in_flight + slots)
            })
            .ok()
            .map(|_| InFlightPermit {
                queue: self.clone(),
                slots,
            })
    }

    /// Ask to be woken when slots are released. The caller must retry
    /// `try_acquire` once after registering: a release between its failed
    /// attempt and the registration would otherwise go unseen forever. The
    /// retry closes that window, because a release decrements the counter
    /// before sweeping the waiters: a retry that still finds the queue full
    /// ran before that decrement, so the sweep is still to come and will see
    /// this registration.
    pub(crate) fn register_waiter(&self, worker_idx: usize) {
        const WORD_BITS: usize = usize::BITS as usize;

        let waiters = self.waiters.get_or_init(|| {
            let word_count = crate::waker::waker_set().worker_count().div_ceil(WORD_BITS);
            (0..word_count).map(|_| AtomicUsize::new(0)).collect()
        });
        let word = worker_idx / WORD_BITS;
        assert!(
            word < waiters.len(),
            "worker {worker_idx} is outside the hardware queue's worker pool"
        );
        waiters[word].fetch_or(1usize << (worker_idx % WORD_BITS), Ordering::SeqCst);
    }

    /// Drop a registration whose retry acquired a slot after all, so releases
    /// stop waking a worker that is no longer waiting.
    pub(crate) fn deregister_waiter(&self, worker_idx: usize) {
        const WORD_BITS: usize = usize::BITS as usize;

        let waiters = self
            .waiters
            .get()
            .expect("a waiter cannot deregister before registering");
        waiters[worker_idx / WORD_BITS]
            .fetch_and(!(1usize << (worker_idx % WORD_BITS)), Ordering::SeqCst);
    }

    /// Return `slots` slots and wake every waiting worker to race for them,
    /// starting from a different worker each time.
    fn release(&self, slots: usize) {
        self.in_flight.fetch_sub(slots, Ordering::SeqCst);
        let Some(waiters) = self.waiters.get() else {
            return;
        };
        const WORD_BITS: usize = usize::BITS as usize;
        let mut waiting_workers = Vec::new();
        for (word_idx, word) in waiters.iter().enumerate() {
            let mut waiting = word.swap(0, Ordering::SeqCst);
            while waiting != 0 {
                let bit = waiting.trailing_zeros() as usize;
                waiting_workers.push(word_idx * WORD_BITS + bit);
                waiting &= waiting - 1;
            }
        }
        if waiting_workers.is_empty() {
            return;
        }
        // The same cycle each time, entered one worker further on, so that no
        // worker is always the first to be woken and to find the slots free.
        let first =
            self.first_waiter_to_wake.fetch_add(1, Ordering::Relaxed) % waiting_workers.len();
        waiting_workers.rotate_left(first);
        let wakers = crate::waker::waker_set();
        for worker in waiting_workers {
            wakers.notify_worker(worker);
        }
    }
}

/// RAII reservation of an operation's slots in a [`HardwareQueue`], released
/// on drop.
pub(crate) struct InFlightPermit {
    queue: Arc<HardwareQueue>,
    slots: usize,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.queue.release(self.slots);
    }
}

/// One worker's view of the machine's hardware queues: for each device it has
/// touched, the shared queue its pinned CPU submits into. Devices whose
/// topology cannot be read share a private per-worker fallback queue instead.
pub(crate) struct WorkerHardwareQueues {
    /// The CPU this worker submits from. Workers pin themselves to their core
    /// before building the requester; an unpinned caller (a test) simply gets
    /// the queue of whichever CPU it was on, which still counts consistently.
    #[cfg(target_os = "linux")]
    cpu: Option<usize>,
    /// Device index to the hardware queue serving `cpu`. When a new index is
    /// first seen, every newly registered device through that index is appended
    /// so subsequent lookups are direct indexing operations.
    #[cfg(target_os = "linux")]
    by_device: Vec<Arc<HardwareQueue>>,
    /// The private cap for devices without readable topology.
    fallback: Arc<HardwareQueue>,
    /// Test override: every file resolves to this queue.
    #[cfg(test)]
    forced: Option<Arc<HardwareQueue>>,
}

impl WorkerHardwareQueues {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(target_os = "linux")]
            cpu: read_current_cpu(),
            #[cfg(target_os = "linux")]
            by_device: Vec::new(),
            fallback: HardwareQueue::with_capacity(FALLBACK_WORKER_CAPACITY),
            #[cfg(test)]
            forced: None,
        }
    }

    /// The hardware queue `device_idx` serves this worker's CPU with, or the
    /// private fallback when the device publishes no readable topology.
    #[cfg(target_os = "linux")]
    pub(crate) fn queue_for_device(&mut self, device_idx: DeviceIndex) -> Arc<HardwareQueue> {
        #[cfg(test)]
        if let Some(forced) = &self.forced {
            return forced.clone();
        }
        let Some(cpu) = self.cpu else {
            return self.fallback.clone();
        };
        if let Some(queue) = self.by_device.get(device_idx.0) {
            return queue.clone();
        }

        while self.by_device.len() <= device_idx.0 {
            let next = DeviceIndex(self.by_device.len());
            let queue =
                hardware_queue_for_device(next, cpu).unwrap_or_else(|| self.fallback.clone());
            self.by_device.push(queue);
        }
        self.by_device[device_idx.0].clone()
    }

    #[cfg(not(target_os = "linux"))]
    pub(crate) fn queue_for_device(&mut self, _device_idx: DeviceIndex) -> Arc<HardwareQueue> {
        #[cfg(test)]
        if let Some(forced) = &self.forced {
            return forced.clone();
        }
        self.fallback.clone()
    }

    /// Route every file through `queue`, so a test controls the capacity and
    /// the sharing.
    #[cfg(test)]
    pub(crate) fn force(&mut self, queue: Arc<HardwareQueue>) {
        self.forced = Some(queue);
    }
}

/// The CPU the calling thread runs on, `None` only if the kernel cannot say
/// (in which case every device falls back to the private per-worker cap).
#[cfg(target_os = "linux")]
fn read_current_cpu() -> Option<usize> {
    let cpu = unsafe { libc::sched_getcpu() };
    (cpu >= 0).then_some(cpu as usize)
}

/// One device's hardware queues, indexed by the CPUs they serve.
#[cfg(target_os = "linux")]
struct DeviceQueues {
    by_cpu: HashMap<usize, Arc<HardwareQueue>>,
}

/// Process-wide device registry. A local file joins this registry once, when
/// it is wrapped after opening; its requests carry the resulting vector index.
#[cfg(target_os = "linux")]
struct DeviceRegistry {
    by_device: HashMap<u64, DeviceIndex>,
    devices: Vec<(u64, Option<DeviceQueues>)>,
}

#[cfg(target_os = "linux")]
fn device_registry() -> &'static Mutex<DeviceRegistry> {
    use std::sync::LazyLock;
    static REGISTRY: LazyLock<Mutex<DeviceRegistry>> = LazyLock::new(|| {
        Mutex::new(DeviceRegistry {
            by_device: HashMap::new(),
            devices: Vec::new(),
        })
    });
    &REGISTRY
}

/// Register the block device behind the file `metadata` describes, returning
/// the compact index all of that file's requests will carry.
#[cfg(target_os = "linux")]
pub(crate) fn register_file_device(metadata: &std::fs::Metadata) -> std::io::Result<DeviceIndex> {
    use std::os::unix::fs::MetadataExt;

    let device = metadata.dev();
    let mut registry = device_registry().lock().unwrap();
    if let Some(&device_idx) = registry.by_device.get(&device) {
        return Ok(device_idx);
    }

    let queues = read_device_queues(device);
    if queues.is_none() {
        tracing::warn!(
            device,
            "cannot read the device's hardware-queue topology; capping in-flight \
             operations per worker instead"
        );
    }
    let device_idx = DeviceIndex(registry.devices.len());
    registry.devices.push((device, queues));
    registry.by_device.insert(device, device_idx);
    Ok(device_idx)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn register_file_device(_metadata: &std::fs::Metadata) -> std::io::Result<DeviceIndex> {
    Ok(DeviceIndex(0))
}

/// The hardware queue of `device_idx` that submissions from `cpu` enter.
/// `None` if the topology cannot be read or names no queue for `cpu`.
#[cfg(target_os = "linux")]
fn hardware_queue_for_device(device_idx: DeviceIndex, cpu: usize) -> Option<Arc<HardwareQueue>> {
    let registry = device_registry().lock().unwrap();
    let (device, queues) = registry.devices.get(device_idx.0)?;

    let queue = queues
        .as_ref()
        .and_then(|queues| queues.by_cpu.get(&cpu).cloned());
    if queues.is_some() && queue.is_none() {
        tracing::warn!(
            device = *device,
            cpu,
            "the device's hardware queues name no queue for this CPU; capping its \
             in-flight operations per worker instead"
        );
    }
    queue
}

/// Read `device`'s queue layout from sysfs: which CPUs feed each hardware
/// queue (`mq/<n>/cpu_list`) and the per-queue tag budget
/// (`queue/nr_requests`). A partition submits through its whole disk's queues.
#[cfg(target_os = "linux")]
fn read_device_queues(device: u64) -> Option<DeviceQueues> {
    let (major, minor) = split_device_number(device);
    let node = std::path::PathBuf::from(format!("/sys/dev/block/{major}:{minor}"))
        .canonicalize()
        .ok()?;
    let disk = if node.join("partition").exists() {
        node.parent()?.to_path_buf()
    } else {
        node
    };
    let read_queue_limit = |name: &str| -> Option<usize> {
        std::fs::read_to_string(disk.join("queue").join(name))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let capacity = read_queue_limit("nr_requests")?;
    let largest_request_bytes = read_queue_limit("max_sectors_kb")? * 1024;
    let most_runs_per_request = read_queue_limit("max_segments")?;
    let mut cpu_lists = Vec::new();
    for entry in std::fs::read_dir(disk.join("mq")).ok()? {
        let cpu_list = std::fs::read_to_string(entry.ok()?.path().join("cpu_list")).ok()?;
        cpu_lists.push(crate::numa::parse_cpulist(&cpu_list));
    }
    if cpu_lists.is_empty() {
        return None;
    }
    tracing::info!(
        device,
        queues = cpu_lists.len(),
        capacity,
        largest_request_bytes,
        most_runs_per_request,
        "discovered the device's hardware queues"
    );
    Some(build_device_queues(
        cpu_lists,
        capacity,
        largest_request_bytes,
        most_runs_per_request,
    ))
}

/// Build the cpu-indexed queue map: one shared counter per hardware queue,
/// reachable from every CPU that submits into it.
#[cfg(target_os = "linux")]
fn build_device_queues(
    queue_cpu_lists: Vec<Vec<usize>>,
    capacity: usize,
    largest_request_bytes: usize,
    most_runs_per_request: usize,
) -> DeviceQueues {
    let mut by_cpu = HashMap::new();
    for cpus in queue_cpu_lists {
        let queue = HardwareQueue::with_request_limits(
            capacity,
            largest_request_bytes,
            most_runs_per_request,
        );
        for cpu in cpus {
            by_cpu.insert(cpu, queue.clone());
        }
    }
    DeviceQueues { by_cpu }
}

/// Split a Linux `dev_t` into `(major, minor)`, mirroring glibc's
/// `gnu_dev_major`/`gnu_dev_minor` bit layout.
#[cfg(target_os = "linux")]
fn split_device_number(device: u64) -> (u64, u64) {
    let major = ((device >> 8) & 0xfff) | ((device >> 32) & 0xffff_f000);
    let minor = (device & 0xff) | ((device >> 12) & 0xffff_ff00);
    (major, minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_queue_rejects_until_a_permit_is_released() {
        let queue = HardwareQueue::with_capacity(2);

        let first = queue.try_acquire(1);
        let second = queue.try_acquire(1);
        let over_capacity = queue.try_acquire(1);
        drop(first);
        let after_release = queue.try_acquire(1);

        assert!(second.is_some());
        assert!(over_capacity.is_none());
        assert!(after_release.is_some());
    }

    #[test]
    fn an_operation_holds_a_slot_for_each_request_the_device_cuts_it_into() {
        let queue = HardwareQueue::with_request_limits(63, 256 * 1024, 65);

        let small = queue.count_slots(4096, 1);
        let four_requests_of_bytes = queue.count_slots(1024 * 1024, 1);
        let three_requests_of_runs = queue.count_slots(64 * 1024, 140);
        let larger_than_the_queue = queue.count_slots(1 << 30, 1);

        assert_eq!(small, 1);
        assert_eq!(four_requests_of_bytes, 4);
        assert_eq!(three_requests_of_runs, 3);
        assert_eq!(larger_than_the_queue, 63);
    }

    #[test]
    fn an_operation_of_several_requests_waits_for_all_of_its_slots() {
        let queue = HardwareQueue::with_capacity(4);
        let held: Vec<_> = (0..3).map(|_| queue.try_acquire(1)).collect();

        let with_one_slot_free = queue.try_acquire(2);
        drop(held);
        let with_the_queue_free = queue.try_acquire(2);

        assert!(with_one_slot_free.is_none());
        assert!(with_the_queue_free.is_some());
    }

    #[test]
    fn waiter_bitmap_spans_words_and_skips_deregistered_workers() {
        use crate::waker::{WakerSet, WorkerWaker, init_waker_set};

        let worker_count = usize::BITS as usize + 1;
        let waker = Arc::new(WorkerWaker::new(worker_count));
        init_waker_set(WakerSet::new(vec![waker.clone()], worker_count));
        let queue = HardwareQueue::with_capacity(1);
        let permit = queue.try_acquire(1).unwrap();

        queue.register_waiter(0);
        queue.register_waiter(1);
        queue.register_waiter(usize::BITS as usize);
        queue.deregister_waiter(1);
        drop(permit);

        assert_eq!(waker.ring_wake_count(), 2);
        assert!(
            queue
                .waiters
                .get()
                .unwrap()
                .iter()
                .all(|word| word.load(Ordering::SeqCst) == 0)
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpus_feeding_one_hardware_queue_share_its_counter() {
        let queues = build_device_queues(vec![vec![0, 1], vec![2, 3]], 63, 256 * 1024, 65);

        let queue_of = |cpu: usize| queues.by_cpu.get(&cpu).unwrap();

        assert!(Arc::ptr_eq(queue_of(0), queue_of(1)));
        assert!(Arc::ptr_eq(queue_of(2), queue_of(3)));
        assert!(!Arc::ptr_eq(queue_of(0), queue_of(2)));
        assert_eq!(queue_of(0).capacity, 63);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn device_numbers_split_into_major_and_minor() {
        // major 259, minor 1: the common nvme partition layout.
        let device = (259 << 8) | 1;

        assert_eq!(split_device_number(device), (259, 1));
        // High major and minor bits live above bit 32 and bit 12 respectively.
        let device = (0x1000u64 << 32) | (259 << 8) | (0x100u64 << 12) | 1;
        assert_eq!(split_device_number(device), (0x1000 | 259, 0x100 | 1));
    }
}
