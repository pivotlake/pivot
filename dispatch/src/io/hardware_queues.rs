//! Bounds in-flight local-file operations by the device's real queue capacity.
//!
//! A submission into a saturated device queue either sleeps in the kernel's
//! request allocation or pays an io-wq punt, so each worker holds its excess in
//! a backlog instead of submitting it (see [`super::requester::IORequester`]).
//! The bound itself lives here: a block device exposes its hardware queues in
//! sysfs, each serving a fixed set of CPUs with a fixed tag budget
//! (`queue/nr_requests`). The workers whose CPUs submit into the same hardware
//! queue share one atomic in-flight counter capped at that budget, so together
//! they can fill the queue exactly and never oversubscribe it.
//!
//! A worker that finds the shared counter full parks with nothing of its own in
//! flight, so releasing a slot must wake it: each queue records the waiting
//! workers in an atomic bitmap and targets them through the worker waker when a
//! slot frees.

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
    /// The queue's tag budget (`queue/nr_requests`).
    capacity: usize,
    /// Operations currently holding one of the queue's slots.
    in_flight: AtomicUsize,
    /// Workers that found this queue full, one bit per global worker index.
    /// Sized on first use from the worker pool; after that registration and
    /// release are atomic bitmap operations only.
    waiters: OnceLock<Box<[AtomicUsize]>>,
}

impl HardwareQueue {
    pub(crate) fn with_capacity(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            in_flight: AtomicUsize::new(0),
            waiters: OnceLock::new(),
        })
    }

    /// Reserve one in-flight slot, or `None` if the queue is at capacity.
    pub(crate) fn try_acquire(self: &Arc<Self>) -> Option<InFlightPermit> {
        self.in_flight
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |in_flight| {
                (in_flight < self.capacity).then_some(in_flight + 1)
            })
            .ok()
            .map(|_| InFlightPermit {
                queue: self.clone(),
            })
    }

    /// Ask to be woken when a slot is released. The caller must retry
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

    /// Return a slot and wake every waiting worker to race for it.
    fn release(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        let Some(waiters) = self.waiters.get() else {
            return;
        };
        const WORD_BITS: usize = usize::BITS as usize;
        for (word_idx, word) in waiters.iter().enumerate() {
            let mut waiting = word.swap(0, Ordering::SeqCst);
            if waiting == 0 {
                continue;
            }
            let wakers = crate::waker::waker_set();
            while waiting != 0 {
                let bit = waiting.trailing_zeros() as usize;
                wakers.notify_worker(word_idx * WORD_BITS + bit);
                waiting &= waiting - 1;
            }
        }
    }
}

/// RAII reservation of one slot in a [`HardwareQueue`], released on drop.
pub(crate) struct InFlightPermit {
    queue: Arc<HardwareQueue>,
}

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.queue.release();
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

/// Register the block device behind `file`, returning the compact index all of
/// that file's requests will carry. This is the only `metadata().dev()` call in
/// the file's lifetime.
#[cfg(target_os = "linux")]
pub(crate) fn register_file_device(file: &std::fs::File) -> std::io::Result<DeviceIndex> {
    use std::os::unix::fs::MetadataExt;

    let device = file.metadata()?.dev();
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
pub(crate) fn register_file_device(_file: &std::fs::File) -> std::io::Result<DeviceIndex> {
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
    let capacity: usize = std::fs::read_to_string(disk.join("queue/nr_requests"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
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
        "discovered the device's hardware queues"
    );
    Some(build_device_queues(cpu_lists, capacity))
}

/// Build the cpu-indexed queue map: one shared counter per hardware queue,
/// reachable from every CPU that submits into it.
#[cfg(target_os = "linux")]
fn build_device_queues(queue_cpu_lists: Vec<Vec<usize>>, capacity: usize) -> DeviceQueues {
    let mut by_cpu = HashMap::new();
    for cpus in queue_cpu_lists {
        let queue = HardwareQueue::with_capacity(capacity);
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

        let first = queue.try_acquire();
        let second = queue.try_acquire();
        let over_capacity = queue.try_acquire();
        drop(first);
        let after_release = queue.try_acquire();

        assert!(second.is_some());
        assert!(over_capacity.is_none());
        assert!(after_release.is_some());
    }

    #[test]
    fn waiter_bitmap_spans_words_and_skips_deregistered_workers() {
        use crate::waker::{WakerSet, WorkerWaker, init_waker_set};

        let worker_count = usize::BITS as usize + 1;
        let waker = Arc::new(WorkerWaker::new(worker_count));
        init_waker_set(WakerSet::new(vec![waker.clone()], worker_count));
        let queue = HardwareQueue::with_capacity(1);
        let permit = queue.try_acquire().unwrap();

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
        let queues = build_device_queues(vec![vec![0, 1], vec![2, 3]], 63);

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
