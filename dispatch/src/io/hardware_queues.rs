//! Bounds in-flight local-file operations by the device's real queue capacity.
//!
//! A submission into a saturated device queue sleeps in the kernel's request
//! allocation, with the submitting worker in it, or pays an io-wq punt. So
//! each worker holds its excess in a backlog instead of submitting it (see
//! [`super::requester::IORequester`]). The bound itself lives here: a block
//! device exposes its hardware queues in sysfs, each serving a fixed set of
//! CPUs with a fixed budget of requests (`queue/nr_requests`). The workers
//! whose CPUs submit into the same hardware queue share one count of the
//! requests in flight, capped at that budget, so together they can fill the
//! queue exactly and never oversubscribe it.
//!
//! What the budget counts is device requests, not operations. The kernel cuts
//! an operation into requests no larger than the device takes
//! (`queue/max_sectors_kb`) and of no more separate memory runs than it takes
//! (`queue/max_segments`), so an operation holds one slot for each request it
//! becomes (see [`HardwareQueue::count_slots`]).
//!
//! A worker that finds the queue full waits in line for it. The line is first
//! come, first served: a freed slot wakes only the worker at the front, and
//! nobody takes a slot while a worker that asked earlier is still waiting.
//! That holds for the worker that freed the slot too, which would otherwise
//! reuse it for its own backlog before any waiting worker had woken, and it
//! keeps a large operation from waiting forever behind a stream of small
//! ones.

#[cfg(target_os = "linux")]
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

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
    slots: Mutex<Slots>,
}

struct Slots {
    /// Slots held by the operations in flight.
    in_flight: usize,
    /// The workers waiting for slots, longest wait first.
    waiting: VecDeque<WaitingWorker>,
}

struct WaitingWorker {
    /// The worker's global index, which is how it is woken.
    worker_idx: usize,
    /// The slots the operation it is waiting to submit needs.
    slots: usize,
}

impl Slots {
    /// The worker at the front of the line, if the free slots now cover what
    /// it is waiting for.
    fn find_worker_to_wake(&self, capacity: usize) -> Option<usize> {
        self.waiting
            .front()
            .filter(|front| self.in_flight + front.slots <= capacity)
            .map(|front| front.worker_idx)
    }
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
            slots: Mutex::new(Slots {
                in_flight: 0,
                waiting: VecDeque::new(),
            }),
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

    /// Reserve `slots` slots if they are free and no worker is waiting for
    /// the queue. Unlike [`try_acquire_or_wait`](Self::try_acquire_or_wait)
    /// the caller does not join the line when it gets `None`.
    pub(crate) fn try_acquire(self: &Arc<Self>, slots: usize) -> Option<InFlightPermit> {
        let mut state = self.slots.lock().unwrap();
        if !state.waiting.is_empty() || state.in_flight + slots > self.capacity {
            return None;
        }
        state.in_flight += slots;
        Some(InFlightPermit {
            queue: self.clone(),
            slots,
        })
    }

    /// Reserve `slots` slots for worker `worker_idx`, or put it in line for
    /// them. A worker not yet in line gets the slots only if they are free
    /// and nobody is waiting; a worker in line gets them only from the front.
    /// On `None` the worker stays in line, waiting for `slots`, and is woken
    /// when it is at the front and they are free; it then calls this again.
    pub(crate) fn try_acquire_or_wait(
        self: &Arc<Self>,
        worker_idx: usize,
        slots: usize,
    ) -> Option<InFlightPermit> {
        let mut state = self.slots.lock().unwrap();
        let place = state
            .waiting
            .iter()
            .position(|waiting| waiting.worker_idx == worker_idx);
        let is_its_turn = match place {
            Some(place) => place == 0,
            None => state.waiting.is_empty(),
        };
        if !is_its_turn || state.in_flight + slots > self.capacity {
            match place {
                Some(place) => state.waiting[place].slots = slots,
                None => state.waiting.push_back(WaitingWorker { worker_idx, slots }),
            }
            return None;
        }
        if place.is_some() {
            state.waiting.pop_front();
        }
        state.in_flight += slots;
        // What is still free may cover the worker now at the front.
        let worker_to_wake = state.find_worker_to_wake(self.capacity);
        drop(state);
        if let Some(worker) = worker_to_wake {
            crate::waker::waker_set().notify_worker(worker);
        }
        Some(InFlightPermit {
            queue: self.clone(),
            slots,
        })
    }

    /// Take worker `worker_idx` out of the line, for a worker that will not
    /// come back for the slots it was waiting for.
    pub(crate) fn stop_waiting(&self, worker_idx: usize) {
        let mut state = self.slots.lock().unwrap();
        state
            .waiting
            .retain(|waiting| waiting.worker_idx != worker_idx);
        let worker_to_wake = state.find_worker_to_wake(self.capacity);
        drop(state);
        if let Some(worker) = worker_to_wake {
            crate::waker::waker_set().notify_worker(worker);
        }
    }

    /// Return `slots` slots, and wake the worker at the front of the line if
    /// they cover what it is waiting for.
    fn release(&self, slots: usize) {
        let mut state = self.slots.lock().unwrap();
        state.in_flight -= slots;
        let worker_to_wake = state.find_worker_to_wake(self.capacity);
        drop(state);
        if let Some(worker) = worker_to_wake {
            crate::waker::waker_set().notify_worker(worker);
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
    /// The queue this worker is in line for, if it is waiting. A worker waits
    /// only for the operation at the front of its backlog, so for one queue
    /// at a time.
    waiting_in: Option<Arc<HardwareQueue>>,
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
            waiting_in: None,
            #[cfg(test)]
            forced: None,
        }
    }

    /// Reserve `slots` slots of `queue` for the operation at the front of
    /// this worker's backlog, or wait in line for them (see
    /// [`HardwareQueue::try_acquire_or_wait`]).
    pub(crate) fn try_acquire_or_wait(
        &mut self,
        queue: &Arc<HardwareQueue>,
        slots: usize,
    ) -> Option<InFlightPermit> {
        let worker_idx = crate::worker::WORKER_IDX.get();
        // The front of the backlog may have changed to an operation on
        // another device since this worker got in line: it gives up its place
        // there instead of holding up that line while it waits in this one.
        if let Some(previous) = &self.waiting_in
            && !Arc::ptr_eq(previous, queue)
        {
            previous.stop_waiting(worker_idx);
        }
        let permit = queue.try_acquire_or_wait(worker_idx, slots);
        self.waiting_in = match permit {
            Some(_) => None,
            None => Some(queue.clone()),
        };
        permit
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

impl Drop for WorkerHardwareQueues {
    /// A worker that goes away while in line would hold up everyone behind
    /// it, so it leaves the line.
    fn drop(&mut self) {
        if let Some(queue) = self.waiting_in.take() {
            queue.stop_waiting(crate::worker::WORKER_IDX.get());
        }
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
/// queue (`mq/<n>/cpu_list`), the per-queue budget of requests
/// (`queue/nr_requests`), and the most one request takes in bytes
/// (`queue/max_sectors_kb`) and in separate memory runs
/// (`queue/max_segments`). A partition submits through its whole disk's
/// queues.
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

    /// A waker set over `worker_count` workers of one node, installed on the
    /// test's thread, and that node's waker to count wakes on.
    fn install_wakers(worker_count: usize) -> Arc<crate::waker::WorkerWaker> {
        use crate::waker::{WakerSet, WorkerWaker, init_waker_set};

        let waker = Arc::new(WorkerWaker::new(worker_count));
        init_waker_set(WakerSet::new(vec![waker.clone()], worker_count));
        waker
    }

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
    fn a_freed_slot_goes_to_the_worker_that_has_waited_longest() {
        let waker = install_wakers(3);
        let queue = HardwareQueue::with_capacity(1);
        let permit = queue.try_acquire_or_wait(0, 1).unwrap();
        let first_in_line = queue.try_acquire_or_wait(1, 1);
        let second_in_line = queue.try_acquire_or_wait(2, 1);

        drop(permit);
        let out_of_turn = queue.try_acquire_or_wait(2, 1);
        let in_turn = queue.try_acquire_or_wait(1, 1);

        assert!(first_in_line.is_none() && second_in_line.is_none());
        assert!(out_of_turn.is_none());
        assert!(in_turn.is_some());
        assert_eq!(waker.ring_wake_count(), 1, "only the front worker is woken");
    }

    #[test]
    fn free_slots_are_not_taken_past_a_worker_waiting_for_more_of_them() {
        install_wakers(2);
        let queue = HardwareQueue::with_capacity(4);
        let _held = queue.try_acquire(2).unwrap();
        let waiting_for_the_whole_queue = queue.try_acquire_or_wait(1, 4);

        let past_the_line = queue.try_acquire(1);

        assert!(waiting_for_the_whole_queue.is_none());
        assert!(past_the_line.is_none());
    }

    #[test]
    fn a_worker_that_stops_waiting_lets_the_next_one_through() {
        let waker = install_wakers(3);
        let queue = HardwareQueue::with_capacity(2);
        let _held = queue.try_acquire(1).unwrap();
        let waiting_for_two = queue.try_acquire_or_wait(1, 2);
        let waiting_for_one = queue.try_acquire_or_wait(2, 1);

        queue.stop_waiting(1);
        let next = queue.try_acquire_or_wait(2, 1);

        assert!(waiting_for_two.is_none() && waiting_for_one.is_none());
        assert!(next.is_some());
        assert_eq!(waker.ring_wake_count(), 1);
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
