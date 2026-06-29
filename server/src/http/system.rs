//! The process's own CPU/memory plus host CPU, memory, and disk usage, sampled
//! from a persistent [`System`] so CPU deltas are measured across polls. Folded
//! into `/api/overview`.

use std::sync::Mutex;

use serde::Serialize;
use sysinfo::{Disks, Pid, ProcessesToUpdate, System};

#[derive(Serialize)]
pub(super) struct SystemInfo {
    cpu_avg: f32,
    cpu_per_core: Vec<f32>,
    mem_used: u64,
    mem_total: u64,
    process_mem: u64,
    process_cpu: f32,
    disks: Vec<DiskInfo>,
}

#[derive(Serialize)]
struct DiskInfo {
    name: String,
    mount: String,
    total: u64,
    available: u64,
}

pub(super) fn collect_system(system: &Mutex<System>, pid: Option<Pid>) -> SystemInfo {
    let mut sys = system.lock().unwrap();
    sys.refresh_cpu_all();
    sys.refresh_memory();
    let (process_mem, process_cpu) = match pid {
        Some(pid) => {
            sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            sys.process(pid)
                .map(|p| (p.memory(), p.cpu_usage()))
                .unwrap_or((0, 0.0))
        }
        None => (0, 0.0),
    };

    let cpu_per_core = sys.cpus().iter().map(|c| c.cpu_usage()).collect();
    let disks = Disks::new_with_refreshed_list()
        .list()
        .iter()
        .map(|d| DiskInfo {
            name: d.name().to_string_lossy().into_owned(),
            mount: d.mount_point().to_string_lossy().into_owned(),
            total: d.total_space(),
            available: d.available_space(),
        })
        .collect();

    SystemInfo {
        cpu_avg: sys.global_cpu_usage(),
        cpu_per_core,
        mem_used: sys.used_memory(),
        mem_total: sys.total_memory(),
        process_mem,
        process_cpu,
        disks,
    }
}
