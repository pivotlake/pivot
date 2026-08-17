//! NUMA topology discovery for splitting the worker pool into node-local groups.
//!
//! Every dataflow runs on all workers of all nodes; what the split buys is that each
//! worker's *memory* stays on its own node (see [`crate::memory`]) and hot coordination
//! (work stealing, wakeups) stays between same-node siblings. [`Topology`] is the shared
//! description of that split which the rest of the crate partitions by.
//!
//! Discovery happens in two steps:
//! 1. [`group_cores_by_node`] buckets the cores this process may run on by their NUMA
//!    node (read from `/sys` on Linux; everything else, and a process confined to one
//!    node, collapses to a single group).
//! 2. [`balance_worker_groups`] trims those buckets to equal size so every group has
//!    the same worker count. Equal groups keep the per-node split of the ring and of
//!    per-worker structures uniform, so nothing needs per-node special cases.

use crate::env::get_env_var_with_default;
use core_affinity::CoreId;
use std::collections::HashMap;

/// The node contributing the most rows in a per-node tally.
///
/// An empty tally selects node 0. Equal nonempty tallies select the
/// highest-indexed node, matching [`Iterator::max_by_key`].
pub fn dominant_node(rows_by_node: &[usize]) -> usize {
    rows_by_node
        .iter()
        .enumerate()
        .max_by_key(|(_, rows)| **rows)
        .map_or(0, |(node_id, _)| node_id)
}

/// Describes the shape of the worker pool: equal-sized groups of workers,
/// one group per NUMA node. Worker indices are global and dense
/// (`0..total_workers()`), with node 0's workers first, so
/// `worker / workers_per_node` is a worker's node.
///
/// Everything that partitions per-worker or per-node state (channel stealers,
/// ring slot ownership, waker routing) derives its split from this one type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Topology {
    pub workers_per_node: usize,
    pub node_count: usize,
}

impl Topology {
    /// Build the topology of a single-socket machine, where every worker is
    /// on one node. This is the degenerate case all NUMA-aware code must
    /// reduce to.
    pub fn single_node(workers: usize) -> Self {
        Self {
            workers_per_node: workers,
            node_count: 1,
        }
    }

    pub fn total_workers(&self) -> usize {
        self.workers_per_node * self.node_count
    }

    /// The NUMA node group `worker` belongs to.
    pub fn node_of_worker(&self, worker: usize) -> usize {
        worker / self.workers_per_node
    }

    /// `worker`'s position within its NUMA node group, the counterpart of
    /// [`node_of_worker`](Self::node_of_worker) for indexing per-node state.
    pub fn local_index_of_worker(&self, worker: usize) -> usize {
        worker % self.workers_per_node
    }

    /// Global indices of the workers on the same node as `worker`, itself included.
    pub fn node_siblings(&self, worker: usize) -> std::ops::Range<usize> {
        let node = self.node_of_worker(worker);
        node * self.workers_per_node..(node + 1) * self.workers_per_node
    }
}

/// The default worker count is every available core.
pub fn default_worker_count() -> usize {
    core_affinity::get_core_ids().map_or(1, |cores| cores.len().max(1))
}

/// Group the cores this process may run on by NUMA node.
///
/// On Linux, reads each node's `cpulist` from `/sys/devices/system/node/node*/cpulist`
/// and buckets `available` accordingly, preserving node order and dropping nodes with no
/// available core. Returns a single group containing all cores when there is one node,
/// when the topology can't be read (non-Linux, missing `/sys`), or when `PIVOT_NUMA=false`
/// disables the split. The returned groups always cover `available` exactly once.
pub fn group_cores_by_node(available: Vec<CoreId>) -> Vec<Vec<CoreId>> {
    if !get_env_var_with_default("PIVOT_NUMA", true) {
        return vec![available];
    }
    let Some(nodes) = read_node_cpulists().filter(|nodes| nodes.len() > 1) else {
        return vec![available];
    };
    // Map each cpu to its node, then bucket the available cores in one pass. A core
    // absent from every node's cpulist (unusual, but possible if `/sys` and the
    // affinity mask disagree) falls into the first node so it is never dropped.
    let cpu_to_node: HashMap<usize, usize> = nodes
        .iter()
        .enumerate()
        .flat_map(|(node, cpus)| cpus.iter().map(move |&cpu| (cpu, node)))
        .collect();
    let mut groups = vec![Vec::new(); nodes.len()];
    for core in available {
        let node = cpu_to_node.get(&core.id).copied().unwrap_or(0);
        groups[node].push(core);
    }
    groups.retain(|group: &Vec<CoreId>| !group.is_empty());
    groups
}

/// Trim `groups` to equal size so every node runs the same number of workers.
///
/// `workers_per_node` is the per-node share of `requested_total`, capped by the smallest
/// node so no node is overcommitted; leftover cores beyond the equal share are dropped.
/// When `requested_total` is smaller than the node count, only the first `requested_total`
/// nodes are used with one worker each. Every returned group has the same length (at least
/// one), and at least one group is returned.
pub fn balance_worker_groups(groups: Vec<Vec<CoreId>>, requested_total: usize) -> Vec<Vec<CoreId>> {
    assert!(!groups.is_empty(), "no core groups to balance");
    let num_nodes = groups.len();
    let requested_total = requested_total.max(1);

    if requested_total <= num_nodes {
        return groups
            .into_iter()
            .take(requested_total)
            .map(|group| group.into_iter().take(1).collect())
            .collect();
    }

    let smallest = groups
        .iter()
        .map(Vec::len)
        .min()
        .expect("groups is non-empty");
    let workers_per_node = (requested_total / num_nodes).min(smallest.max(1));
    // Uniform group sizes are a hard requirement (worker indexing, ring
    // regions, and per-node barriers all assume them), so an asymmetric core
    // set caps every node at the smallest node's share. That can drop a large
    // fraction of an unevenly pinned allocation (a cgroup cpuset with 40 cores
    // on one node and 2 on the other yields 2 workers per node), which must
    // never happen silently.
    let dropped = groups.iter().map(Vec::len).sum::<usize>() - workers_per_node * num_nodes;
    if dropped > 0 {
        tracing::warn!(
            workers_per_node,
            num_nodes,
            dropped,
            "asymmetric core groups: capping every node at the smallest node's \
             share and leaving {dropped} usable cores without workers"
        );
    }
    groups
        .into_iter()
        .map(|group| group.into_iter().take(workers_per_node).collect())
        .collect()
}

/// Read every NUMA node's cpu set from `/sys`, in node order. `None` when the topology
/// is unavailable (non-Linux, or `/sys` not mounted), which the caller treats as one
/// node. An unexpected error mid-read also falls back to one node, but loudly: silently
/// halving a two-node box's placement quality is the kind of quiet degradation this
/// module exists to prevent.
#[cfg(target_os = "linux")]
fn read_node_cpulists() -> Option<Vec<Vec<usize>>> {
    let dir = std::fs::read_dir("/sys/devices/system/node").ok()?;
    let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
    for entry in dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                tracing::warn!(%error, "failed reading /sys node entries; assuming one node");
                return None;
            }
        };
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(index) = name
            .strip_prefix("node")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        match std::fs::read_to_string(entry.path().join("cpulist")) {
            Ok(cpulist) => nodes.push((index, parse_cpulist(&cpulist))),
            Err(error) => {
                tracing::warn!(%error, node = index, "failed reading a node's cpulist; assuming one node");
                return None;
            }
        }
    }
    if nodes.is_empty() {
        return None;
    }
    nodes.sort_by_key(|(index, _)| *index);
    Some(nodes.into_iter().map(|(_, cpus)| cpus).collect())
}

#[cfg(not(target_os = "linux"))]
fn read_node_cpulists() -> Option<Vec<Vec<usize>>> {
    None
}

/// Parse a Linux cpulist such as `"0-95"` or `"0-23,48-71"` or `"5"` into cpu indices.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpulist(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((start, end)) => {
                let (start, end) = (
                    start
                        .parse::<usize>()
                        .expect("malformed /sys cpulist range"),
                    end.parse::<usize>().expect("malformed /sys cpulist range"),
                );
                cpus.extend(start..=end);
            }
            None => cpus.push(part.parse::<usize>().expect("malformed /sys cpulist entry")),
        }
    }
    cpus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cores(ids: impl IntoIterator<Item = usize>) -> Vec<CoreId> {
        ids.into_iter().map(|id| CoreId { id }).collect()
    }

    fn ids(group: &[CoreId]) -> Vec<usize> {
        group.iter().map(|c| c.id).collect()
    }

    #[test]
    fn parse_cpulist_handles_ranges_and_singletons() {
        assert_eq!(parse_cpulist("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpulist("0-2,5,7-8"), vec![0, 1, 2, 5, 7, 8]);
        assert_eq!(parse_cpulist("4\n"), vec![4]);
    }

    #[test]
    fn balanced_groups_are_equal_sized_capped_by_smallest_node() {
        let groups = vec![cores(0..6), cores(6..10)];

        let balanced = balance_worker_groups(groups, 8);

        assert_eq!(balanced.len(), 2);
        assert_eq!(ids(&balanced[0]), vec![0, 1, 2, 3]);
        assert_eq!(ids(&balanced[1]), vec![6, 7, 8, 9]);
    }

    #[test]
    fn balanced_groups_fall_back_to_one_worker_per_node_when_total_is_small() {
        let groups = vec![cores(0..4), cores(4..8), cores(8..12)];

        let balanced = balance_worker_groups(groups, 2);

        assert_eq!(balanced.len(), 2);
        assert!(balanced.iter().all(|g| g.len() == 1));
    }
}
