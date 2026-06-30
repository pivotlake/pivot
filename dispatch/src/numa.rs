//! NUMA topology discovery for splitting the worker pool into node-local groups.
//!
//! A dataflow is dispatched to exactly one group, so the cores running it and the
//! memory ring they touch all live on a single NUMA node, avoiding the remote-memory
//! latency that the all-to-all merge phase otherwise pays on a multi-node machine.
//!
//! Two steps:
//! 1. [`group_cores_by_node`] buckets the cores this process may run on by their NUMA
//!    node (read from `/sys` on Linux; everything else, and a process confined to one
//!    node, collapses to a single group).
//! 2. [`balance_worker_groups`] trims those buckets to equal size so every group has
//!    the same worker count. A constant worker count lets a dataflow's operator chain be
//!    built identically regardless of which node it lands on.

use crate::env::get_env_var_with_default;
use core_affinity::CoreId;
use std::collections::HashMap;

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
pub fn balance_worker_groups(
    groups: Vec<Vec<CoreId>>,
    requested_total: usize,
) -> Vec<Vec<CoreId>> {
    let num_nodes = groups.len().max(1);
    let requested_total = requested_total.max(1);

    if requested_total <= num_nodes {
        return groups
            .into_iter()
            .take(requested_total)
            .map(|group| group.into_iter().take(1).collect())
            .collect();
    }

    let smallest = groups.iter().map(Vec::len).min().unwrap_or(0).max(1);
    let workers_per_node = (requested_total / num_nodes).min(smallest);
    groups
        .into_iter()
        .map(|group| group.into_iter().take(workers_per_node).collect())
        .collect()
}

/// Read every NUMA node's cpu set from `/sys`, in node order. `None` when the topology
/// is unavailable (non-Linux, or `/sys` not mounted), which the caller treats as one node.
#[cfg(target_os = "linux")]
fn read_node_cpulists() -> Option<Vec<Vec<usize>>> {
    let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
    for entry in std::fs::read_dir("/sys/devices/system/node").ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name();
        let name = name.to_str()?;
        let Some(index) = name.strip_prefix("node").and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        let cpulist = std::fs::read_to_string(entry.path().join("cpulist")).ok()?;
        nodes.push((index, parse_cpulist(&cpulist)));
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
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    cpus.extend(start..=end);
                }
            }
            None => {
                if let Ok(cpu) = part.parse::<usize>() {
                    cpus.push(cpu);
                }
            }
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
