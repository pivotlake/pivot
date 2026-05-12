//! Aggregate filtered samples into a flamegraph tree.
use ahash::AHashSet;
use serde::Serialize;

use crate::profile::{Category, Profile};

#[derive(Debug, Serialize)]
pub struct FlameNode {
    pub id: u32,
    pub parent: i32,
    pub label: String,
    #[serde(rename = "self")]
    pub self_count: u64,
    pub total: u64,
    pub depth: u32,
}

#[derive(Debug, Serialize)]
pub struct FlameResponse {
    pub total: u64,
    pub nodes: Vec<FlameNode>,
}

pub fn build_flamegraph(
    profile: &Profile,
    categories: &[Category],
    cpus: Option<&[u32]>,
    time_range_ns: Option<(u64, u64)>,
) -> FlameResponse {
    let cat_set: AHashSet<Category> = categories.iter().copied().collect();
    let cpu_set: Option<AHashSet<u32>> = cpus.map(|c| c.iter().copied().collect());

    // Tree storage as parallel arrays; children edge map per node.
    let mut parents: Vec<i32> = vec![-1];
    let mut frame_ids: Vec<i32> = vec![-1]; // -1 for the synthetic root
    let mut self_counts: Vec<u64> = vec![0];
    let mut total_counts: Vec<u64> = vec![0];
    let mut children: Vec<ahash::AHashMap<u32, u32>> = vec![ahash::AHashMap::new()];

    let mut total: u64 = 0;
    for s in &profile.samples {
        if !cat_set.contains(&s.category) {
            continue;
        }
        if let Some(ref set) = cpu_set {
            if !set.contains(&s.cpu) {
                continue;
            }
        }
        if let Some((lo, hi)) = time_range_ns {
            if s.time_ns < lo || s.time_ns > hi {
                continue;
            }
        }
        total += 1;
        total_counts[0] += 1;

        // Walk root → leaf. Stack is leaf-first, so iterate reversed.
        let mut node: u32 = 0;
        for &fid in s.stack.iter().rev() {
            let next = match children[node as usize].get(&fid) {
                Some(&n) => n,
                None => {
                    let new_id = parents.len() as u32;
                    parents.push(node as i32);
                    frame_ids.push(fid as i32);
                    self_counts.push(0);
                    total_counts.push(0);
                    children.push(ahash::AHashMap::new());
                    children[node as usize].insert(fid, new_id);
                    new_id
                }
            };
            total_counts[next as usize] += 1;
            node = next;
        }
        self_counts[node as usize] += 1;
    }

    // depth via single pass (parents always come before children in this
    // construction, so a simple linear sweep works).
    let mut depths: Vec<u32> = vec![0; parents.len()];
    for i in 1..parents.len() {
        depths[i] = depths[parents[i] as usize] + 1;
    }

    let mut nodes = Vec::with_capacity(parents.len());
    for i in 0..parents.len() {
        let label = if i == 0 {
            "(all)".to_string()
        } else {
            profile.frames.name(frame_ids[i] as u32).to_string()
        };
        nodes.push(FlameNode {
            id: i as u32,
            parent: parents[i],
            label,
            self_count: self_counts[i],
            total: total_counts[i],
            depth: depths[i],
        });
    }

    FlameResponse { total, nodes }
}
