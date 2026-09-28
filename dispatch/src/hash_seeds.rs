//! Seeds for the hash maps the process creates.
//!
//! Every `ahash` map draws a seed of its own when it is created, and ahash's
//! default source for it is one process-wide atomic counter. The workers build
//! their operators, and the maps inside them, at the same moment a dataflow
//! arrives, so they all contend for that counter's cache line.
//! [`install_per_thread_hash_seeds`] swaps in a counter private to each thread.
//! The keys a seed is mixed with are still drawn at random once per process.

use ahash::random_state::{RandomSource, set_random_source};
use std::cell::Cell;

/// Draws map seeds from a counter private to each thread, mixed with the
/// address of that counter so threads do not repeat each other's seeds.
struct PerThreadSeeds;

impl RandomSource for PerThreadSeeds {
    fn gen_hasher_seed(&self) -> usize {
        thread_local! {
            static NEXT_SEED: Cell<usize> = const { Cell::new(0) };
        }
        NEXT_SEED.with(|next| {
            let seed = next.get();
            next.set(seed.wrapping_add(1));
            seed ^ (next as *const Cell<usize> as usize).rotate_left(32)
        })
    }
}

/// Makes every hash map created from here on draw its seed from its thread's
/// own counter. Must run before the process creates its first hash map.
pub fn install_per_thread_hash_seeds() {
    set_random_source(PerThreadSeeds)
        .expect("hash map seeds are installed before the first map is created");
}
