//! Machine-derived defaults shared by every binary that embeds the engine:
//! the `pivotdb-server` binary and the `pivot-cli` shell both size their
//! buffer pool here.

use dispatch::env::get_env_var_with_default;
use metastore_disk::ByteSize;

/// The buffer pool's byte budget: `memory` verbatim when set, otherwise
/// `default_pct` percent of the machine's memory, with the percentage
/// overridable through `PIVOT_MEMORY_PCT`.
pub fn compute_pool_bytes(memory: Option<ByteSize>, default_pct: usize) -> usize {
    match memory {
        Some(size) => size.as_bytes() as usize,
        None => {
            let memory_pct: usize = get_env_var_with_default("PIVOT_MEMORY_PCT", default_pct);
            read_total_memory_bytes() * memory_pct / 100
        }
    }
}

/// The total physical memory of the machine in bytes.
fn read_total_memory_bytes() -> usize {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
    .total_memory() as usize
}
