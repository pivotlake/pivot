//! The buffer-pool budget a `pivot` process takes when none is configured,
//! shared by `pivot server` and `pivot open`.

use dispatch::BUFFER_SIZE;
use dispatch::env::get_env_var_with_default;

/// Bytes in a gibibyte, the unit memory budgets are reported in.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// Share of the machine's physical memory the buffer pool takes when no budget
/// is configured. The `PIVOT_MEMORY_PCT` environment variable overrides it.
pub const DEFAULT_MEMORY_PCT: u64 = 80;

/// Memory held back from the default budget for allocations that do not come
/// from the ring: catalogs, file metadata, plan caches and the like.
///
/// HACK: those allocations still go through the global allocator, so a pool
/// sized to the whole share would leave them fighting the kernel for what is
/// left. Delete this reserve once every catalog allocation is served from the
/// ring and the pool budget is the whole memory budget.
pub const OVERHEAD_RESERVE_BYTES: u64 = 4 * GIB;

/// The machine cannot fit one pool slot next to the overhead reserve.
#[derive(Debug, thiserror::Error)]
#[error(
    "this machine's {} GiB is too little for the default buffer-pool budget ({memory_pct}% of \
     memory minus a {} GiB reserve); set one with `--memory` or `server.memory`",
    .total_bytes / GIB,
    OVERHEAD_RESERVE_BYTES / GIB,
)]
pub struct MachineTooSmall {
    total_bytes: u64,
    memory_pct: u64,
}

/// Read the pool's share of physical memory from `PIVOT_MEMORY_PCT`, falling
/// back to [`DEFAULT_MEMORY_PCT`].
pub fn read_memory_pct() -> u64 {
    get_env_var_with_default("PIVOT_MEMORY_PCT", DEFAULT_MEMORY_PCT)
}

/// The pool budget for a machine with `total_bytes` of physical memory:
/// `memory_pct` of it, minus [`OVERHEAD_RESERVE_BYTES`].
pub fn compute_default_pool_bytes(
    total_bytes: u64,
    memory_pct: u64,
) -> Result<u64, MachineTooSmall> {
    let share = total_bytes * memory_pct / 100;
    let pool_bytes = share.saturating_sub(OVERHEAD_RESERVE_BYTES);
    if pool_bytes < BUFFER_SIZE as u64 {
        return Err(MachineTooSmall {
            total_bytes,
            memory_pct,
        });
    }
    Ok(pool_bytes)
}
