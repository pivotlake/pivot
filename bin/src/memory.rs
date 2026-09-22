//! The buffer-pool budget a `pivot` process takes, shared by `pivot server`
//! and `pivot open`: an absolute size, or a share of the machine's memory.

use std::str::FromStr;

use dispatch::BUFFER_SIZE;
use metastore_disk::ByteSize;
use serde::Deserialize;

/// Bytes in a gibibyte, the unit memory budgets are reported in.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// Share of the machine's physical memory the buffer pool takes when no budget
/// is configured.
pub const DEFAULT_MEMORY_PCT: u64 = 80;

/// Memory held back from a share-of-the-machine budget for allocations that do
/// not come from the ring: catalogs, file metadata, plan caches and the like.
///
/// HACK: those allocations still go through the global allocator, so a pool
/// sized to the whole share would leave them fighting the kernel for what is
/// left. Delete this reserve once every catalog allocation is served from the
/// ring and the pool budget is the whole memory budget.
pub const OVERHEAD_RESERVE_BYTES: u64 = 4 * GIB;

/// A buffer-pool budget as the config file or the command line spells it: an
/// absolute size such as `32g`, or a share of the machine's physical memory
/// such as `80%`, from which [`OVERHEAD_RESERVE_BYTES`] is held back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBudget {
    Bytes(ByteSize),
    Percent(u64),
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::Percent(DEFAULT_MEMORY_PCT)
    }
}

impl FromStr for MemoryBudget {
    type Err = String;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let Some(percent) = input.strip_suffix('%') else {
            return input.parse().map(Self::Bytes);
        };
        let percent: u64 = percent
            .trim()
            .parse()
            .map_err(|_| format!("`{input}` is not a percentage of memory"))?;
        if !(1..=100).contains(&percent) {
            return Err(format!(
                "`{input}` is not a share of memory a pool can take (1% to 100%)"
            ));
        }
        Ok(Self::Percent(percent))
    }
}

impl<'de> Deserialize<'de> for MemoryBudget {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// The machine cannot fit one pool slot next to the overhead reserve.
#[derive(Debug, thiserror::Error)]
#[error(
    "this machine's {} GiB is too little for a buffer-pool budget of {memory_pct}% of memory \
     minus a {} GiB reserve; set an absolute one with `--memory` or the `memory` config key",
    .total_bytes / GIB,
    OVERHEAD_RESERVE_BYTES / GIB,
)]
pub struct MachineTooSmall {
    total_bytes: u64,
    memory_pct: u64,
}

impl MemoryBudget {
    /// The pool's byte budget on a machine with `total_bytes` of physical
    /// memory: the size as written, or the share of `total_bytes` minus
    /// [`OVERHEAD_RESERVE_BYTES`].
    pub fn resolve(self, total_bytes: u64) -> Result<u64, MachineTooSmall> {
        match self {
            Self::Bytes(size) => Ok(size.as_bytes()),
            Self::Percent(memory_pct) => {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_size_is_taken_as_written() {
        let budget: MemoryBudget = "32g".parse().unwrap();

        assert_eq!(budget, MemoryBudget::Bytes(ByteSize::from_bytes(32 * GIB)));
        assert_eq!(budget.resolve(8 * GIB).unwrap(), 32 * GIB);
    }

    #[test]
    fn a_percentage_is_a_share_of_the_machine_minus_the_reserve() {
        let budget: MemoryBudget = "50%".parse().unwrap();

        assert_eq!(budget, MemoryBudget::Percent(50));
        assert_eq!(
            budget.resolve(64 * GIB).unwrap(),
            32 * GIB - OVERHEAD_RESERVE_BYTES
        );
    }

    #[test]
    fn the_default_is_most_of_the_machine() {
        assert_eq!(MemoryBudget::default(), MemoryBudget::Percent(80));
    }

    #[test]
    fn a_percentage_outside_the_pool_range_is_rejected() {
        assert!("0%".parse::<MemoryBudget>().is_err());
        assert!("101%".parse::<MemoryBudget>().is_err());
        assert!("%".parse::<MemoryBudget>().is_err());
        assert!("eighty%".parse::<MemoryBudget>().is_err());
    }
}
