//! The `memory` module contains all dispatch's memory management/utilities.
//!
//! There are a few targets in our memory management:
//!     1. Hit as few page faults as possible
//!     2. Know when we're hitting disk and when we have something cached
//!     3. Allow spilling to disk
//!
//! Memory usage is divided into two- [`WriteBuffer`]s (each 2 mb) for large allocations,
//! regular `malloc` for either small allocations or allocations that necessitate continuous memory.
//!
//! Our assumption is that we're the only important process on the machine; we therefore immediately
//! allocate on initialization half the system memory for [`WriteBuffer`]s, as this will be our main
//! usage of memory. Every worker upon initialization faults in all [`WriteBuffer`]s and pushes equal
//! amounts to local pools for use afterward. These [`WriteBuffer`]s will be used for any disk access
//! (and subsequently saved in [`FILE_CACHE`]) as well as large allocations.
//!
//! [`WriteBuffer`]s can also be used for many miscellaneous things, such as Vectors and HashTables. It
//! is generally preferred to use [`WriteBuffer`]s as the memory is easily accounted for. See [`SlabAllocator`].

mod ring;
pub use ring::{BUFFER_SIZE, Ring};

pub mod file_cache;
pub use file_cache::{CacheLookup, region_base_of};

mod free_pool;

mod write_buffer;
pub use write_buffer::WriteBuffer;

mod read_buffer;
pub use read_buffer::ReadBuffer;

mod slab;
pub use slab::{MultiSlabBuffer, SlabAllocator, SlabBuffer};

mod reader;
pub use reader::{MultiBufferReader, ReaderPosition};

mod context;
pub use context::{MemoryContextFactory, has_memory_context, init_memory_context, memory_ctx};

#[cfg(test)]
pub use context::init_test_free_pool;
