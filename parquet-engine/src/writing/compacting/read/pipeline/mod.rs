//! The operators a compaction's read adds between the operators a scan
//! fetches, decompresses and decodes with: the two gates that hold a read's
//! pages and ranges of rows until the compacter releases the rows, what the
//! gates and the compacter share, and the counter that makes the memory
//! budget count a decompressed page.

pub(crate) mod decode_gate;
pub(crate) mod decompress_gate;
pub(crate) mod gated_reads;
pub(crate) mod page_memory_counter;
