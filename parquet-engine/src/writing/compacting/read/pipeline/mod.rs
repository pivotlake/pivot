//! The operators a compaction's read adds between the operators a scan
//! fetches, decompresses and decodes with: the gate that lets a read's
//! compressed pages on, and the cutter that cuts its decompressed pages,
//! as the compacter releases the rows, what those and the compacter share,
//! and the counter that makes the memory budget count a decompressed page.
//!
//! Neither the gate nor the cutter holds the pages that wait for their
//! rows. Each keeps where such a page lies in its file and lets go of its
//! bytes, which the compressed and the decompressed cache keep for as long
//! as nothing else needs their memory. A page whose bytes are gone when
//! its rows are released is read again.

pub(crate) mod decompress_gate;
pub(crate) mod gated_reads;
pub(crate) mod page_memory_counter;
pub(crate) mod released_rows_cutter;
