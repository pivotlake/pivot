//! What a ring slot was taken for, so a census can say which stage of the
//! pipeline is holding the pool rather than only how much of it is held.
//!
//! A worker sets the tag around the work it runs ([`tagged`]); every buffer it
//! acquires while the guard is alive records it. The tag rides on the slot, so
//! it is still there when another worker reads the ring, and it is read only
//! for a slot a writer holds - a freed slot's tag is last time's, and nothing
//! looks at it.

use std::cell::Cell;

/// The stage a ring slot was taken for. `Other` covers everything no guard
/// names: ingest, query operators, the caches' own fills.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MemoryTag {
    Other = 0,
    /// Decoding a column chunk's pages into arrays.
    Decode = 1,
    /// Bringing a compaction input's batches to the union layout.
    Widen = 2,
    /// Folding a shredded batch back to whole documents.
    Unshred = 3,
    /// The k-way merge of one node's runs.
    LocalMerge = 4,
    /// The k-way merge across nodes.
    GlobalMerge = 5,
    /// Applying a file's shredding layout to a slice of its rows.
    Shred = 6,
    /// Encoding a leaf's values into Parquet pages.
    Encode = 7,
    /// Holding a file's encoded bytes until it is uploaded.
    Assemble = 8,
    /// Decompressing a column chunk's pages.
    Decompress = 9,
    /// Holding the rows gathered for one output file until its layout is
    /// decided and its row groups are planned.
    Collect = 10,
}

pub const TAG_COUNT: usize = 11;

impl MemoryTag {
    pub fn name(self) -> &'static str {
        match self {
            MemoryTag::Other => "other",
            MemoryTag::Decode => "decode",
            MemoryTag::Widen => "widen",
            MemoryTag::Unshred => "unshred",
            MemoryTag::LocalMerge => "local_merge",
            MemoryTag::GlobalMerge => "global_merge",
            MemoryTag::Shred => "shred",
            MemoryTag::Encode => "encode",
            MemoryTag::Assemble => "assemble",
            MemoryTag::Decompress => "decompress",
            MemoryTag::Collect => "collect",
        }
    }

    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => MemoryTag::Decode,
            2 => MemoryTag::Widen,
            3 => MemoryTag::Unshred,
            4 => MemoryTag::LocalMerge,
            5 => MemoryTag::GlobalMerge,
            6 => MemoryTag::Shred,
            7 => MemoryTag::Encode,
            8 => MemoryTag::Assemble,
            9 => MemoryTag::Decompress,
            10 => MemoryTag::Collect,
            _ => MemoryTag::Other,
        }
    }
}

thread_local! {
    static CURRENT_TAG: Cell<u8> = const { Cell::new(0) };
}

/// The tag this thread's buffers are taken under right now.
pub fn current_tag() -> u8 {
    CURRENT_TAG.get()
}

/// Tag every buffer this thread takes until the guard drops. Restores whatever
/// tag was set before, so an inner stage nested in an outer one gives the
/// buffers back to the outer stage when it returns.
pub fn tagged(tag: MemoryTag) -> TagGuard {
    let previous = CURRENT_TAG.replace(tag as u8);
    TagGuard { previous }
}

pub struct TagGuard {
    previous: u8,
}

impl Drop for TagGuard {
    fn drop(&mut self) {
        CURRENT_TAG.set(self.previous);
    }
}
