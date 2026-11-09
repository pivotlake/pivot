//! Per-partition rows produced by radix scatter and consumed by merge.
//!
//! Every value stores its rows at the hash table entry layout:
//!
//! ```text
//! hash | key | stored value
//! ```
//!
//! For a compiled value the layout is fully constant; for a dynamic value the
//! stride comes from the query. Values are seeded directly in the destination
//! row either way.
//!
//! Rows are written through a [`PartitionAllocator`], which keeps every
//! partition's rows in slabs that the partitions of a group share, and which
//! can spill sealed buffers to disk. Once a worker is done consuming, the
//! allocator is taken apart into each partition's [`PartitionRows`], which the
//! merge reads.

use super::hash_table::{PersistedKey, entry_layout};
use crate::io::{FileRange, IORequesterError, LocalFile, ReadData};
use crate::memory::{
    BUFFER_SIZE, Dynamic, PARTITIONS, PartitionAllocator, PartitionSlab, SpilledBuffer,
    SpilledSlab, Spiller,
};
use crate::operations::unary::group::values::AggregationValue;
use std::marker::PhantomData;

/// Target size of a partition's slabs once it has filled a few, and of the
/// first one, which is kept small so a partition with a handful of rows wastes
/// little.
const SLAB_BYTES: usize = 64 * 1024;
const FIRST_SLAB_BYTES: usize = SLAB_BYTES / 8;

/// Copyable layout for scatter rows.
///
/// The caller holds one layout value for all partitions. Keeping it out of
/// each partition's state keeps the layout fields local in the scatter loop.
pub struct ScatterLayout<V: AggregationValue + ?Sized> {
    hash_offset: usize,
    key_offset: usize,
    value_offset: usize,
    stride: usize,
    align: usize,
    metadata: V::StorageMetadata,
}

impl<V: AggregationValue + ?Sized> Clone for ScatterLayout<V> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<V: AggregationValue + ?Sized> Copy for ScatterLayout<V> {}

impl<V: AggregationValue + ?Sized> ScatterLayout<V> {
    /// Bytes from one row to the next.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The alignment a row's address must have for its fields to be read in
    /// place.
    pub fn align(&self) -> usize {
        self.align
    }

    /// The row layout shared by every partition.
    ///
    /// `N` is the specialized slot count of the surrounding
    /// [`dispatch_arity`](AggregationValue::dispatch_arity) body: a nonzero
    /// `N` derives the layout from that constant so it folds, while `N == 0`
    /// reads the runtime metadata off the shared context.
    pub fn new<KP: PersistedKey, const N: usize>(ctx: &V::SharedContext) -> Self {
        let metadata = if N == 0 {
            V::storage_metadata(ctx)
        } else {
            V::metadata_for_arity::<N>()
        };
        let layout = entry_layout::<KP, V>(metadata);
        ScatterLayout {
            hash_offset: layout.hash_offset,
            key_offset: layout.key_offset,
            value_offset: layout.value_offset,
            stride: layout.stride,
            align: layout.align,
            metadata,
        }
    }
}

/// Where a worker scatters rows while it consumes.
pub struct ScatterRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    allocator: PartitionAllocator<Dynamic>,
    /// How many partitions rows are scattered into; the allocator can hold
    /// [`PARTITIONS`], of which only the first this many are used.
    partition_count: usize,
    row_size: usize,
    row_align: usize,
    // A raw-pointer marker: a plain `(KP, V)` tuple would require `V: Sized`.
    _phantom: PhantomData<(KP, *const V)>,
}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> ScatterRows<KP, V> {
    pub fn new(layout: ScatterLayout<V>, partition_count: usize) -> Self {
        assert!(
            partition_count <= PARTITIONS,
            "{partition_count} scatter partitions, more than the {PARTITIONS} the allocator holds"
        );
        Self {
            allocator: PartitionAllocator::new(Dynamic::new(
                layout.stride,
                layout.align,
                FIRST_SLAB_BYTES,
                SLAB_BYTES,
            )),
            partition_count,
            row_size: layout.stride,
            row_align: layout.align,
            _phantom: PhantomData,
        }
    }

    /// Spill the next sealed buffer, if there is one. See
    /// [`PartitionAllocator::spill`].
    pub fn spill(
        &mut self,
        spiller: &mut Spiller,
    ) -> Result<Option<SpilledBuffer>, IORequesterError> {
        self.allocator.spill(spiller)
    }

    /// Appends a row to `partition` and initializes its value in place.
    #[inline(always)]
    pub fn push_with(
        &mut self,
        layout: ScatterLayout<V>,
        partition: usize,
        hash: u64,
        key: KP,
        seed: impl FnOnce(&mut V),
    ) {
        let row = self.allocator.reserve(partition).as_mut_ptr();
        // Field offsets come from the caller's shared layout snapshot, and the
        // row is `stride` bytes aligned for its fields.
        unsafe {
            *(row.add(layout.hash_offset) as *mut u64) = hash;
            (row.add(layout.key_offset) as *mut KP).write(key);
            seed(V::from_entry_mut(
                row.add(layout.value_offset),
                layout.metadata,
            ));
        }
    }

    /// Every partition's rows, in partition order, for the merge to read: the
    /// rows still in the allocator, and the partition's spilled slabs as weak
    /// handles (`spilled[partition]`), which the merge resolves when it gets
    /// to the partition.
    pub fn into_partition_rows(
        self,
        mut spilled: Vec<Vec<SpilledSlab>>,
    ) -> Vec<PartitionRows<KP, V>> {
        let row_size = self.row_size;
        let row_align = self.row_align;
        self.allocator
            .into_partitions()
            .into_iter()
            .enumerate()
            .take(self.partition_count)
            .map(|(partition, contents)| {
                PartitionRows::new(
                    contents.slabs,
                    std::mem::take(&mut spilled[partition]),
                    contents.spill_file,
                    row_size,
                    row_align,
                )
            })
            .collect()
    }
}

/// The rows of one partition that exist only on disk: ranges of the
/// partition's spill file, each holding whole rows back to back and no
/// longer than a ring buffer, so that one slab takes it.
#[derive(Clone)]
pub struct OnDiskRows {
    pub file: LocalFile,
    pub ranges: Vec<FileRange>,
    /// The alignment memory must have for a range read into it to be read
    /// as rows in place.
    pub row_align: usize,
}

/// Join ranges that follow one another in the file into single ranges of at
/// most `longest` bytes each, so that what was written as many slabs is read
/// back in as few reads as the bytes allow. No range is split: one joined
/// range holds whole input ranges.
fn join_adjacent_ranges(mut ranges: Vec<FileRange>, longest: usize) -> Vec<FileRange> {
    ranges.sort_unstable_by_key(|range| range.offset);
    let mut joined: Vec<FileRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match joined.last_mut() {
            Some(last)
                if last.offset + last.len == range.offset && last.len + range.len <= longest =>
            {
                last.len += range.len;
            }
            _ => joined.push(range),
        }
    }
    joined
}

/// One partition's scatter rows, as the merge reads them: the slabs they were
/// written into, each with how many rows it holds, and the file ranges of the
/// rows that are only on disk, which the merge loads before it runs.
pub struct PartitionRows<KP: PersistedKey, V: AggregationValue + ?Sized> {
    slabs: Vec<PartitionSlab>,
    /// The partition's spilled slabs, not yet resolved into `slabs` or
    /// `on_disk`; see [`resolve_spilled`](Self::resolve_spilled).
    spilled: Vec<SpilledSlab>,
    /// The file the partition's spilled rows went to, if it ever spilled.
    spill_file: Option<LocalFile>,
    on_disk: Option<OnDiskRows>,
    /// Rows in memory and on disk together.
    rows: usize,
    row_size: usize,
    row_align: usize,
    _phantom: PhantomData<(KP, *const V)>,
}

// SAFETY: the slabs are frozen, so reading them from any thread is sound.
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Send for PartitionRows<KP, V> {}
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Sync for PartitionRows<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> PartitionRows<KP, V> {
    fn new(
        slabs: Vec<PartitionSlab>,
        spilled: Vec<SpilledSlab>,
        spill_file: Option<LocalFile>,
        row_size: usize,
        row_align: usize,
    ) -> Self {
        let rows_in_allocator: usize = slabs.iter().map(|slab| slab.rows).sum();
        let bytes_spilled: usize = spilled.iter().map(|handle| handle.slab.size()).sum();
        Self {
            rows: rows_in_allocator + bytes_spilled / row_size,
            slabs,
            spilled,
            spill_file,
            on_disk: None,
            row_size,
            row_align,
            _phantom: PhantomData,
        }
    }

    /// Settle where the partition's spilled rows are read from, now that the
    /// merge is about to read them: a slab whose buffer is still resident is
    /// read from memory, and holding the upgraded handle keeps it there until
    /// these rows are dropped; a slab whose buffer was evicted is loaded from
    /// the partition's file. A partition's slabs were appended to its file
    /// one after another, so evicted slabs that are neighbours there are
    /// loaded as one range.
    pub fn resolve_spilled(&mut self) {
        let mut ranges = Vec::new();
        for handle in self.spilled.drain(..) {
            match handle.slab.upgrade() {
                Some(slab) => self.slabs.push(PartitionSlab {
                    rows: slab.size() / self.row_size,
                    slab,
                }),
                None => ranges.push(FileRange::new(
                    handle.file_offset as usize,
                    handle.slab.size(),
                )),
            }
        }
        if !ranges.is_empty() {
            self.on_disk = Some(OnDiskRows {
                file: self
                    .spill_file
                    .clone()
                    .expect("a partition with rows on disk has a spill file"),
                ranges: join_adjacent_ranges(ranges, BUFFER_SIZE),
                row_align: self.row_align,
            });
        }
    }

    /// Number of rows in the partition, in memory and on disk together.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// The rows that have to be loaded from disk before the partition can be
    /// merged, if any.
    pub fn on_disk(&self) -> Option<&OnDiskRows> {
        self.on_disk.as_ref()
    }

    /// Visits every row in memory. Rows on disk are visited through
    /// [`for_each_loaded`](Self::for_each_loaded) once they have been read.
    #[inline(always)]
    pub fn for_each(&self, layout: ScatterLayout<V>, mut f: impl FnMut(u64, &KP, &V)) {
        for slab in &self.slabs {
            let mut row = slab.slab.as_slice().as_ptr();
            for _ in 0..slab.rows {
                unsafe {
                    let hash = *(row.add(layout.hash_offset) as *const u64);
                    let key = &*(row.add(layout.key_offset) as *const KP);
                    f(
                        hash,
                        key,
                        V::from_entry(row.add(layout.value_offset), layout.metadata),
                    );
                    row = row.add(layout.stride);
                }
            }
        }
    }

    /// Visits every row in `loaded`, the data read back for this partition's
    /// [`on_disk`](Self::on_disk) ranges. Each range holds whole rows, but the
    /// read may hand a range back in several pieces, so a row that straddles
    /// two pieces, or sits in a piece whose address is not aligned for a row,
    /// is copied into `row_scratch` first. The scratch must hold one row and
    /// be aligned for it.
    pub fn for_each_loaded(
        layout: ScatterLayout<V>,
        loaded: &[ReadData],
        row_scratch: &mut [u8],
        mut f: impl FnMut(u64, &KP, &V),
    ) {
        assert!(row_scratch.len() >= layout.stride);
        assert!(row_scratch.as_ptr() as usize % layout.align == 0);
        let mut visit = |row: *const u8| unsafe {
            let hash = *(row.add(layout.hash_offset) as *const u64);
            let key = &*(row.add(layout.key_offset) as *const KP);
            f(
                hash,
                key,
                V::from_entry(row.add(layout.value_offset), layout.metadata),
            );
        };
        for range in loaded {
            let mut pieces = range.bytes.iter().map(|piece| piece.as_ref()).peekable();
            // Bytes of a row that began at the end of the previous piece.
            let mut carried = 0;
            while let Some(piece) = pieces.next() {
                let mut rest = piece;
                if carried > 0 {
                    let needed = layout.stride - carried;
                    row_scratch[carried..layout.stride].copy_from_slice(&rest[..needed]);
                    visit(row_scratch.as_ptr());
                    rest = &rest[needed..];
                    carried = 0;
                }
                let whole_rows = rest.len() / layout.stride;
                let aligned = rest.as_ptr() as usize % layout.align == 0;
                for row in rest.chunks_exact(layout.stride).take(whole_rows) {
                    if aligned {
                        visit(row.as_ptr());
                    } else {
                        row_scratch[..layout.stride].copy_from_slice(row);
                        visit(row_scratch.as_ptr());
                    }
                }
                let tail = &rest[whole_rows * layout.stride..];
                if !tail.is_empty() {
                    assert!(
                        pieces.peek().is_some(),
                        "a range read back from a spill file holds whole rows"
                    );
                    row_scratch[..tail.len()].copy_from_slice(tail);
                    carried = tail.len();
                }
            }
        }
    }

    /// Visits rows and exposes a future row for software prefetching.
    #[inline(always)]
    pub fn for_each_prefetched<const AHEAD: usize>(
        &self,
        layout: ScatterLayout<V>,
        mut f: impl FnMut(u64, &KP, &V, Option<(u64, &KP)>),
    ) {
        for slab in &self.slabs {
            let mut row = slab.slab.as_slice().as_ptr();
            for i in 0..slab.rows {
                unsafe {
                    let hash = *(row.add(layout.hash_offset) as *const u64);
                    let key = &*(row.add(layout.key_offset) as *const KP);
                    let ahead = (i + AHEAD < slab.rows).then(|| {
                        let ahead_row = row.add(AHEAD * layout.stride);
                        (
                            *(ahead_row.add(layout.hash_offset) as *const u64),
                            &*(ahead_row.add(layout.key_offset) as *const KP),
                        )
                    });
                    f(
                        hash,
                        key,
                        V::from_entry(row.add(layout.value_offset), layout.metadata),
                        ahead,
                    );
                    row = row.add(layout.stride);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::group::values::{Compiled, CountSlot};
    use bytes::Bytes;

    type CountValue = Compiled<(CountSlot,), u8>;

    #[test]
    fn ranges_that_follow_one_another_are_joined_up_to_the_longest_allowed() {
        let ranges = vec![
            FileRange::new(300, 50),
            FileRange::new(0, 100),
            FileRange::new(100, 100),
            FileRange::new(200, 100),
            FileRange::new(400, 10),
        ];

        let joined = join_adjacent_ranges(ranges, 250);

        assert_eq!(
            joined,
            [
                FileRange::new(0, 200),
                FileRange::new(200, 150),
                FileRange::new(400, 10)
            ]
        );
    }

    /// Rows of `hash | key | count` as they would be read back from a spill
    /// file: `rows` of them, cut into pieces at `cuts`.
    fn rows_read_back(layout: ScatterLayout<CountValue>, rows: u64, cuts: &[usize]) -> ReadData {
        let mut bytes = vec![0u8; layout.stride() * rows as usize];
        for row in 0..rows {
            let start = row as usize * layout.stride();
            bytes[start + layout.hash_offset..start + layout.hash_offset + 8]
                .copy_from_slice(&row.to_ne_bytes());
            bytes[start + layout.key_offset..start + layout.key_offset + 4]
                .copy_from_slice(&(row as i32 * 10).to_ne_bytes());
        }
        let mut pieces = Vec::new();
        let mut from = 0;
        for &cut in cuts {
            pieces.push(Bytes::copy_from_slice(&bytes[from..cut]));
            from = cut;
        }
        pieces.push(Bytes::copy_from_slice(&bytes[from..]));
        ReadData {
            offset: 0,
            bytes: pieces,
        }
    }

    #[test]
    fn rows_read_back_in_pieces_are_visited_whole_and_in_order() {
        let layout = ScatterLayout::<CountValue>::new::<i32, 0>(&());
        // Cut inside the second row and exactly on the boundary after the third.
        let loaded = [rows_read_back(
            layout,
            5,
            &[layout.stride() + 3, 3 * layout.stride()],
        )];
        // Scratch aligned for any row: `u128` words cover every field type.
        let mut scratch_words = vec![0u128; layout.stride().div_ceil(16)];
        let row_scratch = unsafe {
            std::slice::from_raw_parts_mut(
                scratch_words.as_mut_ptr() as *mut u8,
                scratch_words.len() * 16,
            )
        };
        let mut seen = Vec::new();

        PartitionRows::<i32, CountValue>::for_each_loaded(
            layout,
            &loaded,
            row_scratch,
            |hash, key, _| seen.push((hash, *key)),
        );

        assert_eq!(seen, [(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]);
    }
}
