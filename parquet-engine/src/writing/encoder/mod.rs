//! Encodes one row-group leaf at a time as Parquet pages.
//!
//! Each [`LeafChunkJob`] becomes one [`EncodedLeafChunk`]. Jobs run in
//! parallel, then the output channel routes each result to its file's assembly
//! worker.
//!
//! The [`shredder`](super::shredder) has already split every column into its
//! primitive leaves, so the encoder's input is one leaf's values in row-ordered
//! chunks. It materializes them into ring memory and encodes the result.
//!
//! Encoding prefers a dictionary when it is beneficial, then a supported delta
//! encoding, and finally PLAIN. [`pages`] frames and compresses the output.
//!
//! A leaf of a large row group is many pages, and encoding and compressing
//! them all takes long enough to hold up everything else its worker runs. So
//! the stage keeps the leaf it is encoding and encodes one page of it per
//! turn, letting the worker's other operators run in between.

mod column_index;
pub(crate) mod delta;
mod dictionary;
mod pages;
mod plain;
mod rle;

use std::sync::Arc;

use crate::thrift::footer::Statistics;
use crate::thrift::general::{CompressionCodec, Encoding};
use dispatch::memory::SlabAllocator;
use dispatch::{DefaultUnaryFactory, Sender, Unary, UnaryResult, WorkStatus};

use super::compression;
use super::error::{WriteError, WriteResult};
use super::leaves::Leaf;
use super::stats;
use super::types::{EncodedLeaf, EncodedLeafChunk, EncodedPage, LeafChunkJob, RowGroupContext};
use dispatch::arrays::concat_chunks;
use pages::PageRange;

pub(super) type LeafEncoderFactory = DefaultUnaryFactory<LeafEncoder>;

pub(super) fn factories(worker_count: usize) -> Vec<LeafEncoderFactory> {
    DefaultUnaryFactory::create_for_workers(worker_count)
}

/// The leaf a [`LeafEncoder`] is part-way through.
struct HeldLeaf {
    context: Arc<RowGroupContext>,
    leaf_index: usize,
    encoding: LeafEncoding,
}

#[derive(Default)]
pub(super) struct LeafEncoder {
    /// Holds a leaf's values while it is being encoded. Initialized on first
    /// use so an inactive encoder holds no ring buffer.
    value_allocator: Option<SlabAllocator>,
    /// Holds the encoded pages, which live until their row group is written.
    /// It is separate from the values' so that a page shares its ring buffer
    /// only with other pages: a page that shared one with a leaf's values
    /// would keep the whole buffer in memory long after the values are gone.
    page_allocator: Option<SlabAllocator>,
    /// The leaf being encoded. While one is held, no other is taken.
    held: Option<HeldLeaf>,
}

impl LeafEncoder {
    /// Send the held leaf on if its last page is encoded.
    fn send_if_finished(&mut self, sender: &mut dyn Sender<EncodedLeafChunk>) -> UnaryResult<()> {
        if self
            .held
            .as_ref()
            .is_some_and(|held| held.encoding.is_finished())
        {
            let held = self.held.take().unwrap();
            sender.send(EncodedLeafChunk {
                context: held.context,
                leaf_index: held.leaf_index,
                leaf: held.encoding.finish(),
            })?;
        }
        Ok(())
    }
}

impl Unary<LeafChunkJob, EncodedLeafChunk> for LeafEncoder {
    fn consume(
        &mut self,
        job: LeafChunkJob,
        sender: &mut dyn Sender<EncodedLeafChunk>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        let value_allocator = self
            .value_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let values = concat_chunks(value_allocator, &job.value_chunks).map_err(WriteError::from)?;
        let page_allocator = self
            .page_allocator
            .get_or_insert_with(|| SlabAllocator::new(false));
        let leaf = Leaf {
            path: job.path,
            values,
            def_levels: job.def_levels.map(Arc::from),
            max_def_level: job.max_def_level,
        };
        self.held = Some(HeldLeaf {
            context: job.context,
            leaf_index: job.leaf_index,
            encoding: LeafEncoding::start(leaf, page_allocator)?,
        });
        self.send_if_finished(sender)
    }

    /// Encode the held leaf's next page.
    fn run(
        &mut self,
        sender: &mut dyn Sender<EncodedLeafChunk>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<WorkStatus> {
        let Some(held) = self.held.as_mut() else {
            return Ok(WorkStatus::Pending);
        };
        let page_allocator = self
            .page_allocator
            .as_mut()
            .expect("a held leaf was started with the page allocator");
        held.encoding.encode_next_page(page_allocator)?;
        self.send_if_finished(sender)?;
        Ok(WorkStatus::Ran)
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.held.is_none()
    }

    /// A held leaf keeps the stage from finishing until its pages are out.
    fn has_pending_work(&self) -> bool {
        self.held.is_some()
    }
}

/// One leaf being encoded a page at a time.
///
/// The encoding prefers a dictionary, then a delta form, then PLAIN, in the
/// order of what each costs the reader. A dictionary is best where it fits:
/// the values are stored once and the column becomes small integers, which
/// also lets a reader prune a row group by comparing against the dictionary
/// alone. Where it does not fit, a delta form packs the differences to the
/// width they need, which is most of the size of a key column. Floats, and
/// decimals too wide to store as an integer, have no delta form and take
/// PLAIN.
pub(in crate::writing) struct LeafEncoding {
    leaf: Leaf,
    physical_type: i32,
    codec: CompressionCodec,
    statistics: Statistics,
    dictionary_page: Option<EncodedPage>,
    data_page_encoding: Encoding,
    data_pages: Vec<EncodedPage>,
    /// The pages still to encode, in order.
    remaining_pages: std::vec::IntoIter<PageRange>,
}

impl LeafEncoding {
    /// Choose `leaf`'s encoding. A leaf that takes a dictionary is encoded
    /// here whole, since its one page of indices is small; any other leaf has
    /// its pages cut and none encoded yet.
    pub(in crate::writing) fn start(
        leaf: Leaf,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<Self> {
        let physical_type = crate::arrow_to_parquet_physical(leaf.values.data_type())?;
        let codec = compression::page_codec();
        let statistics = leaf_statistics(&leaf);
        let (dictionary_page, data_page_encoding, data_pages, remaining_pages) =
            match dictionary::try_encode(&leaf, codec, allocator)? {
                Some((dictionary_page, index_page)) => (
                    Some(dictionary_page),
                    Encoding::RLE_DICTIONARY,
                    vec![index_page],
                    Vec::new(),
                ),
                None => (
                    None,
                    delta::encoding_for(leaf.values.data_type()).unwrap_or(Encoding::PLAIN),
                    Vec::new(),
                    pages::page_ranges(&leaf)?,
                ),
            };
        Ok(Self {
            leaf,
            physical_type,
            codec,
            statistics,
            dictionary_page,
            data_page_encoding,
            data_pages,
            remaining_pages: remaining_pages.into_iter(),
        })
    }

    /// Whether every page is encoded.
    pub(in crate::writing) fn is_finished(&self) -> bool {
        self.remaining_pages.len() == 0
    }

    /// Encode the next page still to encode.
    pub(in crate::writing) fn encode_next_page(
        &mut self,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<()> {
        let range = self
            .remaining_pages
            .next()
            .expect("a leaf that is not finished has a page to encode");
        let page = match self.data_page_encoding {
            Encoding::PLAIN => plain::encode_data_page(&self.leaf, range, self.codec, allocator)?,
            encoding => {
                delta::encode_data_page(&self.leaf, range, encoding, self.codec, allocator)?
            }
        };
        self.data_pages.push(page);
        Ok(())
    }

    /// The encoded leaf, once every page is encoded.
    pub(in crate::writing) fn finish(self) -> EncodedLeaf {
        assert!(self.is_finished(), "every page of the leaf is encoded");
        let column_index =
            column_index::build_column_index(&self.leaf, &self.statistics, &self.data_pages);
        EncodedLeaf {
            path: self.leaf.path,
            physical_type: self.physical_type,
            statistics: self.statistics,
            dictionary_page: self.dictionary_page,
            data_page_encoding: self.data_page_encoding,
            data_pages: self.data_pages,
            column_index,
            codec: self.codec,
        }
    }
}

/// Encode one leaf in a single call, every page of it.
#[cfg(test)]
pub(in crate::writing) fn encode_leaf(
    leaf: Leaf,
    allocator: &mut SlabAllocator,
) -> WriteResult<EncodedLeaf> {
    let mut encoding = LeafEncoding::start(leaf, allocator)?;
    while !encoding.is_finished() {
        encoding.encode_next_page(allocator)?;
    }
    Ok(encoding.finish())
}

/// This leaf's footer statistics: the range of the values it stores, and how many
/// of its rows store none at all.
///
/// Every leaf gets them, which is what lets a reader prune row groups by any
/// column instead of only by the sort key. The null count is recorded even for a
/// leaf with no range to give (a leaf absent on every row has no values to take
/// one from), because it prunes on its own account: a shredded path's typed leaf
/// may only be trusted when every `value` fallback beside it is all-null, and the
/// null count is how a reader establishes that.
fn leaf_statistics(leaf: &Leaf) -> Statistics {
    let (min_value, max_value) = match stats::column_min_max(&leaf.values) {
        Some((min, max)) => (stats::stat_bytes(&min), stats::stat_bytes(&max)),
        None => (None, None),
    };
    Statistics {
        // The deprecated pair, superseded by `min_value`/`max_value`.
        min: None,
        max: None,
        // Absent rows: the leaf spans every row, but stores only the present.
        null_count: Some((leaf.rows() - leaf.values.len()) as i64),
        distinct_count: None,
        // The bounds are recorded whole, never truncated.
        is_max_value_exact: max_value.as_ref().map(|_| true),
        is_min_value_exact: min_value.as_ref().map(|_| true),
        min_value,
        max_value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;

    /// A leaf of several pages is encoded one page per call, so the stage can
    /// yield its worker between them.
    #[test]
    fn a_leaf_of_several_pages_takes_one_call_per_page() {
        dispatch::memory::init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        // Distinct eight-byte values: no dictionary, and four megabytes of
        // them at a megabyte a page.
        let values: Vec<f64> = (0..524_288).map(f64::from).collect();
        let leaf = Leaf {
            path: vec!["value".to_string()],
            values: Arc::new(Float64Array::from(values)),
            def_levels: None,
            max_def_level: 0,
        };

        let mut encoding = LeafEncoding::start(leaf, &mut allocator).unwrap();
        let mut calls = 0;
        while !encoding.is_finished() {
            encoding.encode_next_page(&mut allocator).unwrap();
            calls += 1;
        }

        assert_eq!(calls, 4);
        assert_eq!(encoding.finish().data_pages.len(), 4);
    }
}
