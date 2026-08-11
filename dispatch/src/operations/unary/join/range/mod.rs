//! Range join: an inner join whose one condition is `<`, `<=`, `>`, or `>=`.
//!
//! A hash table cannot answer "which build keys are smaller than this one",
//! but sorted data can: every probe row's matches are one contiguous span of
//! the build side sorted by key. So the build input is run through the ORDER
//! BY operator, its sorted chunks are funneled to a single worker (a
//! [`to_single_worker_mpsc`] channel, whose one consumer sees the chunks in
//! the order the sort emitted them) and published as-is, and the probe finds
//! each row's matching span with a binary search over the chunks' first keys
//! followed by one inside a chunk. The spans are slices of already-sorted,
//! already-materialized batches, so the probe emits them with bulk range
//! copies; no row ids and no gathering.
//!
//! NULL keys never satisfy a comparison. The sort puts them last, so the
//! published chunks simply stop at the first NULL; NULL probe keys are
//! filtered before searching.
//!
//! Probe input is gated on a readiness flag flipped when the chunks publish,
//! exactly as for the hash join.
//!
//! [`to_single_worker_mpsc`]: crate::operations::channels::to_single_worker_mpsc

mod build;
mod probe;

pub(crate) use build::create_range_join_factories;

use std::sync::Arc;

use arrow_array::PrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::types::ArrowPrimitiveType;
use arrow_schema::Field;

use crate::operations::unary::join::JoinCell;

/// The join's comparison, read as `probe key OP build key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeCompare {
    /// `probe < build`: matches the build keys strictly above the probe key.
    Less,
    /// `probe <= build`.
    LessEq,
    /// `probe > build`: matches the build keys strictly below the probe key.
    Greater,
    /// `probe >= build`.
    GreaterEq,
}

impl RangeCompare {
    /// Whether the boundary splitting matches from non-matches is the *lower*
    /// bound of the probe key (first build key `>=` it). The strict variants
    /// use the upper bound (first build key `>` it), so equal keys land on
    /// their side of the split.
    pub(crate) fn splits_at_lower_bound(self) -> bool {
        matches!(self, RangeCompare::LessEq | RangeCompare::Greater)
    }

    /// Whether the matching span is the keys *above* the boundary (`<`/`<=`)
    /// rather than below it (`>`/`>=`).
    pub(crate) fn matches_key_above_boundary(self) -> bool {
        matches!(self, RangeCompare::Less | RangeCompare::LessEq)
    }
}

/// How a range join is configured beyond the key type it is instantiated for.
/// The field meanings match [`JoinSpec`](super::JoinSpec), restricted to the
/// one shape a range join supports: an inner join on a single key column per
/// side.
#[derive(Debug, Clone)]
pub struct RangeJoinSpec {
    /// Index of the probe input column holding the key.
    pub probe_key_index: usize,
    /// Index of the build input column holding the key.
    pub build_key_index: usize,
    /// The comparison, read as `probe key OP build key`.
    pub compare: RangeCompare,
    /// Indices of probe input columns emitted first.
    pub probe_output_indices: Vec<usize>,
    /// Indices of build input columns emitted second.
    pub build_output_indices: Vec<usize>,
    /// The fields the listed probe columns take in the output.
    pub probe_fields: Vec<Field>,
    /// The fields the listed build columns take in the output.
    pub build_fields: Vec<Field>,
}

/// One published chunk of the sorted build side: its key column (no NULLs,
/// ascending, at or above every key of the previous chunk) and the same rows
/// projected to the join's build output columns.
pub(crate) struct SortedChunk<T: ArrowPrimitiveType> {
    pub(crate) keys: PrimitiveArray<T>,
    pub(crate) output: RecordBatch,
}

impl<T: ArrowPrimitiveType> SortedChunk<T> {
    /// The chunk's smallest key — the chunk-picking binary search's fence.
    pub(crate) fn first_key(&self) -> T::Native {
        self.keys.value(0)
    }
}

/// The sorted build side, handed from the build factories to the probe
/// factories. Written by the one publishing consumer and read by probe only
/// after the readiness flag flips.
pub(crate) struct RangeTable<T: ArrowPrimitiveType> {
    pub(crate) chunks: Arc<JoinCell<Vec<SortedChunk<T>>>>,
}

// Derived `Clone` would demand `T: Clone`; the one field is an `Arc`.
impl<T: ArrowPrimitiveType> Clone for RangeTable<T> {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
        }
    }
}
