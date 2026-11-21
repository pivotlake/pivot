//! What a data manifest says of a table's files, cut into the parts a scan
//! asks for: by partition, and then by column.
//!
//! A manifest never changes, so neither does anything read off it, and a
//! table version is nothing but the manifests its list names. Three kinds of
//! part are read off a manifest, each held on its own by the process-wide
//! [`FileMetadataCache`](crate::file_metadata_cache::FileMetadataCache):
//!
//! - [`ManifestPartitions`], one to a manifest: the partitions the manifest
//!   holds a live file of.
//! - [`ManifestFiles`], one to a partition of a manifest: the partition's
//!   live files in the manifest, in the manifest's order, with what a scan
//!   reads of each besides the bounds of its columns.
//! - [`ColumnBounds`], one to a column of a partition of a manifest: the
//!   lower and the upper bound the manifest records for the column in each
//!   of those files, as two Arrow arrays that stand file for file beside the
//!   partition's [`ManifestFiles`], and whether the manifest counts every
//!   value of the column in a file as NULL.
//!
//! A scan selects a partition's files by going through the manifests that
//! may hold a file of it and checking its filters against the bounds of the
//! columns they compare, an array at a time: each comparison yields a bit
//! for each file, and the bits of a scan's filters are combined a word at a
//! time. The parts of the columns no filter compares are not read, so they
//! are the first the cache lets go of, and what stays held is what scans
//! prune by.
//!
//! Parsing a manifest decodes every entry of it whole, so one parse yields
//! every part of the manifest ([`GroupedManifest`]): a part the cache let go
//! of costs a parse of its manifest to get again.
//!
//! # The types bounds are held in
//!
//! A manifest records a column's bounds in the type the column had in the
//! schema the manifest was written under, one type for every file it lists.
//! Bounds of every type are held as Arrow arrays of their values
//! ([`BoundArrays`]), a missing bound as a null.
//!
//! # How bounds are compared
//!
//! A bound is ordered against a filter's constant as the spec crate orders
//! the two ([`StoredColumn::compare_bound`]). Where the bounds and the
//! constant are of types that order as their values do, booleans, integers,
//! dates, timestamps, floats, decimals of one scale and strings, a whole
//! array is ordered against the constant in one pass over its values, and
//! the comparison the filter makes is chosen before the pass, not for each
//! value. Bounds the spec crate does not compare with any constant, those
//! of a `time`, a `uuid`, a `fixed` or a `binary` column, rule no file out.
//! Any other pairing, a constant of another type than its column's among
//! them, is compared one bound at a time.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Decimal128Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow_buffer::bit_chunk_iterator::BitChunks;
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use iceberg::expr::PredicateOperator;
use iceberg::spec::{
    DataContentType, DataFileFormat, Datum, FormatVersion, Manifest, ManifestEntry, ManifestFile,
    ManifestStatus, PrimitiveLiteral, PrimitiveType, TableMetadata,
};
use parquet_engine::{measure_allocation, measure_shared_allocation};

use crate::file_metadata_cache::MetadataPart;

use crate::files::partition::{Partition, encode_partition};
use crate::files::pruning::{BoundComparison, StoredColumn, may_bounds_satisfy};
use crate::metadata::layout::TableLocation;
use crate::{Error, Result};

/// The side of a file's rows a bound is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundSide {
    Lower,
    Upper,
}

/// What an Arrow buffer's shared record of its allocation takes.
const BUFFER_RECORD_BYTES: usize = 64;

/// The partitions one data manifest holds a live file of, in the order of
/// their keys.
pub(crate) struct ManifestPartitions {
    partitions: Vec<Arc<Partition>>,
    heap_bytes: u64,
}

impl ManifestPartitions {
    fn new(partitions: Vec<Arc<Partition>>) -> Self {
        let heap_bytes = measure_shared_allocation(size_of::<Self>())
            + measure_allocation(size_of_val(partitions.as_slice()))
            + partitions
                .iter()
                .map(|partition| partition.estimate_heap_bytes())
                .sum::<usize>();
        Self {
            partitions,
            heap_bytes: heap_bytes as u64,
        }
    }

    /// Whether the manifest holds a live file of the partition `key`.
    pub(crate) fn holds(&self, key: &[u8]) -> bool {
        self.partitions
            .binary_search_by(|held| held.key().as_ref().cmp(key))
            .is_ok()
    }

    /// The partitions, in the order of their keys.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Arc<Partition>> {
        self.partitions.iter()
    }
}

impl MetadataPart for ManifestPartitions {
    fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// The live files of one partition in one data manifest, in the manifest's
/// order: what a scan reads of each besides the bounds of its columns.
pub(crate) struct ManifestFiles {
    /// The schema the manifest was written under: the fields its files
    /// have, and the types their bounds are recorded in.
    schema_id: i32,
    /// Each file's path as the table names it.
    paths: StringArray,
    sizes: Vec<u64>,
    /// The rows each file holds, deleted ones included.
    record_counts: Vec<u64>,
    /// Where each file's commit stands among the table's commits.
    commit_sequences: Vec<u64>,
    /// The id of the table sort order each file's rows are in, as its
    /// entry records it; null for an entry that says nothing.
    sort_order_ids: Int32Array,
    /// The ids of the fields the manifest records a bound or a null count
    /// of in one of the files, in order: the columns the partition has a
    /// [`ColumnBounds`] of in the manifest.
    stats_field_ids: Vec<i32>,
    /// Whether the manifest counts no NULL of a field in a file: a bit for
    /// each file under each field of `stats_field_ids`, field after field,
    /// so the bits of one field stand together.
    null_free: BooleanBuffer,
    heap_bytes: u64,
}

impl ManifestFiles {
    pub(crate) fn count_files(&self) -> usize {
        self.sizes.len()
    }

    /// The schema the manifest was written under.
    pub(crate) fn schema_id(&self) -> i32 {
        self.schema_id
    }

    /// The path of the file at `position`, as the table names it.
    pub(crate) fn path(&self, position: usize) -> &str {
        self.paths.value(position)
    }

    pub(crate) fn size(&self, position: usize) -> u64 {
        self.sizes[position]
    }

    /// The rows the file at `position` holds, deleted ones included.
    pub(crate) fn record_count(&self, position: usize) -> u64 {
        self.record_counts[position]
    }

    /// Where the commit of the file at `position` stands among the table's
    /// commits.
    pub(crate) fn commit_sequence(&self, position: usize) -> u64 {
        self.commit_sequences[position]
    }

    /// The id of the table sort order the rows of the file at `position`
    /// are in, as its entry records it; `None` for an entry that says
    /// nothing.
    pub(crate) fn sort_order_id(&self, position: usize) -> Option<i32> {
        let sort_order_ids = &self.sort_order_ids;
        sort_order_ids
            .is_valid(position)
            .then(|| sort_order_ids.value(position))
    }

    /// Which of the files hold a row, deleted ones included: a bit for each
    /// file.
    pub(crate) fn select_files_with_rows(&self) -> BooleanBuffer {
        pack_bits(&self.record_counts, |record_count| record_count != 0)
    }

    /// The ids of the fields the partition has a [`ColumnBounds`] of in the
    /// manifest, in order.
    pub(crate) fn stats_field_ids(&self) -> &[i32] {
        &self.stats_field_ids
    }

    /// Where `field_id` stands among
    /// [`stats_field_ids`](Self::stats_field_ids); `None` for a field the
    /// manifest records nothing of in any of the files.
    pub(crate) fn find_stats_field(&self, field_id: i32) -> Option<usize> {
        self.stats_field_ids.binary_search(&field_id).ok()
    }

    /// Whether one of the files `is_selected` has a bit set for may hold
    /// a NULL of the field standing at `field_position` among
    /// [`stats_field_ids`](Self::stats_field_ids): the manifest does not
    /// count none in it. The files are gone through a word at a time.
    pub(crate) fn may_hold_nulls(
        &self,
        is_selected: &BooleanBuffer,
        field_position: usize,
    ) -> bool {
        let file_count = self.count_files();
        let selected = is_selected.bit_chunks();
        let null_free = BitChunks::new(
            self.null_free.values(),
            self.null_free.offset() + field_position * file_count,
            file_count,
        );
        selected
            .iter_padded()
            .zip(null_free.iter_padded())
            .any(|(selected, null_free)| selected & !null_free != 0)
    }
}

impl MetadataPart for ManifestFiles {
    fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// The lower bounds and the upper bounds of one column, file for file, a
/// missing bound as a null.
struct BoundPair<Values> {
    lower: Values,
    upper: Values,
}

impl<Values> BoundPair<Values> {
    /// The bounds on `side`.
    fn on(&self, side: BoundSide) -> &Values {
        match side {
            BoundSide::Lower => &self.lower,
            BoundSide::Upper => &self.upper,
        }
    }
}

/// The bounds one data manifest records for one column, as arrays of their
/// values.
enum BoundArrays {
    Boolean(BoundPair<BooleanArray>),
    /// An `int` or a `date`, as its days.
    Int(BoundPair<Int32Array>),
    /// A `long`, a `time` or a timestamp of either unit, as its count
    /// since midnight or since the epoch.
    Long(BoundPair<Int64Array>),
    Float(BoundPair<Float32Array>),
    Double(BoundPair<Float64Array>),
    /// A decimal's unscaled value. The arrays' own precision and scale say
    /// nothing: the bounds' type holds the column's.
    Decimal(BoundPair<Decimal128Array>),
    String(BoundPair<StringArray>),
    /// A UUID's sixteen bytes, the most significant first.
    Uuid(BoundPair<FixedSizeBinaryArray>),
    /// The bytes of a `fixed` or a `binary` bound, which a writer may have
    /// cut short of its column's length.
    Bytes(BoundPair<BinaryArray>),
}

/// The bounds one data manifest records for one column, with the type it
/// records them in.
struct TypedBounds {
    bound_type: PrimitiveType,
    arrays: BoundArrays,
}

/// What one data manifest records of one column in the files of one
/// partition, file for file beside the partition's [`ManifestFiles`].
pub(crate) struct ColumnBounds {
    /// `None` when the manifest records no bound of the column in any of
    /// the files.
    bounds: Option<TypedBounds>,
    /// Whether the manifest counts every value of the column in a file as
    /// NULL: a bit for each file.
    only_nulls: BooleanBuffer,
    heap_bytes: u64,
}

impl MetadataPart for ColumnBounds {
    fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// What the buffers of `array` take, each at the size an allocation of it
/// takes and with its record of the allocation; the array itself is counted
/// where it stands.
fn measure_array_buffers(array: &dyn Array) -> usize {
    let data = array.to_data();
    let validity = data.nulls().map(|nulls| nulls.buffer());
    data.buffers()
        .iter()
        .chain(validity)
        .map(|buffer| measure_allocation(buffer.capacity()) + BUFFER_RECORD_BYTES)
        .sum()
}

/// What `bits` take outside themselves.
fn measure_bits(bits: &BooleanBuffer) -> usize {
    measure_allocation(bits.inner().capacity()) + BUFFER_RECORD_BYTES
}

/// How a bound stands against a constant, as a comparison asks it to.
#[derive(Clone, Copy)]
enum Standing {
    Below,
    AtMost,
    Equal,
    AtLeast,
    Above,
}

/// A bit for each of `values`: whether `holds` of it. A word of bits is
/// packed from sixty-four values at a time.
fn pack_bits<Value: Copy>(values: &[Value], holds: impl Fn(Value) -> bool) -> BooleanBuffer {
    let mut words: Vec<u64> = Vec::with_capacity(values.len().div_ceil(64));
    for chunk in values.chunks(64) {
        let mut word = 0u64;
        for (bit, value) in chunk.iter().enumerate() {
            word |= u64::from(holds(*value)) << bit;
        }
        words.push(word);
    }
    BooleanBuffer::new(Buffer::from_vec(words), 0, values.len())
}

/// A bit for each of `values`: whether it stands as `standing` asks against
/// the constant `order` orders it against. The comparison is chosen here,
/// once for all the values.
fn pack_standing<Value: Copy>(
    values: &[Value],
    standing: Standing,
    order: impl Fn(Value) -> Ordering,
) -> BooleanBuffer {
    match standing {
        Standing::Below => pack_bits(values, |value| order(value).is_lt()),
        Standing::AtMost => pack_bits(values, |value| order(value).is_le()),
        Standing::Equal => pack_bits(values, |value| order(value).is_eq()),
        Standing::AtLeast => pack_bits(values, |value| order(value).is_ge()),
        Standing::Above => pack_bits(values, |value| order(value).is_gt()),
    }
}

/// [`pack_standing`] for values an array hands out by position: `order`
/// orders the value at a position against the constant.
fn pack_standing_at(
    value_count: usize,
    standing: Standing,
    order: impl Fn(usize) -> Ordering,
) -> BooleanBuffer {
    match standing {
        Standing::Below => BooleanBuffer::collect_bool(value_count, |at| order(at).is_lt()),
        Standing::AtMost => BooleanBuffer::collect_bool(value_count, |at| order(at).is_le()),
        Standing::Equal => BooleanBuffer::collect_bool(value_count, |at| order(at).is_eq()),
        Standing::AtLeast => BooleanBuffer::collect_bool(value_count, |at| order(at).is_ge()),
        Standing::Above => BooleanBuffer::collect_bool(value_count, |at| order(at).is_gt()),
    }
}

/// `bits` where `validity` holds a value, and nothing where it holds a
/// null: a missing bound stands in no way against a constant.
fn keep_valid(bits: BooleanBuffer, validity: Option<&NullBuffer>) -> BooleanBuffer {
    match validity {
        Some(validity) => &bits & validity.inner(),
        None => bits,
    }
}

impl TypedBounds {
    fn estimate_heap_bytes(&self) -> usize {
        fn measure_pair<Values: Array>(pair: &BoundPair<Values>) -> usize {
            measure_array_buffers(&pair.lower) + measure_array_buffers(&pair.upper)
        }
        match &self.arrays {
            BoundArrays::Boolean(pair) => measure_pair(pair),
            BoundArrays::Int(pair) => measure_pair(pair),
            BoundArrays::Long(pair) => measure_pair(pair),
            BoundArrays::Float(pair) => measure_pair(pair),
            BoundArrays::Double(pair) => measure_pair(pair),
            BoundArrays::Decimal(pair) => measure_pair(pair),
            BoundArrays::String(pair) => measure_pair(pair),
            BoundArrays::Uuid(pair) => measure_pair(pair),
            BoundArrays::Bytes(pair) => measure_pair(pair),
        }
    }

    /// A bit for each file: whether its bound on `side` is recorded and
    /// stands as `standing` asks against the constant of `comparison`,
    /// which orders bounds of these arrays' type as their values order.
    fn select_standing(
        &self,
        side: BoundSide,
        standing: Standing,
        comparison: &BoundComparison<'_>,
    ) -> BooleanBuffer {
        match (&self.arrays, comparison) {
            (BoundArrays::Boolean(pair), BoundComparison::Boolean(constant)) => {
                let bounds = pair.on(side);
                let bits =
                    pack_standing_at(bounds.len(), standing, |at| bounds.value(at).cmp(constant));
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Int(pair), BoundComparison::Integer(constant)) => {
                let bounds = pair.on(side);
                let bits = pack_standing(bounds.values(), standing, |bound: i32| {
                    i64::from(bound).cmp(constant)
                });
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Long(pair), BoundComparison::Integer(constant)) => {
                let bounds = pair.on(side);
                let bits =
                    pack_standing(bounds.values(), standing, |bound: i64| bound.cmp(constant));
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Float(pair), BoundComparison::Float(constant)) => {
                let bounds = pair.on(side);
                let bits = pack_standing(bounds.values(), standing, |bound: f32| {
                    bound.total_cmp(constant)
                });
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Float(pair), BoundComparison::FloatAsDouble(constant)) => {
                let bounds = pair.on(side);
                let bits = pack_standing(bounds.values(), standing, |bound: f32| {
                    f64::from(bound).total_cmp(constant)
                });
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Double(pair), BoundComparison::Double(constant)) => {
                let bounds = pair.on(side);
                let bits = pack_standing(bounds.values(), standing, |bound: f64| {
                    bound.total_cmp(constant)
                });
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::Decimal(pair), BoundComparison::Decimal(constant)) => {
                let bounds = pair.on(side);
                let bits =
                    pack_standing(bounds.values(), standing, |bound: i128| bound.cmp(constant));
                keep_valid(bits, bounds.nulls())
            }
            (BoundArrays::String(pair), BoundComparison::String(constant)) => {
                let bounds = pair.on(side);
                let bits =
                    pack_standing_at(bounds.len(), standing, |at| bounds.value(at).cmp(constant));
                keep_valid(bits, bounds.nulls())
            }
            _ => unreachable!("a comparison is planned for the type its bounds are held in"),
        }
    }

    /// A bit for each file: whether its bounds rule out that it holds a
    /// value satisfying `<operator> constant`, the constant being that of
    /// `comparison` ([`may_bounds_satisfy`]).
    fn select_ruled_out(
        &self,
        operator: PredicateOperator,
        comparison: &BoundComparison<'_>,
    ) -> BooleanBuffer {
        let lower = |standing| self.select_standing(BoundSide::Lower, standing, comparison);
        let upper = |standing| self.select_standing(BoundSide::Upper, standing, comparison);
        match operator {
            PredicateOperator::Eq => &lower(Standing::Above) | &upper(Standing::Below),
            // Every value equals the constant only when both bounds do.
            PredicateOperator::NotEq => &lower(Standing::Equal) & &upper(Standing::Equal),
            PredicateOperator::LessThan => lower(Standing::AtLeast),
            PredicateOperator::LessThanOrEq => lower(Standing::Above),
            PredicateOperator::GreaterThan => upper(Standing::AtMost),
            PredicateOperator::GreaterThanOrEq => upper(Standing::Below),
            _ => BooleanBuffer::new_unset(self.count_files()),
        }
    }

    fn count_files(&self) -> usize {
        match &self.arrays {
            BoundArrays::Boolean(pair) => pair.lower.len(),
            BoundArrays::Int(pair) => pair.lower.len(),
            BoundArrays::Long(pair) => pair.lower.len(),
            BoundArrays::Float(pair) => pair.lower.len(),
            BoundArrays::Double(pair) => pair.lower.len(),
            BoundArrays::Decimal(pair) => pair.lower.len(),
            BoundArrays::String(pair) => pair.lower.len(),
            BoundArrays::Uuid(pair) => pair.lower.len(),
            BoundArrays::Bytes(pair) => pair.lower.len(),
        }
    }

    /// The bound on `side` of the file at `position`, as the manifest
    /// recorded it; `None` for a bound it does not record.
    fn read_datum(&self, side: BoundSide, position: usize) -> Option<Datum> {
        fn read<Values: Array, const WIDTH: usize>(
            bounds: &Values,
            position: usize,
            read_bytes: impl Fn(&Values, usize) -> [u8; WIDTH],
        ) -> Option<Vec<u8>> {
            bounds
                .is_valid(position)
                .then(|| read_bytes(bounds, position).to_vec())
        }
        // A bound is serialized as a manifest holds it, and read as the
        // manifest's own was.
        let bytes = match &self.arrays {
            BoundArrays::Boolean(pair) => read(pair.on(side), position, |bounds, at| {
                [u8::from(bounds.value(at))]
            }),
            BoundArrays::Int(pair) => read(pair.on(side), position, |bounds, at| {
                bounds.value(at).to_le_bytes()
            }),
            BoundArrays::Long(pair) => read(pair.on(side), position, |bounds, at| {
                bounds.value(at).to_le_bytes()
            }),
            BoundArrays::Float(pair) => read(pair.on(side), position, |bounds, at| {
                bounds.value(at).to_le_bytes()
            }),
            BoundArrays::Double(pair) => read(pair.on(side), position, |bounds, at| {
                bounds.value(at).to_le_bytes()
            }),
            BoundArrays::Decimal(pair) => read(pair.on(side), position, |bounds, at| {
                bounds.value(at).to_be_bytes()
            }),
            BoundArrays::String(pair) => {
                let bounds = pair.on(side);
                bounds
                    .is_valid(position)
                    .then(|| bounds.value(position).as_bytes().to_vec())
            }
            BoundArrays::Uuid(pair) => {
                let bounds = pair.on(side);
                bounds
                    .is_valid(position)
                    .then(|| bounds.value(position).to_vec())
            }
            BoundArrays::Bytes(pair) => {
                let bounds = pair.on(side);
                bounds
                    .is_valid(position)
                    .then(|| bounds.value(position).to_vec())
            }
        }?;
        Some(
            Datum::try_from_bytes(&bytes, self.bound_type.clone())
                .expect("a bound read from a manifest reads again from its bytes"),
        )
    }
}

impl ColumnBounds {
    /// The bound on `side` of the file at `position`, as the manifest
    /// recorded it, in the type it recorded it in; `None` for a bound it
    /// does not record.
    pub(crate) fn read_datum(&self, side: BoundSide, position: usize) -> Option<Datum> {
        self.bounds.as_ref()?.read_datum(side, position)
    }

    /// The bound on `side` of the file at `position`, as the value the
    /// manifest recorded; `None` for a bound it does not record.
    pub(crate) fn read_literal(
        &self,
        side: BoundSide,
        position: usize,
    ) -> Option<PrimitiveLiteral> {
        fn read<Values: Array>(
            bounds: &Values,
            position: usize,
            read_value: impl Fn(&Values, usize) -> PrimitiveLiteral,
        ) -> Option<PrimitiveLiteral> {
            bounds
                .is_valid(position)
                .then(|| read_value(bounds, position))
        }
        match &self.bounds.as_ref()?.arrays {
            BoundArrays::Boolean(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Boolean(bounds.value(at))
            }),
            BoundArrays::Int(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Int(bounds.value(at))
            }),
            BoundArrays::Long(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Long(bounds.value(at))
            }),
            BoundArrays::Float(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Float(bounds.value(at).into())
            }),
            BoundArrays::Double(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Double(bounds.value(at).into())
            }),
            BoundArrays::Decimal(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Int128(bounds.value(at))
            }),
            BoundArrays::String(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::String(bounds.value(at).to_string())
            }),
            BoundArrays::Uuid(pair) => read(pair.on(side), position, |bounds, at| {
                let bytes = bounds.value(at).try_into();
                PrimitiveLiteral::UInt128(u128::from_be_bytes(
                    bytes.expect("every UUID is sixteen bytes"),
                ))
            }),
            BoundArrays::Bytes(pair) => read(pair.on(side), position, |bounds, at| {
                PrimitiveLiteral::Binary(bounds.value(at).to_vec())
            }),
        }
    }

    /// A bit for each file: whether it may hold a value of the column that
    /// satisfies `<operator> constant`, `column` being how the files of the
    /// manifest record the column. A file whose every value is NULL may
    /// not, and any other may unless its bounds rule the comparison out
    /// ([`may_bounds_satisfy`]); a bound the manifest does not record rules
    /// nothing out.
    pub(crate) fn may_satisfy(
        &self,
        column: &StoredColumn,
        operator: PredicateOperator,
        constant: &Datum,
    ) -> BooleanBuffer {
        if !column.is_written() {
            return BooleanBuffer::new_unset(self.only_nulls.len());
        }
        let Some(bounds) = &self.bounds else {
            return !&self.only_nulls;
        };
        let satisfying = match column.plan_comparison(&bounds.bound_type, constant) {
            BoundComparison::Unknown => !&self.only_nulls,
            BoundComparison::OneByOne => self.select_one_by_one(bounds, column, operator, constant),
            comparison => {
                let ruled_out = bounds.select_ruled_out(operator, &comparison);
                !&(&self.only_nulls | &ruled_out)
            }
        };
        debug_assert_eq!(
            satisfying,
            self.select_one_by_one(bounds, column, operator, constant),
            "a column's bounds select the files the spec crate's comparison of each selects"
        );
        satisfying
    }

    /// [`may_satisfy`](Self::may_satisfy), each bound of `bounds` ordered
    /// against the constant by the spec crate's own comparison.
    fn select_one_by_one(
        &self,
        bounds: &TypedBounds,
        column: &StoredColumn,
        operator: PredicateOperator,
        constant: &Datum,
    ) -> BooleanBuffer {
        let order = |side: BoundSide, position: usize| {
            column.compare_bound(&bounds.read_datum(side, position)?, constant)
        };
        BooleanBuffer::collect_bool(self.only_nulls.len(), |position| {
            !self.only_nulls.value(position)
                && may_bounds_satisfy(
                    operator,
                    || order(BoundSide::Lower, position),
                    || order(BoundSide::Upper, position),
                )
        })
    }
}

/// The data sequence number of the live `entry` of the manifest `manifest`:
/// its own, or the manifest's for an ADDED entry, which leaves it implicit.
fn read_sequence_number(
    entry: &ManifestEntry,
    manifest: &ManifestFile,
    metadata: &TableMetadata,
) -> Result<i64> {
    match entry.sequence_number() {
        Some(number) => Ok(number),
        None if entry.status() == ManifestStatus::Added => Ok(manifest.sequence_number),
        // A version 1 table has no sequence numbers: every entry reads as
        // the spec's inherited 0.
        None if metadata.format_version() == FormatVersion::V1 => Ok(0),
        None => Err(Error::InvalidDataFile {
            file: entry.data_file().file_path().to_string(),
            message: "an existing entry carries no sequence number".to_string(),
        }),
    }
}

/// The live entries of one partition in a parsed data manifest, in the
/// manifest's order.
pub(crate) struct GroupedPartition {
    partition: Arc<Partition>,
    entries: Vec<ManifestEntry>,
}

/// A parsed data manifest with its live entries apart by partition: what
/// every part of the manifest is built from.
pub(crate) struct GroupedManifest {
    /// The schema the manifest was written under.
    schema_id: i32,
    /// The partitions the manifest holds a live file of, by key.
    partitions: BTreeMap<Arc<[u8]>, GroupedPartition>,
}

impl GroupedManifest {
    /// Put the live entries of `manifest`, a data manifest of `table`,
    /// apart by partition. A partition's key is encoded once for each run
    /// of entries of the partition. A delete file among the entries, or a
    /// file that is not Parquet, is refused.
    pub(crate) fn group(table: &str, manifest: Manifest) -> Result<Self> {
        let schema_id = manifest.metadata().schema_id();
        let partition_spec_id = manifest.metadata().partition_spec().spec_id();
        let mut partitions: BTreeMap<Arc<[u8]>, GroupedPartition> = BTreeMap::new();
        // The key of the partition of the entry before: a manifest's
        // entries mostly come in runs of one partition.
        let mut latest: Option<Arc<[u8]>> = None;
        let (entries, _) = manifest.into_parts();
        for entry in entries {
            if !entry.is_alive() {
                continue;
            }
            let data = entry.data_file();
            if data.content_type() != DataContentType::Data {
                return Err(Error::DeleteFile {
                    file: data.file_path().to_string(),
                });
            }
            if data.file_format() != DataFileFormat::Parquet {
                return Err(Error::NonParquetFile {
                    table: table.to_string(),
                    file: data.file_path().to_string(),
                    format: data.file_format().to_string(),
                });
            }
            let is_of_latest = latest
                .as_ref()
                .is_some_and(|key| partitions[key].partition.tuple() == data.partition());
            if !is_of_latest {
                let key = encode_partition(partition_spec_id, data.partition());
                partitions
                    .entry(Arc::clone(&key))
                    .or_insert_with(|| GroupedPartition {
                        partition: Arc::new(Partition::with_key(
                            partition_spec_id,
                            data.partition().clone(),
                            Arc::clone(&key),
                        )),
                        entries: Vec::new(),
                    });
                latest = Some(key);
            }
            let key = latest
                .as_ref()
                .expect("the partition of the entry in hand is the latest");
            // The manifest was parsed for this alone, so its entries are
            // its own to give away.
            partitions
                .get_mut(key)
                .expect("the latest partition was met")
                .entries
                .push(Arc::unwrap_or_clone(entry));
        }
        Ok(Self {
            schema_id,
            partitions,
        })
    }

    /// The partitions the manifest holds a live file of.
    pub(crate) fn list_partitions(&self) -> ManifestPartitions {
        let partitions = self.partitions.values();
        ManifestPartitions::new(
            partitions
                .map(|grouped| Arc::clone(&grouped.partition))
                .collect(),
        )
    }

    /// The live entries of each partition, in the order of the
    /// partitions' keys.
    pub(crate) fn partitions(&self) -> impl Iterator<Item = &GroupedPartition> {
        self.partitions.values()
    }

    /// The live files of `partition`, entries of this manifest, which
    /// `file` names in the manifest list of the table `metadata` describes
    /// at `location`, with everything the manifest leaves implicit filled
    /// in.
    pub(crate) fn build_files(
        &self,
        partition: &GroupedPartition,
        file: &ManifestFile,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<ManifestFiles> {
        let entries = &partition.entries;
        let mut paths = Vec::with_capacity(entries.len());
        let mut sizes = Vec::with_capacity(entries.len());
        let mut record_counts = Vec::with_capacity(entries.len());
        let mut commit_sequences = Vec::with_capacity(entries.len());
        let mut sort_order_ids = Vec::with_capacity(entries.len());
        let mut stats_field_ids = FieldIdSet::default();
        for entry in entries {
            let data = entry.data_file();
            let sequence_number = read_sequence_number(entry, file, metadata)?;
            paths.push(location.file_path_for_uri(data.file_path())?);
            sizes.push(data.file_size_in_bytes());
            record_counts.push(data.record_count());
            sort_order_ids.push(data.sort_order_id());
            // A table that numbers no commit reads every file at 0.
            commit_sequences.push(u64::try_from(sequence_number).unwrap_or(0));
            let recorded = data.lower_bounds().keys();
            let recorded = recorded.chain(data.upper_bounds().keys());
            for field_id in recorded.chain(data.null_value_counts().keys()) {
                stats_field_ids.insert(*field_id);
            }
        }
        let stats_field_ids = stats_field_ids.into_sorted();
        let field_positions = FieldPositions::new(&stats_field_ids);
        let mut null_free = vec![false; entries.len() * stats_field_ids.len()];
        for (position, entry) in entries.iter().enumerate() {
            for (field_id, null_count) in entry.data_file().null_value_counts() {
                if *null_count == 0
                    && let Some(field_position) = field_positions.find(*field_id)
                {
                    null_free[field_position * entries.len() + position] = true;
                }
            }
        }
        let null_free = BooleanBuffer::from(null_free);
        let sort_order_ids = Int32Array::from(sort_order_ids);
        let paths = StringArray::from_iter_values(paths.iter().map(|path| path.as_str()));
        let heap_bytes = measure_shared_allocation(size_of::<ManifestFiles>())
            + measure_array_buffers(&paths)
            + measure_allocation(size_of_val(sizes.as_slice()))
            + measure_allocation(size_of_val(record_counts.as_slice()))
            + measure_allocation(size_of_val(commit_sequences.as_slice()))
            + measure_array_buffers(&sort_order_ids)
            + measure_allocation(size_of_val(stats_field_ids.as_slice()))
            + measure_bits(&null_free);
        Ok(ManifestFiles {
            schema_id: self.schema_id,
            paths,
            sizes,
            record_counts,
            commit_sequences,
            sort_order_ids,
            stats_field_ids,
            null_free,
            heap_bytes: heap_bytes as u64,
        })
    }
}

/// A set of field ids, for a manifest's entries to add the fields they
/// record something of to: the ids a table gives its fields are small, so
/// one is mostly told by a flag standing at it.
#[derive(Default)]
struct FieldIdSet {
    /// Whether each id below [`Self::NEAR_IDS`] is in the set, as far as
    /// the flags reach.
    is_near_id_held: Vec<bool>,
    /// The ids in the set that are negative or past the flags.
    far_ids: BTreeSet<i32>,
}

impl FieldIdSet {
    /// How many of the lowest ids are told by a flag.
    const NEAR_IDS: usize = 1 << 16;

    fn insert(&mut self, field_id: i32) {
        match usize::try_from(field_id) {
            Ok(near_id) if near_id < Self::NEAR_IDS => {
                if near_id >= self.is_near_id_held.len() {
                    self.is_near_id_held.resize(near_id + 1, false);
                }
                self.is_near_id_held[near_id] = true;
            }
            _ => {
                self.far_ids.insert(field_id);
            }
        }
    }

    /// The ids in the set, in order.
    fn into_sorted(self) -> Vec<i32> {
        let near_ids = self
            .is_near_id_held
            .iter()
            .enumerate()
            .filter(|(_, is_held)| **is_held)
            .map(|(near_id, _)| near_id as i32);
        let below = self
            .far_ids
            .iter()
            .copied()
            .filter(|field_id| *field_id < 0);
        let above = self
            .far_ids
            .iter()
            .copied()
            .filter(|field_id| *field_id >= 0);
        below.chain(near_ids).chain(above).collect()
    }
}

/// Where each of some fields stands among them, found by the field's id.
enum FieldPositions {
    /// The position of a field at its id, [`Self::NO_POSITION`] at an id
    /// that is none of the fields'.
    ById(Vec<u32>),
    /// The positions of fields whose ids are negative or too high to stand
    /// in a table.
    Sparse(HashMap<i32, usize>),
}

impl FieldPositions {
    const NO_POSITION: u32 = u32::MAX;

    /// The positions of the fields `field_ids`, each where it stands among
    /// them.
    fn new(field_ids: &[i32]) -> Self {
        let highest_id = field_ids.iter().copied().max().unwrap_or(0);
        let is_every_id_near = field_ids.iter().all(|field_id| *field_id >= 0)
            && (highest_id as usize) < FieldIdSet::NEAR_IDS;
        if !is_every_id_near {
            let positions = field_ids.iter().copied().zip(0..);
            return Self::Sparse(positions.collect());
        }
        let mut positions = vec![Self::NO_POSITION; highest_id as usize + 1];
        for (position, field_id) in field_ids.iter().enumerate() {
            positions[*field_id as usize] = position as u32;
        }
        Self::ById(positions)
    }

    /// Where the field `field_id` stands among the fields; `None` when it
    /// is none of them.
    fn find(&self, field_id: i32) -> Option<usize> {
        match self {
            Self::ById(positions) => {
                let position = *positions.get(usize::try_from(field_id).ok()?)?;
                (position != Self::NO_POSITION).then_some(position as usize)
            }
            Self::Sparse(positions) => positions.get(&field_id).copied(),
        }
    }
}

/// The values `read_value` reads of `bounds`, those a manifest's entries
/// record for one field on one side, entry for entry; or the position of
/// the first entry whose bound is not of `bound_type`.
fn read_bounds<'e, Value>(
    bounds: &[Option<&'e Datum>],
    bound_type: &PrimitiveType,
    read_value: impl Fn(&'e PrimitiveLiteral) -> Option<Value>,
) -> std::result::Result<Vec<Option<Value>>, usize> {
    bounds
        .iter()
        .enumerate()
        .map(|(position, bound)| match bound {
            None => Ok(None),
            Some(bound) if bound.data_type() == bound_type => {
                read_value(bound.literal()).map(Some).ok_or(position)
            }
            Some(_) => Err(position),
        })
        .collect()
}

/// Both bounds a manifest's entries record for one field, entry for entry,
/// as the arrays `build_array` makes of the values `read_value` reads of
/// bounds of `bound_type`; or the position of the first entry with a bound
/// that is not of that type.
fn read_pair<'e, Value, Values>(
    lower: &[Option<&'e Datum>],
    upper: &[Option<&'e Datum>],
    bound_type: &PrimitiveType,
    read_value: impl Fn(&'e PrimitiveLiteral) -> Option<Value>,
    build_array: impl Fn(Vec<Option<Value>>) -> Values,
) -> std::result::Result<BoundPair<Values>, usize> {
    Ok(BoundPair {
        lower: build_array(read_bounds(lower, bound_type, &read_value)?),
        upper: build_array(read_bounds(upper, bound_type, &read_value)?),
    })
}

/// The bounds a manifest's entries record for one field, entry for entry,
/// as arrays of the type of the first of them; `None` when no entry records
/// a bound of the field. For bounds that are not all of one type, the
/// position of the first entry whose bound is of another: a manifest read
/// from a file has none, since it types every bound of a field alike, by
/// its own schema.
fn collect_bounds(
    lower: &[Option<&Datum>],
    upper: &[Option<&Datum>],
) -> std::result::Result<Option<TypedBounds>, usize> {
    let Some(first) = lower.iter().chain(upper).flatten().next() else {
        return Ok(None);
    };
    let bound_type = first.data_type();
    let read_int = |literal: &PrimitiveLiteral| match literal {
        PrimitiveLiteral::Int(value) => Some(*value),
        _ => None,
    };
    let read_long = |literal: &PrimitiveLiteral| match literal {
        PrimitiveLiteral::Long(value) => Some(*value),
        _ => None,
    };
    fn read_bytes(literal: &PrimitiveLiteral) -> Option<&[u8]> {
        match literal {
            PrimitiveLiteral::Binary(value) => Some(value.as_slice()),
            _ => None,
        }
    }
    let arrays = match bound_type {
        PrimitiveType::Boolean => BoundArrays::Boolean(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::Boolean(value) => Some(*value),
                _ => None,
            },
            BooleanArray::from,
        )?),
        PrimitiveType::Int | PrimitiveType::Date => BoundArrays::Int(read_pair(
            lower,
            upper,
            bound_type,
            read_int,
            Int32Array::from,
        )?),
        PrimitiveType::Long
        | PrimitiveType::Time
        | PrimitiveType::Timestamp
        | PrimitiveType::Timestamptz
        | PrimitiveType::TimestampNs
        | PrimitiveType::TimestamptzNs => BoundArrays::Long(read_pair(
            lower,
            upper,
            bound_type,
            read_long,
            Int64Array::from,
        )?),
        PrimitiveType::Float => BoundArrays::Float(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::Float(value) => Some(value.0),
                _ => None,
            },
            Float32Array::from,
        )?),
        PrimitiveType::Double => BoundArrays::Double(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::Double(value) => Some(value.0),
                _ => None,
            },
            Float64Array::from,
        )?),
        PrimitiveType::Decimal { .. } => BoundArrays::Decimal(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::Int128(value) => Some(*value),
                _ => None,
            },
            Decimal128Array::from,
        )?),
        PrimitiveType::String => BoundArrays::String(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::String(value) => Some(value.as_str()),
                _ => None,
            },
            StringArray::from,
        )?),
        PrimitiveType::Uuid => BoundArrays::Uuid(read_pair(
            lower,
            upper,
            bound_type,
            |literal| match literal {
                PrimitiveLiteral::UInt128(value) => Some(value.to_be_bytes()),
                _ => None,
            },
            |values| {
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.into_iter(), 16)
                    .expect("every UUID is sixteen bytes")
            },
        )?),
        PrimitiveType::Fixed(_) | PrimitiveType::Binary => BoundArrays::Bytes(read_pair(
            lower,
            upper,
            bound_type,
            read_bytes,
            BinaryArray::from,
        )?),
    };
    Ok(Some(TypedBounds {
        bound_type: bound_type.clone(),
        arrays,
    }))
}

impl GroupedPartition {
    /// How many columns one pass over the entries builds: what a pass
    /// gathers of the entries before the columns are built grows with both.
    const COLUMNS_PER_PASS: usize = 64;

    /// The partition the entries sit in.
    pub(crate) fn partition(&self) -> &Arc<Partition> {
        &self.partition
    }

    /// What the entries record of each of the fields `field_ids`, entry for
    /// entry, in the order the fields are given in. The entries are those
    /// of a manifest read from a file, which types every bound of a field
    /// alike, by its own schema.
    pub(crate) fn build_columns(&self, field_ids: &[i32]) -> Vec<ColumnBounds> {
        let mut columns = Vec::with_capacity(field_ids.len());
        for field_ids in field_ids.chunks(Self::COLUMNS_PER_PASS) {
            columns.append(&mut self.build_columns_in_one_pass(field_ids));
        }
        columns
    }

    /// [`build_columns`](Self::build_columns), by one pass over the
    /// entries: each entry's bounds and null counts are gone through once
    /// and put where their fields' columns have them.
    fn build_columns_in_one_pass(&self, field_ids: &[i32]) -> Vec<ColumnBounds> {
        let field_positions = FieldPositions::new(field_ids);
        let entry_count = self.entries.len();
        let mut lower: Vec<Vec<Option<&Datum>>> = vec![vec![None; entry_count]; field_ids.len()];
        let mut upper = lower.clone();
        let mut only_nulls = vec![vec![false; entry_count]; field_ids.len()];
        for (position, entry) in self.entries.iter().enumerate() {
            let data = entry.data_file();
            for (field_id, bound) in data.lower_bounds() {
                if let Some(field_position) = field_positions.find(*field_id) {
                    lower[field_position][position] = Some(bound);
                }
            }
            for (field_id, bound) in data.upper_bounds() {
                if let Some(field_position) = field_positions.find(*field_id) {
                    upper[field_position][position] = Some(bound);
                }
            }
            let record_count = data.record_count();
            for (field_id, null_count) in data.null_value_counts() {
                if *null_count == record_count
                    && let Some(field_position) = field_positions.find(*field_id)
                {
                    only_nulls[field_position][position] = true;
                }
            }
        }
        let bounds_per_field = lower.iter().zip(&upper).zip(only_nulls);
        bounds_per_field
            .map(|((lower, upper), only_nulls)| {
                let bounds = collect_bounds(lower, upper)
                    .expect("a manifest read from a file types the bounds of a field alike");
                let only_nulls = BooleanBuffer::from(only_nulls);
                let heap_bytes = measure_shared_allocation(size_of::<ColumnBounds>())
                    + bounds.as_ref().map_or(0, TypedBounds::estimate_heap_bytes)
                    + measure_bits(&only_nulls);
                ColumnBounds {
                    bounds,
                    only_nulls,
                    heap_bytes: heap_bytes as u64,
                }
            })
            .collect()
    }
}
