//! What a data manifest says of a table's files, cut into the parts a scan
//! asks for: by partition, and then by column.
//!
//! A manifest never changes, so neither does anything read off it, and a
//! table version is nothing but the manifests its list names. Three kinds of
//! part are read off a manifest, each held on its own by the process-wide
//! [`TableMetadataCache`](crate::table_metadata_cache::TableMetadataCache):
//!
//! - [`ManifestPartitions`], one to a manifest: the partitions the manifest
//!   holds a live file of.
//! - [`CachedPartitionFiles`], one to a partition of a manifest: the partition's
//!   live files in the manifest, in the manifest's order, with what a scan
//!   reads of each besides the bounds of its columns.
//! - [`CachedColumnBounds`], one to a column of a partition of a manifest: the
//!   lower and the upper bound the manifest records for the column in each
//!   of those files, as two Arrow arrays that stand file for file beside the
//!   partition's [`CachedPartitionFiles`], and whether the manifest counts every
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
//! two values of the constant's type. Where the bounds and the constant are
//! of types that order as their values do, booleans, integers, dates,
//! timestamps, floats, decimals of one scale and strings, a whole array is
//! compared with the constant by Arrow's comparison kernel for the
//! comparison the filter makes ([`select_with_kernel`]), chosen before the
//! pass over the array, not for each value. The kernels order as the spec
//! crate does: floats by IEEE 754's total order, a decimal by its unscaled
//! value, a string by its bytes. A kernel compares values of one Arrow
//! type, which leaves two pairings. `float` bounds against a `double` are
//! widened to an array of `double`s, which every `float` is exactly one of,
//! and compared as those. `int` bounds against a constant no `int` holds
//! are not looked at: every one of them is on the same side of the
//! constant, so a comparison holds of every recorded bound or of none. A
//! kernel answers null for a missing bound, which is read as the bound
//! passing no test against the constant, so it rules no file out. Bounds of
//! any other pairing rule no file out ([`ComparedColumn::plan_comparison`]):
//! those the spec crate does not compare with any constant, of a `time`, a
//! nanosecond timestamp, a `uuid`, a `fixed` or a `binary` column, and those
//! recorded in a type the column's is not a widening of.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    Array, BinaryArray, BooleanArray, Datum as ArrowDatum, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int32Array, Int64Array, Scalar, StringArray,
};
use arrow_buffer::BooleanBuffer;
use arrow_ord::cmp::{eq, gt, gt_eq, lt, lt_eq, neq};
use iceberg::expr::PredicateOperator;
use iceberg::spec::{
    DataContentType, DataFileFormat, Datum, FormatVersion, Manifest, ManifestEntry, ManifestFile,
    ManifestStatus, PrimitiveLiteral, PrimitiveType, TableMetadata,
};
use parquet_engine::{BoundSide, measure_allocation, measure_shared_allocation};
use planner::expression::CompareType;

use crate::files::partition::{Partition, holds_only_primitive_values};
use crate::files::pruning::{BoundComparison, ComparedColumn};
use crate::metadata::layout::TableLocation;
use crate::{Error, Result};

/// The fixed heap overhead of one Arrow buffer, besides its data.
const BUFFER_OVERHEAD_BYTES: usize = 64;

/// The partitions one data manifest holds a live file of, in their order.
pub(crate) struct ManifestPartitions {
    partitions: Vec<Arc<Partition>>,
    heap_bytes: u64,
}

impl ManifestPartitions {
    pub(crate) fn new(partitions: Vec<Arc<Partition>>) -> Self {
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

    /// Whether the manifest holds a live file of `partition`.
    pub(crate) fn holds(&self, partition: &Partition) -> bool {
        self.partitions
            .binary_search_by(|held| (**held).cmp(partition))
            .is_ok()
    }

    /// The partitions, in their order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Arc<Partition>> {
        self.partitions.iter()
    }

    /// An estimate of the memory the part holds, itself included.
    pub(crate) fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }
}

/// The live files of one partition in one data manifest, in the manifest's
/// order: what a scan reads of each besides the bounds of its columns.
pub(crate) struct CachedPartitionFiles {
    /// The schema the manifest was written under: the fields its files
    /// have, and the types their bounds are recorded in.
    pub(crate) schema_id: i32,
    /// Each file's path as the table names it.
    pub(crate) paths: StringArray,
    pub(crate) sizes: Vec<u64>,
    /// The rows each file holds, deleted ones included.
    pub(crate) record_counts: Vec<u64>,
    /// Where each file's commit stands among the table's commits.
    pub(crate) commit_sequences: Vec<u64>,
    /// The id of the table sort order each file's rows are in, as its
    /// entry records it; null for an entry that says nothing.
    pub(crate) sort_order_ids: Int32Array,
    /// The ids of the fields the manifest records a bound or a null count
    /// of in one of the files, in order: the columns the partition has a
    /// [`CachedColumnBounds`] of in the manifest.
    pub(crate) stats_field_ids: Vec<i32>,
    /// Whether the manifest counts no NULL of a field in a file: a bit for
    /// each file under each field of `stats_field_ids`, field after field,
    /// so the bits of one field stand together.
    pub(crate) null_free: BooleanBuffer,
    pub(crate) heap_bytes: u64,
}

impl CachedPartitionFiles {
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

    /// The ids of the fields the partition has a [`CachedColumnBounds`] of in the
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
        let null_free = self
            .null_free
            .slice(field_position * file_count, file_count);
        let selected = is_selected.bit_chunks().iter_padded();
        selected
            .zip(null_free.bit_chunks().iter_padded())
            .any(|(selected, null_free)| selected & !null_free != 0)
    }

    /// An estimate of the memory the part holds, itself included.
    pub(crate) fn estimate_heap_bytes(&self) -> u64 {
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
pub(crate) struct TypedBounds {
    bound_type: PrimitiveType,
    arrays: BoundArrays,
}

/// What one data manifest records of one column in the files of one
/// partition, file for file beside the partition's [`CachedPartitionFiles`].
pub(crate) struct CachedColumnBounds {
    /// `None` when the manifest records no bound of the column in any of
    /// the files.
    pub(crate) bounds: Option<TypedBounds>,
    /// Whether the manifest counts every value of the column in a file as
    /// NULL: a bit for each file.
    pub(crate) only_nulls: BooleanBuffer,
    pub(crate) heap_bytes: u64,
}

/// An estimate of the heap memory `array`'s buffers take: each at the size an
/// allocation of it takes, plus the fixed overhead of a buffer; the array
/// itself is counted where it stands.
fn estimate_array_heap_bytes(array: &dyn Array) -> usize {
    let data = array.to_data();
    let validity = data.nulls().map(|nulls| nulls.buffer());
    data.buffers()
        .iter()
        .chain(validity)
        .map(|buffer| measure_allocation(buffer.capacity()) + BUFFER_OVERHEAD_BYTES)
        .sum()
}

/// An estimate of the heap memory `bits` take outside themselves.
fn estimate_bitmap_heap_bytes(bits: &BooleanBuffer) -> usize {
    measure_allocation(bits.inner().capacity()) + BUFFER_OVERHEAD_BYTES
}

/// A bit for each of `bounds`: whether it is recorded.
fn select_recorded(bounds: &dyn Array) -> BooleanBuffer {
    match bounds.nulls() {
        Some(validity) => validity.inner().clone(),
        None => BooleanBuffer::new_set(bounds.len()),
    }
}

/// A bit for each of `bounds`: whether it is recorded and passes `test`
/// against `constant`, a scalar of the bounds' Arrow type, by the Arrow
/// kernel for that test.
fn select_with_kernel(
    bounds: &dyn ArrowDatum,
    test: CompareType,
    constant: &dyn ArrowDatum,
) -> BooleanBuffer {
    let compare = match test {
        CompareType::Less => lt,
        CompareType::LessEqual => lt_eq,
        CompareType::Equal => eq,
        CompareType::NotEqual => neq,
        CompareType::GreaterEqual => gt_eq,
        CompareType::Greater => gt,
    };
    let (bits, validity) = compare(bounds, constant)
        .expect("bounds compare with a constant of their own Arrow type")
        .into_parts();
    // A missing bound compares as a null, and passes no test against a
    // constant.
    match validity {
        Some(validity) => &bits & validity.inner(),
        None => bits,
    }
}

impl TypedBounds {
    /// The arrays of the lower and of the upper bounds.
    fn list_arrays(&self) -> [&dyn Array; 2] {
        match &self.arrays {
            BoundArrays::Boolean(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Int(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Long(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Float(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Double(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Decimal(pair) => [&pair.lower, &pair.upper],
            BoundArrays::String(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Uuid(pair) => [&pair.lower, &pair.upper],
            BoundArrays::Bytes(pair) => [&pair.lower, &pair.upper],
        }
    }

    fn estimate_heap_bytes(&self) -> usize {
        let arrays = self.list_arrays();
        arrays.into_iter().map(estimate_array_heap_bytes).sum()
    }

    /// A bit for each file: whether its bound on `side` is recorded and
    /// passes `test` against the constant of `comparison`, which orders
    /// bounds of these arrays' type as their values order.
    fn select_passing(
        &self,
        side: BoundSide,
        test: CompareType,
        comparison: &BoundComparison<'_>,
    ) -> BooleanBuffer {
        let select = |bounds: &dyn ArrowDatum, constant: &dyn ArrowDatum| {
            select_with_kernel(bounds, test, constant)
        };
        match (&self.arrays, comparison) {
            (BoundArrays::Boolean(pair), BoundComparison::Boolean(constant)) => {
                select(pair.on(side), &BooleanArray::new_scalar(*constant))
            }
            (BoundArrays::Int(pair), BoundComparison::Integer(constant)) => {
                let bounds = pair.on(side);
                match i32::try_from(*constant) {
                    Ok(constant) => select(bounds, &Int32Array::new_scalar(constant)),
                    // A constant no `int` holds: every bound is less than
                    // one past the largest `int`, and greater than one
                    // before the smallest.
                    Err(_) => {
                        let is_every_bound_less = *constant > i64::from(i32::MAX);
                        let passes = match test {
                            CompareType::Less | CompareType::LessEqual => is_every_bound_less,
                            CompareType::Equal => false,
                            CompareType::NotEqual => true,
                            CompareType::GreaterEqual | CompareType::Greater => {
                                !is_every_bound_less
                            }
                        };
                        if passes {
                            select_recorded(bounds)
                        } else {
                            BooleanBuffer::new_unset(bounds.len())
                        }
                    }
                }
            }
            (BoundArrays::Long(pair), BoundComparison::Integer(constant)) => {
                select(pair.on(side), &Int64Array::new_scalar(*constant))
            }
            (BoundArrays::Float(pair), BoundComparison::Float(constant)) => {
                select(pair.on(side), &Float32Array::new_scalar(*constant))
            }
            // Not every `double` is a `float`'s value, and every `float`
            // is a `double`'s: the bounds are widened to compare with the
            // constant.
            (BoundArrays::Float(pair), BoundComparison::FloatAsDouble(constant)) => {
                let widened: Float64Array = pair.on(side).unary(f64::from);
                select(&widened, &Float64Array::new_scalar(*constant))
            }
            (BoundArrays::Double(pair), BoundComparison::Double(constant)) => {
                select(pair.on(side), &Float64Array::new_scalar(*constant))
            }
            (BoundArrays::Decimal(pair), BoundComparison::Decimal(constant)) => {
                let bounds = pair.on(side);
                // The constant takes the arrays' own precision and scale,
                // which say nothing, so the unscaled values are compared.
                let constant = Decimal128Array::from_value(*constant, 1)
                    .with_data_type(bounds.data_type().clone());
                select(bounds, &Scalar::new(constant))
            }
            (BoundArrays::String(pair), BoundComparison::String(constant)) => {
                select(pair.on(side), &StringArray::new_scalar(constant))
            }
            _ => unreachable!("a comparison is planned for the type its bounds are held in"),
        }
    }

    /// A bit for each file: whether its bounds rule out that it holds a
    /// value satisfying `<operator> constant`, the constant being that of
    /// `comparison`: the rule of
    /// [`may_bounds_match_comparison`](crate::files::pruning::may_bounds_match_comparison),
    /// for every file at once.
    fn select_ruled_out(
        &self,
        operator: PredicateOperator,
        comparison: &BoundComparison<'_>,
    ) -> BooleanBuffer {
        let lower = |test| self.select_passing(BoundSide::Lower, test, comparison);
        let upper = |test| self.select_passing(BoundSide::Upper, test, comparison);
        match operator {
            PredicateOperator::Eq => &lower(CompareType::Greater) | &upper(CompareType::Less),
            // Every value equals the constant only when both bounds do.
            PredicateOperator::NotEq => &lower(CompareType::Equal) & &upper(CompareType::Equal),
            PredicateOperator::LessThan => lower(CompareType::GreaterEqual),
            PredicateOperator::LessThanOrEq => lower(CompareType::Greater),
            PredicateOperator::GreaterThan => upper(CompareType::LessEqual),
            PredicateOperator::GreaterThanOrEq => upper(CompareType::Less),
            _ => BooleanBuffer::new_unset(self.count_files()),
        }
    }

    fn count_files(&self) -> usize {
        self.list_arrays()[0].len()
    }
}

impl CachedColumnBounds {
    /// An estimate of the memory the part holds, itself included.
    pub(crate) fn estimate_heap_bytes(&self) -> u64 {
        self.heap_bytes
    }

    /// The type the manifest records the column's bounds in; `None` when
    /// it records no bound of the column in any of the files.
    pub(crate) fn read_bound_type(&self) -> Option<&PrimitiveType> {
        Some(&self.bounds.as_ref()?.bound_type)
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
    /// not, and any other may unless its bounds rule the comparison out. A
    /// bound the manifest does not record rules nothing out, and neither do
    /// bounds that do not order against the constant
    /// ([`BoundComparison::Unknown`]).
    pub(crate) fn may_match_comparison(
        &self,
        column: &ComparedColumn,
        operator: PredicateOperator,
        constant: &Datum,
    ) -> BooleanBuffer {
        if !column.is_written() {
            return BooleanBuffer::new_unset(self.only_nulls.len());
        }
        let Some(bounds) = &self.bounds else {
            return !&self.only_nulls;
        };
        match column.plan_comparison(&bounds.bound_type, constant) {
            BoundComparison::Unknown => !&self.only_nulls,
            comparison => {
                let ruled_out = bounds.select_ruled_out(operator, &comparison);
                !&(&self.only_nulls | &ruled_out)
            }
        }
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
pub(crate) struct PartitionEntries {
    /// The schema the manifest was written under.
    schema_id: i32,
    partition: Arc<Partition>,
    entries: Vec<ManifestEntry>,
}

/// A parsed data manifest with its live entries apart by partition: what
/// every part of the manifest is built from.
pub(crate) struct GroupedManifest {
    /// The partitions the manifest holds a live file of, in their order.
    partitions: BTreeMap<Arc<Partition>, PartitionEntries>,
}

impl GroupedManifest {
    /// Put the live entries of `manifest`, a data manifest of `table`,
    /// apart by partition. A partition is looked up once for each run of
    /// entries of the partition. A delete file among the entries, a
    /// file that is not Parquet, or one whose partition tuple holds a value
    /// that is not a primitive, is refused.
    pub(crate) fn group(table: &str, manifest: Manifest) -> Result<Self> {
        let schema_id = manifest.metadata().schema_id();
        let partition_spec_id = manifest.metadata().partition_spec().spec_id();
        let mut partitions: BTreeMap<Arc<Partition>, PartitionEntries> = BTreeMap::new();
        // The partition of the entry before: a manifest's entries mostly
        // come in runs of one partition.
        let mut latest: Option<&mut PartitionEntries> = None;
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
            let partition = match latest {
                Some(partition) if partition.partition.tuple() == data.partition() => partition,
                _ => {
                    if !holds_only_primitive_values(data.partition()) {
                        return Err(Error::NonPrimitivePartitionValue {
                            table: table.to_string(),
                            file: data.file_path().to_string(),
                        });
                    }
                    let partition =
                        Arc::new(Partition::new(partition_spec_id, data.partition().clone()));
                    partitions
                        .entry(Arc::clone(&partition))
                        .or_insert_with(|| PartitionEntries {
                            schema_id,
                            partition,
                            entries: Vec::new(),
                        })
                }
            };
            // The manifest was parsed for this alone, so its entries are
            // its own to give away.
            partition.entries.push(Arc::unwrap_or_clone(entry));
            latest = Some(partition);
        }
        Ok(Self { partitions })
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
    /// partitions.
    pub(crate) fn partitions(&self) -> impl Iterator<Item = &PartitionEntries> {
        self.partitions.values()
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

impl PartitionEntries {
    /// How many columns one pass over the entries builds: what a pass
    /// gathers of the entries before the columns are built grows with both.
    const COLUMNS_PER_PASS: usize = 64;

    /// The partition the entries sit in.
    pub(crate) fn partition(&self) -> &Arc<Partition> {
        &self.partition
    }

    /// The live files of the partition, entries of a manifest which
    /// `file` names in the manifest list of the table `metadata` describes
    /// at `location`, with everything the manifest leaves implicit filled
    /// in.
    pub(crate) fn build_files(
        &self,
        file: &ManifestFile,
        metadata: &TableMetadata,
        location: &TableLocation,
    ) -> Result<CachedPartitionFiles> {
        let entries = &self.entries;
        let mut paths = Vec::with_capacity(entries.len());
        let mut sizes = Vec::with_capacity(entries.len());
        let mut record_counts = Vec::with_capacity(entries.len());
        let mut commit_sequences = Vec::with_capacity(entries.len());
        let mut sort_order_ids = Vec::with_capacity(entries.len());
        let mut stats_field_ids = Vec::new();
        for entry in entries {
            let data = entry.data_file();
            let sequence_number = read_sequence_number(entry, file, metadata)?;
            paths.push(location.find_path_for_uri(data.file_path())?);
            sizes.push(data.file_size_in_bytes());
            record_counts.push(data.record_count());
            sort_order_ids.push(data.sort_order_id());
            // A table that numbers no commit reads every file at 0.
            commit_sequences.push(u64::try_from(sequence_number).unwrap_or(0));
            let recorded = data.lower_bounds().keys();
            let recorded = recorded.chain(data.upper_bounds().keys());
            for field_id in recorded.chain(data.null_value_counts().keys()) {
                stats_field_ids.push(*field_id);
            }
        }
        // Every entry added the fields it records something of: each field
        // is kept once, in the order of the ids, in no more room than the
        // fields take.
        stats_field_ids.sort_unstable();
        stats_field_ids.dedup();
        stats_field_ids.shrink_to_fit();
        let mut null_free = vec![false; entries.len() * stats_field_ids.len()];
        for (position, entry) in entries.iter().enumerate() {
            for (field_id, null_count) in entry.data_file().null_value_counts() {
                if *null_count == 0
                    && let Ok(field_position) = stats_field_ids.binary_search(field_id)
                {
                    null_free[field_position * entries.len() + position] = true;
                }
            }
        }
        let null_free = BooleanBuffer::from(null_free);
        let sort_order_ids = Int32Array::from(sort_order_ids);
        let paths = StringArray::from_iter_values(paths.iter().map(|path| path.as_str()));
        let heap_bytes = measure_shared_allocation(size_of::<CachedPartitionFiles>())
            + estimate_array_heap_bytes(&paths)
            + measure_allocation(size_of_val(sizes.as_slice()))
            + measure_allocation(size_of_val(record_counts.as_slice()))
            + measure_allocation(size_of_val(commit_sequences.as_slice()))
            + estimate_array_heap_bytes(&sort_order_ids)
            + measure_allocation(size_of_val(stats_field_ids.as_slice()))
            + estimate_bitmap_heap_bytes(&null_free);
        Ok(CachedPartitionFiles {
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

    /// What the entries record of each of the fields `field_ids`, entry for
    /// entry, in the order the fields are given in. The entries are those
    /// of a manifest read from a file, which types every bound of a field
    /// alike, by its own schema.
    pub(crate) fn build_columns(&self, field_ids: &[i32]) -> Vec<CachedColumnBounds> {
        let mut columns = Vec::with_capacity(field_ids.len());
        for field_ids in field_ids.chunks(Self::COLUMNS_PER_PASS) {
            columns.append(&mut self.build_columns_in_one_pass(field_ids));
        }
        columns
    }

    /// [`build_columns`](Self::build_columns), by one pass over the
    /// entries: each entry's bounds and null counts are gone through once
    /// and put where their fields' columns have them.
    fn build_columns_in_one_pass(&self, field_ids: &[i32]) -> Vec<CachedColumnBounds> {
        // Where each field stands among `field_ids`, in the order of the
        // ids, for a field to be found by its id.
        let mut positions_by_id: Vec<(i32, usize)> = field_ids.iter().copied().zip(0..).collect();
        positions_by_id.sort_unstable();
        let find_position = |field_id: &i32| {
            let found = positions_by_id.binary_search_by_key(field_id, |(id, _)| *id);
            found.ok().map(|found| positions_by_id[found].1)
        };
        let entry_count = self.entries.len();
        let mut lower: Vec<Vec<Option<&Datum>>> = vec![vec![None; entry_count]; field_ids.len()];
        let mut upper = lower.clone();
        let mut only_nulls = vec![vec![false; entry_count]; field_ids.len()];
        for (position, entry) in self.entries.iter().enumerate() {
            let data = entry.data_file();
            for (field_id, bound) in data.lower_bounds() {
                if let Some(field_position) = find_position(field_id) {
                    lower[field_position][position] = Some(bound);
                }
            }
            for (field_id, bound) in data.upper_bounds() {
                if let Some(field_position) = find_position(field_id) {
                    upper[field_position][position] = Some(bound);
                }
            }
            let record_count = data.record_count();
            for (field_id, null_count) in data.null_value_counts() {
                if *null_count == record_count
                    && let Some(field_position) = find_position(field_id)
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
                let heap_bytes = measure_shared_allocation(size_of::<CachedColumnBounds>())
                    + bounds.as_ref().map_or(0, TypedBounds::estimate_heap_bytes)
                    + estimate_bitmap_heap_bytes(&only_nulls);
                CachedColumnBounds {
                    bounds,
                    only_nulls,
                    heap_bytes: heap_bytes as u64,
                }
            })
            .collect()
    }
}
