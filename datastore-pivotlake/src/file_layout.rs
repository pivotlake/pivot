//! Where a file sits in its table's layout, in the form maintenance walks
//! it: the partition it belongs to and the stretch of the sort key it covers,
//! each as bytes, so files are grouped and their ranges compared without a
//! typed value being decoded. A partition's bytes are equal for exactly the
//! files of that partition; a sort key's bytes order as the keys do.
//!
//! A format that keeps such bytes with its files hands them out as they are.
//! One that keeps its files as plain [`FileEntry`]s derives them here, once
//! per version, in an [`EntryLayout`].

use std::collections::HashMap;
use std::ops::ControlFlow;

use arrow_array::{Array, ArrayRef, new_null_array};
use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;
use object_storage::{FileRef, ObjectPath};

use crate::Result;
use crate::file_entry::{FileEntry, FileStats, PartitionValues};

/// One live data file as a walk over a table hands it out.
pub(crate) struct WalkedFile<'a> {
    pub file: &'a FileRef,
    /// How many rows the file holds, when its metadata recorded it.
    pub rows: Option<u64>,
    /// The file's partition: equal bytes for exactly the files of one
    /// partition.
    pub partition: &'a [u8],
    /// The lowest and the highest sort key in the file, as bytes that order
    /// as the keys do. `None` for a table without a sort key, and for a file
    /// whose metadata does not bound the column the key leads with.
    pub sort_range: Option<(&'a [u8], &'a [u8])>,
}

/// The lowest and the highest sort key of a file, as bytes that order as
/// the keys do.
pub(crate) type SortRange = (Vec<u8>, Vec<u8>);

/// What a walk does with each file: `Break` ends the walk early.
pub(crate) type VisitFile<'v> = dyn FnMut(WalkedFile<'_>) -> ControlFlow<()> + 'v;

/// Encodes the stretch of a table's sort key a file covers, from the file's
/// statistics, as bytes that order as the keys do.
pub(crate) struct SortRangeEncoder {
    sort_by: Vec<String>,
    /// The Arrow type each sort column's bounds are encoded in, with the
    /// converter over those types; `None` for a table without a sort key, or
    /// one sorted by a column whose type has no ordered byte form.
    converter: Option<(Vec<DataType>, RowConverter)>,
}

impl SortRangeEncoder {
    /// The encoder for a table sorted by `sort_by`, columns whose bounds are
    /// of `types` (in `sort_by` order).
    pub(crate) fn new(sort_by: &[String], types: Vec<DataType>) -> Result<Self> {
        let fields: Vec<SortField> = types.iter().cloned().map(SortField::new).collect();
        let converter = if fields.is_empty() || !RowConverter::supports_fields(&fields) {
            None
        } else {
            Some((types, RowConverter::new(fields)?))
        };
        Ok(Self {
            sort_by: sort_by.to_vec(),
            converter,
        })
    }

    /// The lowest and the highest sort key `stats` bound a file's rows by.
    /// `None` when the key has no ordered byte form or `stats` do not bound
    /// the column it leads with; a later column left unbounded orders first.
    pub(crate) fn encode(&self, stats: &FileStats) -> Result<Option<SortRange>> {
        let Some((types, converter)) = &self.converter else {
            return Ok(None);
        };
        let leading = &self.sort_by[0];
        let is_known = |bounds: &HashMap<String, ArrayRef>| {
            bounds
                .get(leading)
                .is_some_and(|bound| bound.len() == 1 && !bound.is_null(0))
        };
        if !is_known(&stats.min_values) || !is_known(&stats.max_values) {
            return Ok(None);
        }
        let encode_bounds = |bounds: &HashMap<String, ArrayRef>| -> Result<Vec<u8>> {
            let arrays: Vec<ArrayRef> = self
                .sort_by
                .iter()
                .zip(types)
                .map(|(column, data_type)| match bounds.get(column) {
                    Some(bound) if bound.len() == 1 && bound.data_type() == data_type => {
                        Ok(ArrayRef::clone(bound))
                    }
                    Some(bound) if bound.len() == 1 => {
                        Ok(arrow_cast::cast(bound.as_ref(), data_type)?)
                    }
                    _ => Ok(new_null_array(data_type, 1)),
                })
                .collect::<Result<_>>()?;
            Ok(converter.convert_columns(&arrays)?.row(0).as_ref().to_vec())
        };
        Ok(Some((
            encode_bounds(&stats.min_values)?,
            encode_bounds(&stats.max_values)?,
        )))
    }
}

/// Encodes a partition tuple as bytes equal for exactly equal tuples: each
/// partition column's value under its type, a column the tuple lacks and a
/// file without a tuple each told apart from any value.
#[derive(Default)]
struct PartitionEncoder {
    converters: HashMap<DataType, RowConverter>,
}

impl PartitionEncoder {
    fn encode(
        &mut self,
        partition_by: &[String],
        partition: Option<&PartitionValues>,
    ) -> Result<Vec<u8>> {
        let Some(partition) = partition else {
            return Ok(vec![0]);
        };
        let mut bytes = vec![1];
        for column in partition_by {
            let Some(value) = partition.get(column) else {
                bytes.push(0);
                continue;
            };
            let (array, _) = arrow_array::Datum::get(value);
            let data_type = array.data_type();
            let type_name = data_type.to_string();
            let converter = match self.converters.get(data_type) {
                Some(converter) => converter,
                None => {
                    let converter = RowConverter::new(vec![SortField::new(data_type.clone())])?;
                    self.converters
                        .entry(data_type.clone())
                        .or_insert(converter)
                }
            };
            let rows = converter.convert_columns(&[array.slice(0, 1)])?;
            let row = rows.row(0);
            bytes.push(1);
            for part in [type_name.as_bytes(), row.as_ref()] {
                bytes.extend_from_slice(&(part.len() as u32).to_le_bytes());
                bytes.extend_from_slice(part);
            }
        }
        Ok(bytes)
    }
}

/// The layout of a version kept as a plain list of entries: the entries
/// grouped by partition, partitions in the order the list first names them
/// and a partition's files in the list's order, each with its sort range,
/// and where each path sits in the list.
pub(crate) struct EntryLayout {
    partitions: Vec<PartitionFiles>,
    position_by_path: HashMap<ObjectPath, usize>,
}

struct PartitionFiles {
    partition: Vec<u8>,
    /// Each file's position in the entry list, with its sort range.
    files: Vec<(usize, Option<SortRange>)>,
}

impl EntryLayout {
    /// The layout of `entries`, files of a table partitioned by
    /// `partition_by` and sorted by the key `sort_ranges` encodes.
    pub(crate) fn of(
        entries: &[FileEntry],
        partition_by: &[String],
        sort_ranges: &SortRangeEncoder,
    ) -> Result<Self> {
        let mut partition_encoder = PartitionEncoder::default();
        let mut partitions: Vec<PartitionFiles> = Vec::new();
        let mut partition_positions: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut position_by_path = HashMap::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            position_by_path.insert(entry.file.path.clone(), position);
            let partition = partition_encoder.encode(partition_by, entry.partition.as_ref())?;
            let sort_range = match &entry.stats {
                Some(stats) => sort_ranges.encode(stats)?,
                None => None,
            };
            let partition_position = match partition_positions.get(&partition) {
                Some(partition_position) => *partition_position,
                None => {
                    partition_positions.insert(partition.clone(), partitions.len());
                    partitions.push(PartitionFiles {
                        partition,
                        files: Vec::new(),
                    });
                    partitions.len() - 1
                }
            };
            partitions[partition_position]
                .files
                .push((position, sort_range));
        }
        Ok(Self {
            partitions,
            position_by_path,
        })
    }

    /// Where the file at `path` sits in the entry list.
    pub(crate) fn find_position(&self, path: &ObjectPath) -> Option<usize> {
        self.position_by_path.get(path).copied()
    }

    /// Visit every entry of `entries`, the list this layout was built from,
    /// partition by partition.
    pub(crate) fn walk(&self, entries: &[FileEntry], visit: &mut VisitFile<'_>) {
        for partition in &self.partitions {
            for (position, sort_range) in &partition.files {
                let entry = &entries[*position];
                let walked = WalkedFile {
                    file: &entry.file,
                    rows: entry
                        .stats
                        .as_ref()
                        .and_then(|stats| stats.num_records)
                        .and_then(|rows| u64::try_from(rows).ok()),
                    partition: &partition.partition,
                    sort_range: sort_range
                        .as_ref()
                        .map(|(lowest, highest)| (lowest.as_slice(), highest.as_slice())),
                };
                if visit(walked).is_break() {
                    return;
                }
            }
        }
    }
}
