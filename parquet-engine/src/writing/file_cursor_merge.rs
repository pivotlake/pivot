//! Merges the sorted input files of a compaction by walking each with a cursor.
//!
//! Every input file of a sorted table is itself sorted, so the merge is a
//! k-way merge of the files: whichever file's cursor holds the smallest key
//! gives the next output row. The files' row groups are fetched on demand,
//! each file's current one and the one after it, and a fetched batch is let
//! go of as soon as its file's cursor has passed it, so the decoded input in
//! memory stays near two row groups per file however large the merge is.
//!
//! [`RequestIssuer`] turns the demand into row-group requests, [`Keyer`]
//! encodes each fetched batch's sort columns as comparable rows on whichever
//! worker decoded it, and the single-worker [`FileCursorMerge`] runs the
//! cursors and hands each stretch of the merged order out as a [`GatherJob`]
//! for the streaming row-group planner.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, UInt32Array};
use arrow_row::{RowConverter, Rows, SortField};
use arrow_schema::{Schema, SchemaRef, SortOptions};
use dispatch::{RECORD_BATCH_SIZE, Sender, Unary, UnaryFactory, UnaryResult};

use super::error::WriteError;
use super::streaming::GatherJob;
use crate::reading::record_batch_metadata::{global_row_group, row_index};
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::projection::Projection;
use crate::{ParquetTable, RowGroupRequest};

/// How many row groups of each file are requested ahead of its cursor,
/// counting the one the cursor is in.
const ROW_GROUPS_AHEAD_PER_FILE: usize = 2;

/// Row-group ids of a table by file, in file order: a row group's index within
/// its file restarts at zero at every file.
pub(super) fn files_of(table: &ParquetTable) -> Vec<Vec<u32>> {
    let mut files: Vec<Vec<u32>> = Vec::new();
    for (group, metadata) in table.row_groups().iter().enumerate() {
        if metadata.file_row_group_idx == 0 {
            files.push(Vec::new());
        }
        files
            .last_mut()
            .expect("a file's first row group starts it")
            .push(group as u32);
    }
    files
}

/// Row groups waiting to be requested, shared by the merge that asks for
/// them and the issuer that sends the requests.
pub(super) type Demand = Arc<Mutex<VecDeque<u32>>>;

/// The demand before any cursor moves: the first row groups of every file.
pub(super) fn initial_demand(files: &[Vec<u32>]) -> Demand {
    Arc::new(Mutex::new(
        files
            .iter()
            .flat_map(|groups| groups.iter().take(ROW_GROUPS_AHEAD_PER_FILE).copied())
            .collect(),
    ))
}

pub(super) struct RequestIssuerFactory {
    pub(super) table: Arc<ParquetTable>,
    pub(super) projection: Projection,
    pub(super) demand: Demand,
    /// Every request goes out from one worker; the others have nothing to do.
    pub(super) issuing: bool,
}

pub(super) struct RequestIssuer {
    table: Arc<ParquetTable>,
    projection: Projection,
    demand: Demand,
    /// Row groups not yet requested; zero once nothing is issuing here.
    remaining: usize,
}

impl UnaryFactory<(), RowGroupRequest> for RequestIssuerFactory {
    type Unary = RequestIssuer;

    fn build_unary(self) -> RequestIssuer {
        RequestIssuer {
            remaining: if self.issuing {
                self.table.row_groups().len()
            } else {
                0
            },
            table: self.table,
            projection: self.projection,
            demand: self.demand,
        }
    }
}

impl Unary<(), RowGroupRequest> for RequestIssuer {
    fn consume(
        &mut self,
        _: (),
        _sender: &mut dyn Sender<RowGroupRequest>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        unreachable!("the issuer has no input; it works from the merge's demand")
    }

    /// Called again until every row group has been requested.
    fn finish(&mut self, sender: &mut dyn Sender<RowGroupRequest>) -> UnaryResult<bool> {
        while self.remaining > 0 {
            let Some(group) = self.demand.lock().unwrap().pop_front() else {
                return Ok(false);
            };
            self.remaining -= 1;
            sender.send(RowGroupRequest::from(
                QueryRowGroupMetadata::new(&self.table, group as usize, RowSelection::All),
                &self.projection,
            ))?;
        }
        Ok(true)
    }
}

/// A fetched batch with its sort columns encoded as comparable rows.
pub(super) struct KeyedBatch {
    group: u32,
    first_row: u32,
    batch: RecordBatch,
    keys: Rows,
}

pub(super) struct KeyerFactory {
    pub(super) key_columns: Arc<[usize]>,
}

pub(super) struct Keyer {
    key_columns: Arc<[usize]>,
    converter: Option<RowConverter>,
}

impl UnaryFactory<RecordBatch, KeyedBatch> for KeyerFactory {
    type Unary = Keyer;

    fn build_unary(self) -> Keyer {
        Keyer {
            key_columns: self.key_columns,
            converter: None,
        }
    }
}

impl Unary<RecordBatch, KeyedBatch> for Keyer {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<KeyedBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let columns: Vec<_> = self
            .key_columns
            .iter()
            .map(|&column| batch.column(column).clone())
            .collect();
        let converter = match &self.converter {
            Some(converter) => converter,
            None => {
                // Ascending with nulls first, the order the sort columns are
                // written in.
                let fields = columns
                    .iter()
                    .map(|column| {
                        SortField::new_with_options(
                            column.data_type().clone(),
                            SortOptions {
                                descending: false,
                                nulls_first: true,
                            },
                        )
                    })
                    .collect();
                self.converter
                    .insert(RowConverter::new(fields).map_err(WriteError::from)?)
            }
        };
        let keys = converter
            .convert_columns(&columns)
            .map_err(WriteError::from)?;
        let groups = global_row_group(&batch);
        let group = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .expect("row group ids are unsigned")
            .value(groups.run_ends().get_start_physical_index());
        let first_row = row_index(&batch).value(0);
        sender.send(KeyedBatch {
            group,
            first_row,
            batch,
            keys,
        })?;
        Ok(())
    }
}

/// One input file's place in the merge.
struct FileCursor {
    groups: Vec<u32>,
    /// The row group the cursor is in, as an index into `groups`, and the row
    /// of that group it is at; `group_index == groups.len()` once exhausted.
    group_index: usize,
    row: u32,
    /// Fetched batches of this file not yet passed, by row group and first row.
    resident: HashMap<u32, Vec<Arc<KeyedBatch>>>,
    /// The batch holding the cursor's row, once it is resident.
    current: Option<Arc<KeyedBatch>>,
    /// Where `current` sits in the stretch being built, if it is in it.
    source: Option<(usize, u32)>,
}

impl FileCursor {
    fn is_exhausted(&self) -> bool {
        self.group_index == self.groups.len()
    }

    fn group(&self) -> u32 {
        self.groups[self.group_index]
    }

    /// Look the cursor's batch up among the resident ones.
    fn locate(&mut self) {
        let Some(batches) = self.resident.get(&self.group()) else {
            return;
        };
        let position = batches.partition_point(|batch| batch.first_row <= self.row);
        if position == 0 {
            return;
        }
        let batch = &batches[position - 1];
        if self.row < batch.first_row + batch.batch.num_rows() as u32 {
            self.current = Some(batch.clone());
        }
    }

    fn key(&self) -> arrow_row::Row<'_> {
        let batch = self
            .current
            .as_ref()
            .expect("a placed cursor has its batch");
        batch.keys.row((self.row - batch.first_row) as usize)
    }
}

pub(super) struct FileCursorMergeFactory {
    pub(super) files: Vec<Vec<u32>>,
    pub(super) group_rows: Arc<[u32]>,
    pub(super) demand: Demand,
    pub(super) file_rows: usize,
}

pub(super) struct FileCursorMerge {
    cursors: Vec<FileCursor>,
    /// Rows per row group, by row-group id.
    group_rows: Arc<[u32]>,
    demand: Demand,
    file_rows: usize,
    /// Files whose cursor is placed, ordered by key descending so the least
    /// is at the end; ties go to the lower file.
    ordered: Vec<usize>,
    /// Files whose cursor waits on a batch still to arrive.
    waiting: usize,
    /// The stretch of the merged order being built.
    sequence: usize,
    sources: Vec<Arc<KeyedBatch>>,
    mapping: Vec<(u32, u32)>,
    /// The data columns' schema, without the metadata columns.
    schema: Option<SchemaRef>,
}

impl UnaryFactory<KeyedBatch, GatherJob> for FileCursorMergeFactory {
    type Unary = FileCursorMerge;

    fn build_unary(self) -> FileCursorMerge {
        let cursors: Vec<FileCursor> = self
            .files
            .into_iter()
            .map(|groups| FileCursor {
                groups,
                group_index: 0,
                row: 0,
                resident: HashMap::new(),
                current: None,
                source: None,
            })
            .collect();
        FileCursorMerge {
            waiting: cursors
                .iter()
                .filter(|cursor| !cursor.is_exhausted())
                .count(),
            cursors,
            group_rows: self.group_rows,
            demand: self.demand,
            file_rows: self.file_rows,
            ordered: Vec::new(),
            sequence: 0,
            sources: Vec::new(),
            mapping: Vec::new(),
            schema: None,
        }
    }
}

impl FileCursorMerge {
    /// Put a placed cursor among the ordered ones.
    fn insert_ordered(&mut self, file: usize) {
        let key = self.cursors[file].key();
        let at = self.ordered.partition_point(|&other| {
            let other_key = self.cursors[other].key();
            other_key > key || (other_key == key && other > file)
        });
        self.ordered.insert(at, file);
    }

    fn emit_stretch(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<()> {
        sender.send(GatherJob {
            sequence: self.sequence,
            file_rows: self.file_rows,
            schema: self.schema.clone().expect("a batch set the schema"),
            sources: self
                .sources
                .drain(..)
                .map(|source| source.batch.clone())
                .collect(),
            mapping: std::mem::take(&mut self.mapping),
        })?;
        self.sequence += 1;
        for cursor in &mut self.cursors {
            cursor.source = None;
        }
        Ok(())
    }

    /// Move `file`'s cursor past its row, letting go of a batch or row group
    /// it finished and asking for the row group that keeps the file ahead.
    fn step(&mut self, file: usize) -> Result<(), WriteError> {
        let cursor = &mut self.cursors[file];
        let previous = cursor
            .current
            .take()
            .expect("a placed cursor has its batch");
        let previous_key = previous
            .keys
            .row((cursor.row - previous.first_row) as usize);
        cursor.row += 1;
        if cursor.row < previous.first_row + previous.batch.num_rows() as u32 {
            let key = previous
                .keys
                .row((cursor.row - previous.first_row) as usize);
            if key < previous_key {
                return Err(WriteError::UnsortedInput { file });
            }
            cursor.current = Some(previous);
            return Ok(());
        }
        // The next batch is another source of the stretch.
        cursor.source = None;
        let group = cursor.group();
        let batches = cursor
            .resident
            .get_mut(&group)
            .expect("the batch was resident");
        batches.retain(|batch| !Arc::ptr_eq(batch, &previous));
        if cursor.row == self.group_rows[group as usize] {
            cursor.resident.remove(&group);
            cursor.group_index += 1;
            cursor.row = 0;
            if let Some(&ahead) = cursor
                .groups
                .get(cursor.group_index + ROW_GROUPS_AHEAD_PER_FILE - 1)
            {
                self.demand.lock().unwrap().push_back(ahead);
            }
            if cursor.is_exhausted() {
                return Ok(());
            }
        }
        cursor.locate();
        match &cursor.current {
            Some(current) => {
                let key = current.keys.row((cursor.row - current.first_row) as usize);
                if key < previous_key {
                    return Err(WriteError::UnsortedInput { file });
                }
            }
            None => self.waiting += 1,
        }
        Ok(())
    }

    /// Merge while every live cursor is placed.
    fn advance(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<()> {
        while self.waiting == 0 {
            let Some(file) = self.ordered.pop() else {
                return Ok(());
            };
            let cursor = &mut self.cursors[file];
            let current = cursor
                .current
                .as_ref()
                .expect("an ordered cursor has its batch");
            let source = match cursor.source {
                Some((sequence, source)) if sequence == self.sequence => source,
                _ => {
                    self.sources.push(current.clone());
                    let source = self.sources.len() as u32 - 1;
                    cursor.source = Some((self.sequence, source));
                    source
                }
            };
            self.mapping.push((source, cursor.row - current.first_row));
            self.step(file)?;
            if self.cursors[file].current.is_some() {
                self.insert_ordered(file);
            }
            if self.mapping.len() == RECORD_BATCH_SIZE {
                self.emit_stretch(sender)?;
            }
        }
        Ok(())
    }
}

impl Unary<KeyedBatch, GatherJob> for FileCursorMerge {
    fn consume(
        &mut self,
        batch: KeyedBatch,
        sender: &mut dyn Sender<GatherJob>,
        _io: &mut dispatch::OperatorIO,
    ) -> UnaryResult<()> {
        if self.schema.is_none() {
            let data_columns = batch.batch.num_columns() - 2;
            self.schema = Some(Arc::new(Schema::new(
                batch.batch.schema().fields()[..data_columns].to_vec(),
            )));
        }
        let file = self
            .cursors
            .iter()
            .position(|cursor| cursor.groups.binary_search(&batch.group).is_ok())
            .expect("a fetched row group belongs to an input file");
        let cursor = &mut self.cursors[file];
        let batches = cursor.resident.entry(batch.group).or_default();
        let at = batches.partition_point(|resident| resident.first_row < batch.first_row);
        batches.insert(at, Arc::new(batch));
        if cursor.current.is_none() && !cursor.is_exhausted() {
            cursor.locate();
            if cursor.current.is_some() {
                self.waiting -= 1;
                self.insert_ordered(file);
            }
        }
        self.advance(sender)
    }

    fn finish(&mut self, sender: &mut dyn Sender<GatherJob>) -> UnaryResult<bool> {
        // Only the worker the batches funnel to merges; its peers saw none.
        if self.schema.is_none() {
            return Ok(true);
        }
        assert!(
            self.waiting == 0 && self.ordered.is_empty(),
            "every input row group arrived before the merge finished"
        );
        if !self.mapping.is_empty() {
            self.emit_stretch(sender)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::record_batch_metadata::with_row_group_metadata;
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field};
    use dispatch::test_utils::feed_unary;

    /// One fetched batch of `group` starting at `first_row`, with these keys.
    fn keyed(group: u32, first_row: u32, keys: &[i64]) -> KeyedBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(keys.to_vec()))]).unwrap();
        let batch = with_row_group_metadata(batch, group as usize, first_row as usize);
        let mut keyer = KeyerFactory {
            key_columns: Arc::from([0]),
        }
        .build_unary();
        feed_unary(&mut keyer, vec![batch]).pop().unwrap()
    }

    /// The keys the emitted stretches lay out, in order.
    fn merged_keys(jobs: &[GatherJob]) -> Vec<i64> {
        jobs.iter()
            .flat_map(|job| {
                job.mapping.iter().map(|&(source, row)| {
                    job.sources[source as usize]
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(row as usize)
                })
            })
            .collect()
    }

    #[test]
    fn merges_files_by_key_across_their_batches_and_row_groups() {
        // File 0 holds row groups 0, 1 and 2, file 1 holds row group 3; the
        // batches arrive out of order.
        let files = vec![vec![0, 1, 2], vec![3]];
        let demand = initial_demand(&files);
        let mut merge = FileCursorMergeFactory {
            files,
            group_rows: Arc::from([2, 2, 2, 4]),
            demand: demand.clone(),
            file_rows: 10,
        }
        .build_unary();

        let mut jobs = feed_unary(
            &mut merge,
            vec![
                keyed(3, 2, &[7, 20]),
                keyed(0, 0, &[2, 3]),
                keyed(3, 0, &[1, 5]),
                keyed(1, 0, &[4, 6]),
                keyed(2, 0, &[8, 9]),
            ],
        );
        let mut sender = dispatch::test_utils::CollectSender::new();
        assert!(merge.finish(&mut sender).unwrap());
        jobs.extend(sender.items);

        assert_eq!(merged_keys(&jobs), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 20]);
        assert_eq!(jobs.len(), 1);
        // Row group 2 was asked for as file 0's cursor entered row group 1.
        assert_eq!(
            demand.lock().unwrap().iter().copied().collect::<Vec<_>>(),
            vec![0, 1, 3, 2]
        );
    }

    #[test]
    fn rejects_an_input_file_that_is_not_sorted() {
        let mut merge = FileCursorMergeFactory {
            files: vec![vec![0]],
            group_rows: Arc::from([4]),
            demand: initial_demand(&[vec![0]]),
            file_rows: 4,
        }
        .build_unary();

        let mut sender = dispatch::test_utils::CollectSender::new();
        let result = merge.consume(
            keyed(0, 0, &[1, 3, 2, 4]),
            &mut sender,
            &mut dispatch::TestOperatorIO::default().io(),
        );

        assert!(result.is_err());
    }
}
