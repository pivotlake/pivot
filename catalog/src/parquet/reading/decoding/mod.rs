//! Decodes decompressed Parquet pages into Arrow [`RecordBatch`]es.
//!
//! The Decoder is the final stage of the Parquet pipeline, sitting after the Decompressor.
//! It receives individual [`DecompressedPage`]s (data and dictionary) and groups them by
//! row group. Each row group is managed by a [`RowGroupDecoder`] that accumulates pages
//! across all projected columns until a full batch of rows is available.
//!
//! ## Batch production
//!
//! On every incoming page the Decoder first tries to produce a batch from the row group
//! that just received the page (exploiting cache locality), then falls back to any other
//! row group that has enough data. Exhausted row groups are removed immediately.

use crate::parquet::DecompressedPage;
use crate::parquet::types::metadata::QueryRowGroupMetadata;
use crate::parquet::types::projection::Projection;
use ahash::HashSet;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use dispatch::Sender;
use dispatch::WorkStatus;
use dispatch::memory::SlabAllocator;
use dispatch::{Unary, UnaryFactory};
use std::sync::Arc;

mod column_decoders;

mod row_group_decoder;
pub use row_group_decoder::{Error as RowGroupDecoderError, RowGroupDecoder};

/// A pushed-down equality predicate (`column == value`) used for dictionary
/// pruning at scan time. `column_idx` indexes the table's full schema. When a
/// row group's dictionary for this column excludes `value` (and the column
/// chunk is fully dictionary encoded), the row group can emit no rows and is
/// dropped without decoding its data pages.
#[derive(Clone, Debug)]
pub struct ScanEqualityPredicate {
    pub column_idx: usize,
    pub value: Scalar<ArrayRef>,
}

/// Factory for creating [`Decoder`] instances, one per worker thread.
pub struct DecoderFactory {
    /// Maximum number of rows per output [`RecordBatch`].
    pub batch_size: usize,
    /// Which columns to decode.
    pub projection: Projection,
    /// Whether to append row-group-id and row-index metadata columns to each
    /// output batch (used by the materializer path).
    pub add_row_group_metadata: bool,
    /// Pushed-down equality predicates for dictionary pruning (may be empty).
    pub eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
}

impl UnaryFactory<DecompressedPage, RecordBatch> for DecoderFactory {
    type Unary = Decoder;

    fn build_unary(self) -> Self::Unary {
        Decoder::new(
            self.batch_size,
            self.projection,
            self.add_row_group_metadata,
            self.eq_predicates,
        )
    }
}

/// Accumulates [`DecompressedPage`]s and emits Arrow [`RecordBatch`]es.
///
/// Maintains one [`RowGroupDecoder`] per in-flight row group. Pages arriving for a
/// row group that has already been fully emitted are silently dropped.
///
/// There is no table-wide output schema here: each [`RowGroupDecoder`] derives
/// its own from its file, because a variant column's physical layout (its
/// shredded leaves) can differ file to file.
pub struct Decoder {
    batch_size: usize,
    projection: Projection,
    /// Slab allocator for decoded Arrow buffers.
    allocator: SlabAllocator,
    /// One decoder per in-flight row group.
    row_group_decoders: Vec<RowGroupDecoder>,
    /// Row groups that have been fully emitted — late-arriving pages for these
    /// are silently dropped.
    closed_row_groups: HashSet<usize>,
    add_row_group_metadata: bool,
    /// Pushed-down equality predicates for dictionary pruning (may be empty).
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
}

impl Decoder {
    pub fn new(
        batch_size: usize,
        projection: Projection,
        add_row_group_metadata: bool,
        eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    ) -> Self {
        Self {
            batch_size,
            projection,
            allocator: SlabAllocator::new(true),
            row_group_decoders: Vec::new(),
            closed_row_groups: Default::default(),
            add_row_group_metadata,
            eq_predicates,
        }
    }

    /// Returns the index into `row_group_decoders` for the given row group,
    /// creating a new [`RowGroupDecoder`] if one doesn't already exist.
    fn get_or_create_row_group_decoder_idx(
        &mut self,
        row_group_metadata: &QueryRowGroupMetadata,
    ) -> dispatch::UnaryResult<usize> {
        if let Some(pos) = self
            .row_group_decoders
            .iter()
            .position(|r| r.row_group_idx() == row_group_metadata.index())
        {
            return Ok(pos);
        }
        self.row_group_decoders.push(
            RowGroupDecoder::new(
                row_group_metadata.clone(),
                &self.projection,
                self.batch_size,
                self.add_row_group_metadata,
                &self.eq_predicates,
            )
            .map_err(crate::parquet::op_err)?,
        );
        Ok(self.row_group_decoders.len() - 1)
    }

    /// Try to send out a single record batch if any row group decoder has available.
    /// This will also, as a side effect, remove any row group decoders that are exhausted.
    ///
    /// # Returns
    ///
    /// Whether a record batch has been sent out
    fn try_produce_batch<S: Sender<RecordBatch>>(
        &mut self,
        sender: &mut S,
    ) -> dispatch::UnaryResult<bool> {
        let mut indexes_to_remove = HashSet::default();
        let mut produced = false;

        let allocator = &mut self.allocator;
        for decoder in &mut self.row_group_decoders {
            if let Some(batch) = decoder
                .try_read(allocator)
                .map_err(crate::parquet::op_err)?
            {
                sender.send(batch)?;
                if decoder.exhausted() {
                    indexes_to_remove.insert(decoder.row_group_idx());
                }
                produced = true;
                break;
            }
        }

        if !indexes_to_remove.is_empty() {
            self.row_group_decoders
                .retain(|d| !indexes_to_remove.contains(&d.row_group_idx()));
        }
        Ok(produced)
    }
}

impl Unary<DecompressedPage, RecordBatch> for Decoder {
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        page: DecompressedPage,
        output: &mut S,
    ) -> dispatch::UnaryResult<()> {
        if self
            .closed_row_groups
            .contains(&page.query_row_group_metadata.index())
        {
            return Ok(());
        }

        let pos = self.get_or_create_row_group_decoder_idx(&page.query_row_group_metadata)?;
        self.row_group_decoders[pos].insert_page(page, &mut self.allocator);

        // A just-inserted dictionary page may have pruned the row group (its
        // dictionary excludes a pushed-down equality constant). Drop it without
        // emitting, and ignore its remaining in-flight data pages.
        if self.row_group_decoders[pos].pruned() {
            let row_group_idx = self.row_group_decoders[pos].row_group_idx();
            self.closed_row_groups.insert(row_group_idx);
            self.row_group_decoders.remove(pos);
            return Ok(());
        }

        // Try sending from the decoder that just got a page; this page might be hot!
        if let Some(b) = self.row_group_decoders[pos]
            .try_read(&mut self.allocator)
            .map_err(crate::parquet::op_err)?
        {
            if self.row_group_decoders[pos].exhausted() {
                let row_group_idx = self.row_group_decoders[pos].row_group_idx();
                self.closed_row_groups.insert(row_group_idx);
                self.row_group_decoders.remove(pos);
            }
            output.send(b)?;
            return Ok(());
        }

        self.try_produce_batch(output)?;
        Ok(())
    }

    fn run<S: Sender<RecordBatch>>(&mut self, sender: &mut S) -> dispatch::UnaryResult<WorkStatus> {
        if self.try_produce_batch(sender)? {
            Ok(WorkStatus::Ran)
        } else {
            Ok(WorkStatus::Pending)
        }
    }

    fn finish<S: Sender<RecordBatch>>(&mut self, _output: &mut S) -> dispatch::UnaryResult<bool> {
        Ok(self.row_group_decoders.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use crate::parquet::reading::decoding::{Decoder, ScanEqualityPredicate};
    use crate::parquet::types::metadata::{
        ColumnChunkMeta, QueryRowGroupMetadata, RowGroupMetadata,
    };
    use crate::parquet::types::page::{DataPage, DecompressedPage, DecompressedPageType};
    use crate::parquet::types::projection::Projection;
    use crate::parquet::types::table::ParquetTable;
    use crate::parquet::types::thrift::general::Encoding;
    use crate::parquet::types::thrift::headers::PageHeader;
    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use bytes::Bytes;
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::{run_unary, run_unary_to_completion};
    use std::sync::Arc;

    fn make_test_table(schema: SchemaRef, num_rows: i64) -> Arc<ParquetTable> {
        let num_cols = schema.fields().len();
        let file = Arc::new(std::fs::File::open("/dev/null").unwrap());
        Arc::new(ParquetTable::new(vec![Arc::new(RowGroupMetadata {
            location: dispatch::io::FileLocation::Local(file),
            schema,
            columns: (0..num_cols)
                .map(|_| ColumnChunkMeta {
                    dictionary_page_offset: None,
                    data_page_offset: 0,
                    total_compressed_size: 0,
                    max_def_level: 0,
                    statistics: None,
                    data_pages_all_dictionary: false,
                })
                .collect(),
            num_rows,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })]))
    }

    fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn make_data_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        data: Vec<u8>,
        num_values: usize,
        page_idx: usize,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn make_skipped_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        num_values: usize,
        page_idx: usize,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            data: DecompressedPageType::SkippedData {
                header: header.data_page_header.unwrap(),
            },
        }
    }

    fn extract_i32s(batch: &RecordBatch, col: usize) -> Vec<i32> {
        let arr = batch.column(col);
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn i32_schema(names: &[&str]) -> SchemaRef {
        Arc::new(Schema::new(
            names
                .iter()
                .map(|n| Field::new(*n, DataType::Int32, false))
                .collect::<Vec<_>>(),
        ))
    }

    /// A table over `(s: Utf8View, v: Int32)` with `s` marked fully
    /// dictionary encoded (or not), the shape the view-equality tests need.
    fn string_and_i32_table(num_rows: i64, all_dictionary: bool) -> Arc<ParquetTable> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let file = Arc::new(std::fs::File::open("/dev/null").unwrap());
        let column = |dict| ColumnChunkMeta {
            dictionary_page_offset: None,
            data_page_offset: 0,
            total_compressed_size: 0,
            max_def_level: 0,
            statistics: None,
            data_pages_all_dictionary: dict,
        };
        Arc::new(ParquetTable::new(vec![Arc::new(RowGroupMetadata {
            location: dispatch::io::FileLocation::Local(file),
            schema,
            columns: vec![column(all_dictionary), column(false)],
            num_rows,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })]))
    }

    fn make_dict_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        entries: &[&str],
    ) -> DecompressedPage {
        let mut data = Vec::new();
        for s in entries {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        let header = PageHeader::for_dict_page(entries.len() as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(data)],
            },
        }
    }

    /// An RLE-dictionary data page holding `keys` (bit width 8: one RLE run
    /// per key keeps the encoding trivial).
    fn make_rle_data_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        keys: &[u8],
        page_idx: usize,
    ) -> DecompressedPage {
        let mut data = vec![8u8];
        for &key in keys {
            data.push(1 << 1);
            data.push(key);
        }
        let header = PageHeader::for_data_page(keys.len() as i32, Encoding::RLE_DICTIONARY);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn string_eq_predicate(column_idx: usize, value: &str) -> ScanEqualityPredicate {
        ScanEqualityPredicate {
            column_idx,
            value: Scalar::new(
                Arc::new(arrow_array::StringViewArray::from(vec![value])) as ArrayRef
            ),
        }
    }

    fn new_decoder(table: &Arc<ParquetTable>, batch_size: usize) -> Decoder {
        Decoder::new(
            batch_size,
            Projection::all_from_schema(table.schema()),
            false,
            Arc::new(Vec::new()),
        )
    }

    fn decoder_with_eq(table: &Arc<ParquetTable>, predicate: ScanEqualityPredicate) -> Decoder {
        Decoder::new(
            1024,
            Projection::all_from_schema(table.schema()),
            false,
            Arc::new(vec![predicate]),
        )
    }

    fn extract_strings(batch: &RecordBatch, col: usize) -> Vec<String> {
        let a = batch
            .column(col)
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
            .unwrap();
        (0..a.len()).map(|i| a.value(i).to_string()).collect()
    }

    /// A pushed string equality filters emitted batches down to matching rows
    /// via view equality against the dictionary's view of the constant.
    #[test]
    fn view_eq_filters_batches_to_matching_rows() {
        init_test_free_pool(4);
        let table = string_and_i32_table(5, true);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let dict = make_dict_page(metadata.clone(), 0, &["MAIL", "DELIVER IN PERSON", "SHIP"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 2, 1, 0], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3, 4, 5]), 5, 0);
        let decoder = decoder_with_eq(&table, string_eq_predicate(0, "DELIVER IN PERSON"));

        let out = run_unary_to_completion(decoder, vec![dict, strings, values]);

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![2, 4]);
        let kept: Vec<String> = out.iter().flat_map(|b| extract_strings(b, 0)).collect();
        assert_eq!(kept, vec!["DELIVER IN PERSON", "DELIVER IN PERSON"]);
    }

    /// A duplicated dictionary entry makes the single-view test unfaithful,
    /// so batches pass through unfiltered (the upstream Filter still applies
    /// the predicate).
    #[test]
    fn duplicate_dictionary_entries_disable_view_eq() {
        init_test_free_pool(4);
        let table = string_and_i32_table(3, true);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let dict = make_dict_page(metadata.clone(), 0, &["AIR", "AIR", "RAIL"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 2], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3]), 3, 0);
        let decoder = decoder_with_eq(&table, string_eq_predicate(0, "AIR"));

        let out = run_unary_to_completion(decoder, vec![dict, strings, values]);

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![1, 2, 3]);
    }

    /// A chunk with non-dictionary data pages must not view-filter: equal
    /// strings from a plain page carry different views.
    #[test]
    fn mixed_encoding_chunk_is_not_view_filtered() {
        init_test_free_pool(4);
        let table = string_and_i32_table(3, false);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let dict = make_dict_page(metadata.clone(), 0, &["MAIL", "SHIP"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 0], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3]), 3, 0);
        let decoder = decoder_with_eq(&table, string_eq_predicate(0, "MAIL"));

        let out = run_unary_to_completion(decoder, vec![dict, strings, values]);

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![1, 2, 3]);
    }

    /// Single Int32 column, one page → one RecordBatch with correct values.
    #[test]
    fn test_single_column_single_page() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0);

        let out = run_unary(new_decoder(&table, 1024), vec![page]);

        assert_eq!(out.len(), 1);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20, 30, 40, 50]);
    }

    /// Two Int32 columns — batch produced only after both columns have data.
    #[test]
    fn test_two_columns_batch_on_completion() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a", "b"]);
        let table = make_test_table(schema, 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page_a = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0);
        let page_b = make_data_page(metadata, 1, encode_i32s(&[40, 50, 60]), 3, 0);

        let out = run_unary(new_decoder(&table, 1024), vec![page_a, page_b]);

        assert_eq!(out.len(), 1);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20, 30]);
        assert_eq!(extract_i32s(&out[0], 1), vec![40, 50, 60]);
    }

    /// batch_size smaller than available rows → multiple batches produced via finish().
    #[test]
    fn test_batch_size_splits_output() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0);

        let out = run_unary_to_completion(new_decoder(&table, 2), vec![page]);

        assert_eq!(out.len(), 3);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20]);
        assert_eq!(extract_i32s(&out[1], 0), vec![30, 40]);
        assert_eq!(extract_i32s(&out[2], 0), vec![50]);
    }

    /// Two data pages for a single column — values from both pages concatenated.
    #[test]
    fn test_multiple_pages_single_column() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page0 = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0);
        let page1 = make_data_page(metadata, 0, encode_i32s(&[40, 50]), 2, 1);

        let out = run_unary_to_completion(new_decoder(&table, 1024), vec![page0, page1]);

        // Each page is eagerly emitted as a separate batch during consume.
        assert_eq!(out.len(), 2);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20, 30]);
        assert_eq!(extract_i32s(&out[1], 0), vec![40, 50]);
    }

    /// Pages from an already-exhausted row group are silently dropped.
    #[test]
    fn test_pages_from_exhausted_row_group_dropped() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0);
        let late_page = make_data_page(metadata, 0, encode_i32s(&[99]), 1, 1);

        let out = run_unary(new_decoder(&table, 1024), vec![page, late_page]);

        assert_eq!(out.len(), 1);
    }

    /// SkippedData page (all-false filter mask) produces no output.
    #[test]
    fn test_skipped_data_page() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, Some(vec![]));
        let page = make_skipped_page(metadata, 0, 3, 0);

        let out = run_unary(new_decoder(&table, 1024), vec![page]);

        assert_eq!(out.len(), 0);
    }

    /// Output schema has correct field names and types.
    #[test]
    fn test_output_schema() {
        init_test_free_pool(4);
        let schema = i32_schema(&["x", "y"]);
        let table = make_test_table(schema, 2);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page_x = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20]), 2, 0);
        let page_y = make_data_page(metadata, 1, encode_i32s(&[30, 40]), 2, 0);

        let out = run_unary(new_decoder(&table, 1024), vec![page_x, page_y]);

        assert_eq!(out[0].schema().fields().len(), 2);
        assert_eq!(out[0].schema().field(0).name(), "x");
        assert_eq!(out[0].schema().field(1).name(), "y");
        assert_eq!(*out[0].schema().field(0).data_type(), DataType::Int32);
    }

    /// finish() drains remaining batches when batch_size < total rows.
    #[test]
    fn test_finish_drains_remaining() {
        init_test_free_pool(4);
        let schema = i32_schema(&["a"]);
        let table = make_test_table(schema, 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, None);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30]), 3, 0);

        let out = run_unary_to_completion(new_decoder(&table, 2), vec![page]);

        assert_eq!(out.len(), 2);
    }
}
