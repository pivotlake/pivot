//! End-to-end tests for the text indexes a write comes with: drive the whole
//! write pipeline over rows of text and look tokens up in the index files it
//! hands out beside the data files.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use dispatch::io::FileRange;
use dispatch::memory::{Slab, SlabAllocator};
use dispatch::text_index::{
    DictIndex, IndexFileFooter, LetterCase, Postings, SparseIndex, TokenRows,
};
use dispatch::{BUFFER_SIZE, Dispatch, values_input};
use object_storage::ObjectPath;

use super::{
    AssembledFile, Partitioning, TextIndexDefinition, TextIndexFile, encode_record_batches_spec,
};

/// A 64 MiB file-cache ring, as the other in-crate write tests use.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

/// Rows of a sort key and a text, in the order given.
fn rows_of_text(rows: &[(i64, Option<&str>)]) -> RecordBatch {
    let keys: ArrayRef = Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0)));
    let texts: ArrayRef = Arc::new(StringViewArray::from_iter(rows.iter().map(|row| row.1)));
    let schema = Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("message", DataType::Utf8View, true),
    ]);
    RecordBatch::try_new(Arc::new(schema), vec![keys, texts]).unwrap()
}

fn index_of(column: &str, letter_case: LetterCase) -> Arc<[TextIndexDefinition]> {
    Arc::from([TextIndexDefinition {
        name: format!("idx_{column}"),
        column: column.to_string(),
        letter_case,
    }])
}

/// Write `batch` as one file sorted by `key`, in row groups of
/// `rows_per_group` rows, with `text_indexes`. Returns the bytes of each
/// index file the data file came with, by its index's name, or the error
/// the write failed with.
fn write_text_indexes(
    batch: RecordBatch,
    rows_per_group: usize,
    text_indexes: Arc<[TextIndexDefinition]>,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let schema = batch.schema();
    let spec = values_input(dispatch.dispatcher(), vec![batch]).record_batches();
    // Copy worker-owned ring memory before returning the bytes to the test
    // thread.
    let files = encode_record_batches_spec(
        spec,
        schema,
        Partitioning::default(),
        Arc::from(["key".to_string()]),
        rows_per_group,
        usize::MAX,
        text_indexes,
    )
    .map_each(|file: AssembledFile| {
        Ok(file
            .text_indexes
            .iter()
            .map(|TextIndexFile { index_name, bytes }| {
                (
                    index_name.clone(),
                    bytes.runs().flatten().copied().collect::<Vec<u8>>(),
                )
            })
            .collect::<Vec<_>>())
    })
    .collect();
    dispatch.exit();
    let mut files = files.map_err(|error| format!("{error:?}"))?;
    assert_eq!(
        files.len(),
        1,
        "an unpartitioned write of a few rows is one file"
    );
    Ok(files.remove(0))
}

/// The rows of `tokens` as lookups find them in the index file `index`, each
/// through the footer, the sparse index, one dict index and, for a token in
/// many rows, its postings.
fn find_rows(index: &[u8], tokens: &[&str]) -> Vec<Option<Vec<u32>>> {
    // Decoding needs ring memory, so a memory context of its own: a thread
    // holds only one, and the pipeline's workers held theirs.
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                dispatch::memory::init_test_free_pool(16);
                let mut allocator = SlabAllocator::new(false);
                let read = |range: FileRange, allocator: &mut SlabAllocator| -> Vec<Slab> {
                    let mut slab = allocator.get_slab_of_size(range.len, false);
                    slab.as_mut_slice()
                        .copy_from_slice(&index[range.offset..range.offset + range.len]);
                    vec![slab]
                };
                let footer_bytes = index[index.len() - IndexFileFooter::ENCODED_LEN..]
                    .try_into()
                    .unwrap();
                let footer = IndexFileFooter::decode(footer_bytes, index.len()).unwrap();
                let sparse_index = read(footer.sparse_index, &mut allocator);
                let sparse_index = SparseIndex::decode(&sparse_index, &mut allocator)
                    .unwrap()
                    .finish();
                tokens
                    .iter()
                    .map(|token| {
                        let dict_index = sparse_index.find_dict_index(token.as_bytes())?;
                        let dict_index = read(dict_index, &mut allocator);
                        let dict_index = DictIndex::decode(&dict_index, &mut allocator)
                            .unwrap()
                            .finish();
                        match dict_index.find_rows(token.as_bytes())? {
                            TokenRows::KeptInDict(few_rows) => Some(few_rows.as_slice().to_vec()),
                            TokenRows::InPostings { postings, .. } => {
                                let postings = read(postings, &mut allocator);
                                let postings = Postings::decode(&postings, &mut allocator)
                                    .unwrap()
                                    .finish(&mut allocator);
                                let mut rows = Vec::new();
                                postings.for_each_row(|row| rows.push(row));
                                Some(rows)
                            }
                        }
                    })
                    .collect()
            })
            .join()
            .unwrap()
    })
}

#[test]
fn a_text_indexed_column_comes_with_an_index_of_its_tokens_by_row_of_the_file() {
    let batch = rows_of_text(&[
        (3, Some("disk full")),
        (1, Some("net down")),
        (2, Some("disk ok")),
    ]);

    let indexes = write_text_indexes(batch, 1_024, index_of("message", LetterCase::Kept)).unwrap();

    // The file is sorted by key, so "net down" is its row 0.
    let (index_name, index) = &indexes[0];
    assert_eq!(index_name, "idx_message");
    assert_eq!(
        find_rows(index, &["disk", "full", "net", "ok", "absent"]),
        [
            Some(vec![1, 2]),
            Some(vec![2]),
            Some(vec![0]),
            Some(vec![1]),
            None
        ]
    );
}

#[test]
fn rows_of_every_row_group_are_indexed_at_their_place_in_the_file() {
    let rows: Vec<(i64, Option<String>)> = (0..10)
        .map(|key| (key, Some(format!("row{key} common"))))
        .collect();
    let rows: Vec<(i64, Option<&str>)> = rows
        .iter()
        .map(|(key, text)| (*key, text.as_deref()))
        .collect();

    let indexes = write_text_indexes(
        rows_of_text(&rows),
        3,
        index_of("message", LetterCase::Kept),
    )
    .unwrap();

    assert_eq!(
        find_rows(&indexes[0].1, &["common", "row0", "row4", "row9"]),
        [
            Some((0..10).collect()),
            Some(vec![0]),
            Some(vec![4]),
            Some(vec![9])
        ]
    );
}

#[test]
fn a_row_without_text_keeps_its_place_among_the_rows() {
    let batch = rows_of_text(&[(1, Some("first")), (2, None), (3, Some("third"))]);

    let indexes = write_text_indexes(batch, 1_024, index_of("message", LetterCase::Kept)).unwrap();

    assert_eq!(
        find_rows(&indexes[0].1, &["first", "third"]),
        [Some(vec![0]), Some(vec![2])]
    );
}

#[test]
fn an_index_of_lowered_letter_case_holds_a_word_once_whatever_its_spelling() {
    let batch = rows_of_text(&[(1, Some("Error")), (2, Some("ERROR")), (3, Some("error"))]);

    let indexes =
        write_text_indexes(batch, 1_024, index_of("message", LetterCase::Lowered)).unwrap();

    assert_eq!(
        find_rows(&indexes[0].1, &["error", "Error"]),
        [Some(vec![0, 1, 2]), None]
    );
}

#[test]
fn a_write_without_text_indexed_columns_comes_with_no_index() {
    let batch = rows_of_text(&[(1, Some("disk full"))]);

    let indexes = write_text_indexes(batch, 1_024, Arc::default()).unwrap();

    assert!(indexes.is_empty());
}

#[test]
fn a_text_index_of_a_column_that_is_not_text_fails_the_write() {
    let batch = rows_of_text(&[(1, Some("disk full"))]);

    let written = write_text_indexes(batch, 1_024, index_of("key", LetterCase::Kept));

    let error = written.unwrap_err();
    assert!(error.contains("TextIndexOfNonTextColumn"), "{error}");
}

#[test]
fn two_indexes_of_one_column_each_come_with_an_index_file_in_their_own_letter_case() {
    let batch = rows_of_text(&[(1, Some("Error")), (2, Some("error"))]);
    let text_indexes: Arc<[TextIndexDefinition]> = Arc::from([
        TextIndexDefinition {
            name: "as_written".to_string(),
            column: "message".to_string(),
            letter_case: LetterCase::Kept,
        },
        TextIndexDefinition {
            name: "lowered".to_string(),
            column: "message".to_string(),
            letter_case: LetterCase::Lowered,
        },
    ]);

    let indexes = write_text_indexes(batch, 1_024, text_indexes).unwrap();

    assert_eq!(indexes[0].0, "as_written");
    assert_eq!(find_rows(&indexes[0].1, &["Error"]), [Some(vec![0])]);
    assert_eq!(indexes[1].0, "lowered");
    assert_eq!(find_rows(&indexes[1].1, &["error"]), [Some(vec![0, 1])]);
}

#[test]
fn an_index_file_is_in_its_index_directory_under_the_name_of_its_data_file() {
    let data_file = ObjectPath::new("pivot-1234.parquet");

    let plain = TextIndexDefinition::path_of_index_file("idx_message", &data_file);
    let with_a_slash = TextIndexDefinition::path_of_index_file("a/b", &data_file);

    assert_eq!(plain.as_str(), "idx_message/pivot-1234.parquet.textindex");
    assert_eq!(with_a_slash.as_str(), "a%2fb/pivot-1234.parquet.textindex");
}
