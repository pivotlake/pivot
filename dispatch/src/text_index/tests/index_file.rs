use super::{encoded_from, token};
use crate::io::FileRange;
use crate::memory::{FileBytes, Slab, SlabAllocator, init_test_free_pool};
use crate::text_index::{
    DictIndex, IndexFileFooter, IndexFileWriter, Postings, SparseIndex, TOKENS_PER_DICT_INDEX,
    TokenRows,
};

/// The rows the `index`-th token of a test file appears in: a few for every
/// other token, enough to need postings for the rest.
fn rows_of_token(index: usize) -> Vec<u32> {
    let row_count = if index.is_multiple_of(2) { 3 } else { 20 };
    (0..row_count)
        .map(|position| (index + position * 1_000) as u32)
        .collect()
}

/// An index file of `token_count` tokens, the `index`-th being `token(index)`
/// in `rows_of_token(index)`.
fn write_index_file(token_count: usize) -> FileBytes {
    let mut allocator = SlabAllocator::new(false);
    let mut writer = IndexFileWriter::new();
    for index in 0..token_count {
        writer.start_token(&mut allocator, &token(index));
        for row in rows_of_token(index) {
            writer.push_row(&mut allocator, row);
        }
    }
    writer.finish(&mut allocator)
}

fn bytes_of(file: &FileBytes) -> Vec<u8> {
    file.runs().flatten().copied().collect()
}

/// The bytes of `file` at `range`, as a read of that range would return them.
fn read_range(file: &[u8], range: FileRange) -> Vec<Slab> {
    encoded_from(&file[range.offset..range.offset + range.len])
}

fn read_footer(file: &[u8]) -> IndexFileFooter {
    let footer_bytes = file[file.len() - IndexFileFooter::ENCODED_LEN..]
        .try_into()
        .unwrap();
    IndexFileFooter::decode(footer_bytes, file.len()).unwrap()
}

fn read_sparse_index(file: &[u8]) -> SparseIndex {
    let mut allocator = SlabAllocator::new(false);
    let encoded = read_range(file, read_footer(file).sparse_index);
    SparseIndex::decode(&encoded, &mut allocator)
        .unwrap()
        .finish()
}

/// The rows of `token` as a lookup finds them in `file`: through the footer,
/// the sparse index, one dict index and, for a token in many rows, its
/// postings.
fn find_rows(file: &[u8], token: &[u8]) -> Option<Vec<u32>> {
    let mut allocator = SlabAllocator::new(false);
    let dict_index_range = read_sparse_index(file).find_dict_index(token)?;
    let dict_index = DictIndex::decode(&read_range(file, dict_index_range), &mut allocator)
        .unwrap()
        .finish();
    match dict_index.find_rows(token)? {
        TokenRows::KeptInDict(few_rows) => Some(few_rows.as_slice().to_vec()),
        TokenRows::InPostings { postings, .. } => {
            let postings = Postings::decode(&read_range(file, postings), &mut allocator)
                .unwrap()
                .finish(&mut allocator);
            let mut rows = Vec::new();
            postings.for_each_row(|row| rows.push(row));
            Some(rows)
        }
    }
}

#[test]
fn every_token_written_is_found_with_its_rows_starting_from_the_footer() {
    init_test_free_pool(16);
    let file = bytes_of(&write_index_file(3_000));

    let tokens_found_wrong = (0..3_000)
        .step_by(7)
        .filter(|&index| find_rows(&file, &token(index)) != Some(rows_of_token(index)))
        .count();

    assert_eq!(tokens_found_wrong, 0);
}

#[test]
fn a_token_that_was_not_written_is_not_found() {
    init_test_free_pool(16);
    let file = bytes_of(&write_index_file(3_000));

    let found = [&b"a"[..], b"token00000001x", b"zebra"].map(|token| find_rows(&file, token));

    assert_eq!(found, [None, None, None]);
}

#[test]
fn a_dict_index_is_closed_every_so_many_tokens() {
    init_test_free_pool(16);
    let file = bytes_of(&write_index_file(2 * TOKENS_PER_DICT_INDEX + 1));

    let sparse_index = read_sparse_index(&file);

    assert_eq!(sparse_index.count_dict_indexes(), 3);
}

#[test]
fn a_file_of_no_tokens_still_ends_in_a_footer_and_an_empty_sparse_index() {
    init_test_free_pool(4);
    let file = bytes_of(&write_index_file(0));

    let sparse_index = read_sparse_index(&file);

    assert_eq!(sparse_index.count_dict_indexes(), 0);
    assert_eq!(find_rows(&file, b"apple"), None);
}

#[test]
fn the_postings_of_a_dict_index_are_written_before_it() {
    init_test_free_pool(8);
    let file = bytes_of(&write_index_file(10));
    let mut allocator = SlabAllocator::new(false);
    let dict_index_range = read_sparse_index(&file).find_dict_index(&token(1)).unwrap();
    let dict_index = DictIndex::decode(&read_range(&file, dict_index_range), &mut allocator)
        .unwrap()
        .finish();

    let Some(TokenRows::InPostings { postings, .. }) = dict_index.find_rows(&token(1)) else {
        panic!("a token in twenty rows has postings");
    };

    assert_eq!(postings.offset, 0);
    assert!(postings.offset + postings.len <= dict_index_range.offset);
}

#[test]
fn bytes_that_do_not_end_as_an_index_file_does_are_not_a_footer() {
    let not_a_footer = [0u8; IndexFileFooter::ENCODED_LEN];

    let decoded = IndexFileFooter::decode(&not_a_footer, 1_000);

    assert!(decoded.is_err());
}

#[test]
fn a_footer_whose_sparse_index_does_not_end_where_it_starts_is_refused() {
    init_test_free_pool(4);
    let file = bytes_of(&write_index_file(10));
    let footer_bytes = file[file.len() - IndexFileFooter::ENCODED_LEN..]
        .try_into()
        .unwrap();

    let decoded = IndexFileFooter::decode(footer_bytes, file.len() + 1);

    assert!(decoded.is_err());
}

#[test]
#[should_panic(expected = "at least one row")]
fn a_token_started_without_any_row_is_refused() {
    init_test_free_pool(4);
    let mut allocator = SlabAllocator::new(false);
    let mut writer = IndexFileWriter::new();
    writer.start_token(&mut allocator, b"apple");

    writer.start_token(&mut allocator, b"date");
}
