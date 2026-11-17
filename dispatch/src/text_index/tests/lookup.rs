use super::{bytes_of, token};
use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::{SlabAllocator, init_test_free_pool};
use crate::text_index::{
    IndexFileFooter, IndexFileWriter, LookupProgress, RowsWithToken, TokenLookup,
};

/// The rows the `index`-th token of a test file appears in: a few for every
/// other token, enough to need postings for the rest.
fn rows_of_token(index: usize) -> Vec<u32> {
    let row_count = if index.is_multiple_of(2) { 3 } else { 20 };
    (0..row_count)
        .map(|position| (index + position * 1_000) as u32)
        .collect()
}

/// The bytes of an index file of `token_count` tokens, the `index`-th being
/// `token(index)` in `rows_of_token(index)`.
fn write_index_file(token_count: usize) -> Vec<u8> {
    let mut allocator = SlabAllocator::new(false);
    let mut writer = IndexFileWriter::new();
    for index in 0..token_count {
        writer.start_token(&mut allocator, &token(index));
        for row in rows_of_token(index) {
            writer.push_row(row);
        }
    }
    bytes_of(&writer.finish(&mut allocator))
}

/// A file to cache an index's pieces under.
fn open_index_file() -> OpenFile {
    OpenFile::Local(LocalFile::new(tempfile::tempfile().unwrap()).unwrap())
}

/// Run `lookup` to its answer, serving each read it asks for from `file`.
/// Returns what it found and the ranges it read.
fn run_lookup(mut lookup: TokenLookup, file: &[u8]) -> (RowsWithToken, Vec<FileRange>) {
    let mut reads = Vec::new();
    loop {
        match lookup.advance() {
            LookupProgress::NeedsRead(range) => {
                reads.push(range);
                lookup
                    .supply(&[&file[range.offset..range.offset + range.len]])
                    .unwrap();
            }
            LookupProgress::Found(rows) => return (rows, reads),
        }
    }
}

fn rows_found(rows: &RowsWithToken) -> Vec<u32> {
    let mut found = Vec::new();
    rows.for_each_row_in(0..u32::MAX, |row| found.push(row));
    found
}

#[test]
fn a_token_in_few_rows_is_found_with_the_tail_and_its_dict_index_read() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let lookup = TokenLookup::new(open_index_file(), file.len(), &token(1_500));

    let (rows, reads) = run_lookup(lookup, &file);

    assert_eq!(rows_found(&rows), rows_of_token(1_500));
    assert_eq!(reads.len(), 2);
}

#[test]
fn a_token_in_many_rows_is_found_with_its_postings_read_too() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let lookup = TokenLookup::new(open_index_file(), file.len(), &token(1_501));

    let (rows, reads) = run_lookup(lookup, &file);

    assert_eq!(rows_found(&rows), rows_of_token(1_501));
    assert_eq!(rows.count_rows(), 20);
    assert_eq!(reads.len(), 3);
}

#[test]
fn a_token_the_file_does_not_hold_appears_in_no_row() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let index_file = open_index_file();

    let found = [&b"a"[..], b"token00000007x", b"zebra"].map(|token| {
        let lookup = TokenLookup::new(index_file.clone(), file.len(), token);
        rows_found(&run_lookup(lookup, &file).0)
    });

    assert!(found.iter().all(Vec::is_empty), "{found:?}");
}

#[test]
fn a_second_lookup_reads_nothing_that_the_first_left_cached() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let index_file = open_index_file();
    let first = TokenLookup::new(index_file.clone(), file.len(), &token(1_500));
    run_lookup(first, &file);

    let second = TokenLookup::new(index_file, file.len(), &token(1_502));
    let (rows, reads) = run_lookup(second, &file);

    assert_eq!(rows_found(&rows), rows_of_token(1_502));
    assert!(reads.is_empty(), "{reads:?}");
}

#[test]
fn a_sparse_index_that_does_not_come_with_the_footer_is_read_next() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let lookup = TokenLookup::new(open_index_file(), file.len(), &token(1_500))
        .with_tail_read_len(IndexFileFooter::ENCODED_LEN);

    let (rows, reads) = run_lookup(lookup, &file);

    assert_eq!(rows_found(&rows), rows_of_token(1_500));
    assert_eq!(reads.len(), 3);
    assert_eq!(reads[0].len, IndexFileFooter::ENCODED_LEN);
}

#[test]
fn only_the_rows_of_the_range_asked_for_are_visited() {
    init_test_free_pool(16);
    let file = write_index_file(3_000);
    let lookup = TokenLookup::new(open_index_file(), file.len(), &token(1_501));
    let (rows, _) = run_lookup(lookup, &file);

    let mut found = Vec::new();
    rows.for_each_row_in(3_000..6_000, |row| found.push(row));

    assert_eq!(found, [3_501, 4_501, 5_501]);
}

#[test]
fn bytes_that_are_not_the_end_of_an_index_file_fail_the_lookup() {
    init_test_free_pool(4);
    let not_an_index_file = [7u8; 100];
    let mut lookup = TokenLookup::new(open_index_file(), not_an_index_file.len(), b"apple");
    let LookupProgress::NeedsRead(range) = lookup.advance() else {
        panic!("nothing is cached, so the lookup reads");
    };

    let supplied = lookup.supply(&[&not_an_index_file[range.offset..range.offset + range.len]]);

    assert!(supplied.is_err());
}
