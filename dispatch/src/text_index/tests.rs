//! The index's three sections, each written in its file form and read back
//! through its cache.

use super::dict_block::{DictBlockEncoder, PostingsLocation, TokenEntry, TokenRows};
use super::postings::encode_postings;
use super::sparse_index::SparseIndexEncoder;
use super::*;
use crate::io::{FileRange, LocalFile, OpenFile};
use crate::memory::cache::CacheKey;
use crate::memory::{init_test_free_pool, memory_ctx};
use bytes::Bytes;
use std::ops::Range;

fn key() -> CacheKey {
    let file = LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
    CacheKey::new(OpenFile::Local(file), 0, 1)
}

fn encoded_postings(rows: &[u32]) -> Vec<u8> {
    let mut encoded = Vec::new();
    encode_postings(rows, &mut encoded);
    encoded
}

fn cached_postings(fragments: &[Bytes]) -> Result<Postings, Error> {
    memory_ctx()
        .text_index_caches()
        .postings
        .insert(key(), |fill| Postings::build(fragments, fill))
}

/// The rows of `postings` inside `rows`, as file rows.
fn rows_within(postings: &Postings, rows: Range<u32>) -> Vec<u32> {
    let mut mask = RowMask::none(rows.len());
    postings.select_rows(rows.clone(), &mut mask);
    mask.selected_rows().map(|row| rows.start + row).collect()
}

#[test]
fn a_sparse_posting_list_reads_back_its_rows() {
    init_test_free_pool(4);
    let rows = [3, 70_000, 70_001, 200_000];

    let postings = cached_postings(&[Bytes::from(encoded_postings(&rows))]).unwrap();

    assert_eq!(rows_within(&postings, 0..300_000), rows);
    assert_eq!(postings.row_count(), 4);
}

#[test]
fn a_dense_posting_list_reads_back_its_rows() {
    init_test_free_pool(4);
    let rows: Vec<u32> = (0..20_000).step_by(2).collect();

    let postings = cached_postings(&[Bytes::from(encoded_postings(&rows))]).unwrap();

    assert_eq!(rows_within(&postings, 0..20_000), rows);
}

#[test]
fn a_posting_list_selects_only_the_rows_of_the_range_asked_for() {
    init_test_free_pool(4);
    let dense: Vec<u32> = (0..20_000).step_by(2).collect();
    let sparse = [5, 101, 65_600, 65_700];
    let dense_postings = cached_postings(&[Bytes::from(encoded_postings(&dense))]).unwrap();
    let sparse_postings = cached_postings(&[Bytes::from(encoded_postings(&sparse))]).unwrap();

    let dense_rows = rows_within(&dense_postings, 101..9_001);
    let sparse_rows = rows_within(&sparse_postings, 101..65_700);

    let expected_dense: Vec<u32> = (102..9_001).step_by(2).collect();
    assert_eq!(dense_rows, expected_dense);
    assert_eq!(sparse_rows, vec![101, 65_600]);
}

#[test]
fn a_posting_list_read_in_fragments_is_put_back_together() {
    init_test_free_pool(4);
    let rows: Vec<u32> = (0..20_000).step_by(2).collect();
    let encoded = Bytes::from(encoded_postings(&rows));
    let fragments = [
        encoded.slice(..13),
        encoded.slice(13..5_000),
        encoded.slice(5_000..),
    ];

    let postings = cached_postings(&fragments).unwrap();

    assert_eq!(rows_within(&postings, 0..20_000), rows);
}

#[test]
fn a_truncated_posting_list_is_rejected() {
    init_test_free_pool(4);
    let encoded = Bytes::from(encoded_postings(&[1, 2, 3]));

    let postings = cached_postings(&[encoded.slice(..encoded.len() - 1)]);

    assert!(matches!(postings, Err(Error::Corrupt(_))));
}

fn cached_dict_block(encoded: &[u8]) -> Result<DictBlock, Error> {
    memory_ctx()
        .text_index_caches()
        .dict_blocks
        .insert(key(), |fill| DictBlock::build(encoded, fill))
}

const TIMEOUT_POSTINGS: PostingsLocation = PostingsLocation {
    offset: 4096,
    len: 300,
    row_count: 120,
};

/// A block holding `error` (rows 1 and 5), `errors` (row 7) and `timeout`
/// (a posting list).
fn encoded_dict_block() -> Vec<u8> {
    let mut block = DictBlockEncoder::new();
    block.push(b"error", TokenRows::Inline(&[1, 5]));
    block.push(b"errors", TokenRows::Inline(&[7]));
    block.push(b"timeout", TokenRows::List(TIMEOUT_POSTINGS));
    let mut encoded = Vec::new();
    block.finish(&mut encoded);
    encoded
}

fn inline_rows(entry: Option<TokenEntry<'_>>) -> Vec<u32> {
    match entry {
        Some(TokenEntry::Inline(rows)) => rows.iter().collect(),
        other => panic!("expected inline rows, found {other:?}"),
    }
}

#[test]
fn a_dictionary_block_finds_the_rows_kept_with_a_token() {
    init_test_free_pool(4);

    let block = cached_dict_block(&encoded_dict_block()).unwrap();

    assert_eq!(inline_rows(block.find(b"error")), vec![1, 5]);
    assert_eq!(inline_rows(block.find(b"errors")), vec![7]);
}

#[test]
fn a_dictionary_block_finds_where_a_common_tokens_posting_list_is() {
    init_test_free_pool(4);

    let block = cached_dict_block(&encoded_dict_block()).unwrap();

    assert!(matches!(
        block.find(b"timeout"),
        Some(TokenEntry::List(list)) if list == TIMEOUT_POSTINGS
    ));
}

#[test]
fn a_dictionary_block_does_not_find_a_token_it_does_not_hold() {
    init_test_free_pool(4);

    let block = cached_dict_block(&encoded_dict_block()).unwrap();

    assert!(block.find(b"err").is_none());
    assert!(block.find(b"zebra").is_none());
}

#[test]
fn a_truncated_dictionary_block_is_rejected() {
    init_test_free_pool(4);
    let encoded = encoded_dict_block();

    let block = cached_dict_block(&encoded[..encoded.len() - 1]);

    assert!(matches!(block, Err(Error::Corrupt(_))));
}

fn cached_sparse_index(first_tokens: &[&[u8]]) -> SparseIndex {
    let mut sparse_index = SparseIndexEncoder::new();
    for (block, first_token) in first_tokens.iter().enumerate() {
        sparse_index.push(first_token, FileRange::new(1000 * block, 1000));
    }
    let mut encoded = Vec::new();
    sparse_index.finish(&mut encoded).unwrap();
    let footer = format::Footer {
        sparse_offset: 0,
        sparse_len: encoded.len() as u64,
        row_count: 10,
        token_count: 3000,
    };
    memory_ctx()
        .text_index_caches()
        .sparse_index
        .insert(key(), |fill| SparseIndex::build(footer, &encoded, fill))
        .unwrap()
}

#[test]
fn a_sparse_index_finds_the_block_a_token_sorts_into() {
    init_test_free_pool(4);

    let sparse_index = cached_sparse_index(&[b"apple", b"date", b"melon"]);

    assert_eq!(
        sparse_index.find_block(b"kiwi"),
        Some(FileRange::new(1000, 1000))
    );
    assert_eq!(
        sparse_index.find_block(b"melon"),
        Some(FileRange::new(2000, 1000))
    );
    assert_eq!(
        sparse_index.find_block(b"zebra"),
        Some(FileRange::new(2000, 1000))
    );
}

#[test]
fn a_sparse_index_has_no_block_for_a_token_before_its_first() {
    init_test_free_pool(4);

    let sparse_index = cached_sparse_index(&[b"apple", b"date", b"melon"]);

    assert_eq!(sparse_index.find_block(b"aardvark"), None);
}

#[test]
fn a_sparse_index_carries_its_files_footer() {
    init_test_free_pool(4);

    let sparse_index = cached_sparse_index(&[b"apple"]);

    assert_eq!(sparse_index.footer().row_count, 10);
}
