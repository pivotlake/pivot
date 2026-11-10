//! A text index built and queried through dataflows: rows are tokenized in
//! parallel ranges, one worker assembles the index file, and lookups read it
//! back through dispatch IO and the text index caches.

mod common;

use common::*;
use dispatch::io::{LocalFile, OpenFile};
use dispatch::text_index::{
    FileTokenRun, IndexAssembler, IndexFile, IndexLookupsFactory, LookupRequest, TextQuery,
    TokenMatch, TokenRunBuilder,
};
use dispatch::{
    DataFlowDispatcher, DefaultUnaryFactory, return_to_worker_mpsc, stealable, values_input,
};
use std::ops::Range;
use std::sync::Arc;
use tempfile::TempDir;

const WORKERS: usize = 2;

/// An index file on disk, with the directory keeping it alive.
struct BuiltIndex {
    index: IndexFile,
    _directory: TempDir,
}

/// Index `texts` as rows `0..texts.len()` of one file, tokenizing
/// `rows_per_run` rows at a time.
fn build_index(dispatch: &DataFlowDispatcher, texts: &[String], rows_per_run: usize) -> BuiltIndex {
    let file_row_count = texts.len() as u32;
    let ranges: Vec<(u32, Vec<String>)> = texts
        .chunks(rows_per_run)
        .enumerate()
        .map(|(run, chunk)| ((run * rows_per_run) as u32, chunk.to_vec()))
        .collect();

    let mut files = values_input(dispatch, ranges)
        .map_each(move |(first_row, texts): (u32, Vec<String>)| {
            let mut builder = TokenRunBuilder::new();
            for (offset, text) in texts.iter().enumerate() {
                builder.add_row(first_row + offset as u32, text.as_bytes());
            }
            Ok(FileTokenRun {
                file: (),
                file_row_count,
                assembly_worker: 0,
                run: builder.finish(first_row..first_row + texts.len() as u32),
            })
        })
        .chain(
            return_to_worker_mpsc(WORKERS).into_iter().collect(),
            DefaultUnaryFactory::<IndexAssembler<()>>::create_for_workers(WORKERS),
        )
        .map_each(|assembled| {
            Ok(assembled
                .bytes
                .runs()
                .flatten()
                .copied()
                .collect::<Vec<u8>>())
        })
        .collect()
        .unwrap();

    let bytes = files.pop().unwrap();
    assert!(files.is_empty(), "one file, one index");
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("rows.textidx");
    std::fs::write(&path, &bytes).unwrap();
    let file = LocalFile::new(std::fs::File::open(path).unwrap()).unwrap();
    BuiltIndex {
        index: IndexFile {
            file: OpenFile::Local(file),
            len: bytes.len(),
        },
        _directory: directory,
    }
}

/// The rows of `rows` the index says match `text`'s tokens, as file rows.
fn matching_rows(
    dispatch: &DataFlowDispatcher,
    built: &BuiltIndex,
    text: &str,
    token_match: TokenMatch,
    rows: Range<u32>,
) -> Vec<u32> {
    let query = Arc::new(TextQuery::from_text(text.as_bytes(), token_match));
    let request = LookupRequest {
        index: built.index.clone(),
        rows: rows.clone(),
        payload: rows.start,
    };

    let mut answers = values_input(dispatch, [request])
        .chain(
            stealable(dispatch.topology()).into_iter().collect(),
            IndexLookupsFactory::create_for_workers(WORKERS, query),
        )
        .map_each(|result| {
            let first_row: u32 = result.payload;
            Ok(result
                .matching
                .selected_rows()
                .map(|row| first_row + row)
                .collect::<Vec<u32>>())
        })
        .collect()
        .unwrap();

    answers.pop().unwrap()
}

fn log_lines() -> Vec<String> {
    [
        "GET /users timeout",
        "POST /users ok",
        "GET /orders timeout retry",
        "DELETE /orders ok",
        "",
        "GET /users ok",
    ]
    .map(String::from)
    .to_vec()
}

#[test]
fn a_token_is_found_in_the_rows_holding_it() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let rows = matching_rows(&dispatch, &built, "timeout", TokenMatch::All, 0..6);

    assert_eq!(rows, vec![0, 2]);
}

#[test]
fn all_tokens_must_be_in_a_row_for_it_to_match() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let rows = matching_rows(&dispatch, &built, "GET users", TokenMatch::All, 0..6);

    assert_eq!(rows, vec![0, 5]);
}

#[test]
fn any_token_in_a_row_makes_it_match() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let rows = matching_rows(&dispatch, &built, "retry DELETE", TokenMatch::Any, 0..6);

    assert_eq!(rows, vec![2, 3]);
}

#[test]
fn a_token_the_index_does_not_hold_matches_no_row() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let all = matching_rows(&dispatch, &built, "GET missing", TokenMatch::All, 0..6);
    let any = matching_rows(&dispatch, &built, "absent missing", TokenMatch::Any, 0..6);

    assert!(all.is_empty());
    assert!(any.is_empty());
}

#[test]
fn matching_is_case_sensitive_and_whole_token() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let lowercase = matching_rows(&dispatch, &built, "get", TokenMatch::All, 0..6);
    let part_of_a_token = matching_rows(&dispatch, &built, "user", TokenMatch::All, 0..6);

    assert!(lowercase.is_empty());
    assert!(part_of_a_token.is_empty());
}

#[test]
fn a_lookup_answers_only_for_the_rows_asked_about() {
    let dispatch = dispatch(WORKERS);
    let built = build_index(&dispatch, &log_lines(), 2);

    let rows = matching_rows(&dispatch, &built, "GET", TokenMatch::All, 1..5);

    assert_eq!(rows, vec![2]);
}

/// A token in more rows than a dictionary entry keeps gets a posting list,
/// gathered here from many tokenized ranges.
#[test]
fn a_common_token_is_found_through_its_posting_list() {
    let dispatch = dispatch(WORKERS);
    let texts: Vec<String> = (0..10_000)
        .map(|row| match row % 3 {
            0 => "status ok".to_string(),
            _ => "status failed".to_string(),
        })
        .collect();
    let built = build_index(&dispatch, &texts, 1_000);

    let rows = matching_rows(&dispatch, &built, "status ok", TokenMatch::All, 0..10_000);

    let expected: Vec<u32> = (0..10_000).step_by(3).collect();
    assert_eq!(rows, expected);
}

/// Enough distinct tokens to fill several dictionary blocks, so the lookup
/// has to pick the right one through the sparse index.
#[test]
fn a_rare_token_is_found_among_many_dictionary_blocks() {
    let dispatch = dispatch(WORKERS);
    let texts: Vec<String> = (0..5_000)
        .map(|row| format!("request id{row:05}"))
        .collect();
    let built = build_index(&dispatch, &texts, 700);

    let rows = matching_rows(&dispatch, &built, "id03123", TokenMatch::All, 0..5_000);

    assert_eq!(rows, vec![3123]);
}

#[test]
fn a_second_lookup_gives_the_same_answer_from_the_caches() {
    let dispatch = dispatch(WORKERS);
    let texts: Vec<String> = (0..5_000)
        .map(|row| format!("request id{row:05}"))
        .collect();
    let built = build_index(&dispatch, &texts, 700);
    let first = matching_rows(
        &dispatch,
        &built,
        "request id00042",
        TokenMatch::All,
        0..5_000,
    );

    let second = matching_rows(
        &dispatch,
        &built,
        "request id00042",
        TokenMatch::All,
        0..5_000,
    );

    assert_eq!(first, vec![42]);
    assert_eq!(second, first);
}
