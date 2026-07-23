//! [`RegexpReplace`] — SQL `regexp_replace(input, pattern, replacement)`,
//! first-match replacement evaluated with the `regex` crate.
//!
//! Matching runs in *byte* mode (Unicode matching off), which matches
//! RE2/ClickHouse byte semantics and handles columns that aren't valid UTF-8.
//! The `regex` crate is the reference engine: its semantics are the ones every
//! `regexp_replace` row is guaranteed. The crate expands the replacement
//! itself (`Regex::replace` splices the first match), so all this needs is to
//! translate the SQL `\N` template into the crate's `$N` dialect once at
//! compile time.
//!
//! A caller who explicitly wants PCRE2's JIT (and accepts that its semantics
//! may differ) opts in through the separate
//! [`RegexpJitReplace`](super::regexp_jit::RegexpJitReplace), which shares the
//! [`replace_column`] driver below but splices matches by hand (the `pcre2`
//! crate has no native replace).

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch, StringViewArray};
use regex::bytes::{Captures, Regex as BytesRegex, RegexBuilder as BytesRegexBuilder};
use std::borrow::Cow;
use std::fmt::{self, Display};
use std::sync::{Arc, LazyLock};

/// SQL `regexp_replace(input, pattern, replacement)` — replaces the *first*
/// match of `pattern` in each row of `input` (DuckDB without the `'g'`
/// option). The replacement string uses PostgreSQL-style `\N` group
/// references. `pattern` and `replacement` must be constants so the regex can
/// be compiled once at plan-compile time.
#[derive(Debug, Clone)]
pub struct RegexpReplace {
    pub input: Box<Expression>,
    pub pattern: String,
    pub replacement: String,
}

impl Display for RegexpReplace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "regexp_replace({}, '{}', '{}')",
            self.input, self.pattern, self.replacement
        )
    }
}

impl RegexpReplace {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        build_regex(&self.pattern).map_err(|source| compile::Error::InvalidRegexPattern {
            pattern: self.pattern.clone(),
            source,
        })?;

        let pattern = self.pattern.clone();
        let replacement = Arc::new(translate_replacement(&self.replacement));
        let input_builder = self.input.compile()?;

        Ok(Box::new(move || {
            let replacement = replacement.clone();
            // A fresh `Regex` per worker: a cloned `Regex` funnels every match
            // through a shared pool guarded by atomics, contending in the row loop.
            let regex = build_regex(&pattern).expect("pattern validated at plan compile");
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                // `replace` returns `Borrowed` (the input, untouched) when nothing
                // matched, so a passthrough row is never copied.
                let result = replace_column(strings, |s| {
                    match regex.replace(s, replacement.as_slice()) {
                        Cow::Owned(replaced) => Some(replaced),
                        Cow::Borrowed(_) => None,
                    }
                });
                ExprResult::Array(result)
            }) as ExprEvalFn
        }))
    }
}

/// SQL `regexp_full_match(input, pattern)` — DuckDB's `~` operator. True when
/// `pattern` matches the **entire** `input` row (an unmatched null stays null).
/// `pattern` must be a constant so the regex compiles once at plan-compile time.
#[derive(Debug, Clone)]
pub struct RegexpFullMatch {
    pub input: Box<Expression>,
    pub pattern: String,
}

impl Display for RegexpFullMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "regexp_full_match({}, '{}')", self.input, self.pattern)
    }
}

impl RegexpFullMatch {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        build_regex(&self.pattern).map_err(|source| compile::Error::InvalidRegexPattern {
            pattern: self.pattern.clone(),
            source,
        })?;

        let pattern = self.pattern.clone();
        let input_builder = self.input.compile()?;

        Ok(Box::new(move || {
            let regex = build_regex(&pattern).expect("pattern validated at plan compile");
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                let result: BooleanArray = strings
                    .iter()
                    .map(|row| {
                        row.map(|value| {
                            let bytes = value.as_bytes();
                            // Full match: the match must span the whole row.
                            regex
                                .find(bytes)
                                .is_some_and(|m| m.start() == 0 && m.end() == bytes.len())
                        })
                    })
                    .collect();
                ExprResult::Array(Arc::new(result))
            }) as ExprEvalFn
        }))
    }
}

/// Build a byte-mode `regex` (Unicode matching off).
fn build_regex(pattern: &str) -> Result<BytesRegex, regex::Error> {
    BytesRegexBuilder::new(pattern).unicode(false).build()
}

/// Translate a PostgreSQL-style replacement into the `regex` crate's dialect so
/// the crate's own `replace` can expand it. Four tokens need rewriting: a
/// single-digit `\N` backreference becomes `${N}`, `\&` (the whole match)
/// becomes `${0}`, `\\` becomes a literal `\`, and a `$` (never special in the
/// SQL form) becomes `$$`, the crate's escape for a real dollar. Everything else
/// is left as is.
fn translate_replacement(replacement: &str) -> Vec<u8> {
    static TOKEN: LazyLock<BytesRegex> =
        LazyLock::new(|| BytesRegex::new(r"\\([0-9])|(\\&)|(\\\\)|\$").unwrap());
    TOKEN
        .replace_all(replacement.as_bytes(), |caps: &Captures| {
            if let Some(digit) = caps.get(1) {
                [b"${", digit.as_bytes(), b"}"].concat()
            } else if caps.get(2).is_some() {
                b"${0}".to_vec()
            } else if caps.get(3).is_some() {
                b"\\".to_vec()
            } else {
                b"$$".to_vec()
            }
        })
        .into_owned()
}

/// Apply a first-match replacement across every row of `strings`.
/// `replace_first` returns the replaced bytes for a row, or `None` to pass the
/// row through unchanged. Values arrive in runs (sessions repeat the same URL),
/// so an equal-to-previous input reuses the prior row's result without
/// re-matching. Shared with [`RegexpJitReplace`](super::regexp_jit::RegexpJitReplace).
pub(super) fn replace_column(
    strings: &StringViewArray,
    mut replace_first: impl FnMut(&[u8]) -> Option<Vec<u8>>,
) -> ArrayRef {
    let mut out = StringViewBuilder::with_capacity(strings.len());
    let mut last_input: Option<&str> = None;
    // `None` means "no match — pass the input through".
    let mut last_output: Option<Vec<u8>> = None;
    for value in strings.iter() {
        let Some(s) = value else {
            out.append_null();
            continue;
        };
        if last_input == Some(s) {
            match &last_output {
                // SAFETY: bytes came from matching/splicing this column's
                // (already `&str`) values.
                Some(bytes) => out.append_value(unsafe { std::str::from_utf8_unchecked(bytes) }),
                None => out.append_value(s),
            }
            continue;
        }
        last_input = Some(s);
        match replace_first(s.as_bytes()) {
            Some(bytes) => {
                out.append_value(unsafe { std::str::from_utf8_unchecked(&bytes) });
                last_output = Some(bytes);
            }
            None => {
                out.append_value(s);
                last_output = None;
            }
        }
    }
    Arc::new(out.finish()) as ArrayRef
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    fn urls(testing_planner: &mut TestingPlanner, values: Vec<&'static str>) {
        testing_planner.add_table(
            "urls",
            &[(
                "url",
                Type::Utf8,
                Arc::new(StringViewArray::from(values)) as ArrayRef,
            )],
        );
    }

    #[rstest]
    fn extracts_domain_group(mut testing_planner: TestingPlanner) {
        urls(
            &mut testing_planner,
            vec![
                "http://www.example.com/page",
                "https://foo.org/x",
                "no-url-here",
            ],
        );

        let mut rows = run(
            &mut testing_planner,
            r"SELECT regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') FROM urls",
        );

        let mut got = rows
            .iter_mut()
            .map(|r| only_column(r).as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        got.sort();
        assert_eq!(got, vec!["example.com", "foo.org", "no-url-here"]);
    }

    #[rstest]
    fn replaces_first_match_only(mut testing_planner: TestingPlanner) {
        urls(&mut testing_planner, vec!["banana"]);

        let rows = run(
            &mut testing_planner,
            "SELECT regexp_replace(url, 'a', 'X') FROM urls",
        );

        assert_eq!(*only_column(&rows[0]), "bXnana");
    }

    #[rstest]
    fn optional_group_absent_expands_empty(mut testing_planner: TestingPlanner) {
        urls(&mut testing_planner, vec!["ac", "abc"]);

        let mut rows = run(
            &mut testing_planner,
            r"SELECT regexp_replace(url, 'a(b)?c', 'X\1Y') FROM urls",
        );

        let got = rows
            .iter_mut()
            .map(|r| only_column(r).as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        // The optional group is absent for "ac" (so `\1` expands to nothing) and
        // captures "b" for "abc".
        assert_eq!(got, vec!["XY", "XbY"]);
    }

    #[rstest]
    fn whole_match_reference(mut testing_planner: TestingPlanner) {
        urls(&mut testing_planner, vec!["abc"]);

        let rows = run(
            &mut testing_planner,
            r"SELECT regexp_replace(url, 'b', '[\&]') FROM urls",
        );

        // PostgreSQL's `\&` inserts the whole match, so "b" becomes "[b]".
        assert_eq!(*only_column(&rows[0]), "a[b]c");
    }
}
