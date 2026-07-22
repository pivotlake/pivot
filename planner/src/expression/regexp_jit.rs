//! [`RegexpJitReplace`] — SQL `regexp_jit_replace(input, pattern, replacement)`,
//! first-match replacement evaluated with PCRE2 JIT-compiled to native code.
//!
//! Unlike [`RegexpReplace`](super::regexp::RegexpReplace) (which picks an engine
//! per pattern and can fall back to the `regex` crate), this always requires
//! PCRE2's JIT: a pattern PCRE2 can't JIT-compile (or when JIT is unavailable)
//! fails the plan rather than falling back to a slower engine. Matching runs in
//! *byte* mode, so it handles columns that aren't valid UTF-8. The
//! replacement-template parsing, the splice that builds a replaced row, and the
//! per-column driver are shared with the `regex`-crate variant.

use super::Expression;
use super::regexp::replace_column;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use pcre2::bytes::{CaptureLocations, Regex as Pcre2Regex, RegexBuilder as Pcre2RegexBuilder};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `regexp_jit_replace(input, pattern, replacement)` — like
/// [`RegexpReplace`](super::regexp::RegexpReplace) but always backed by PCRE2
/// JIT. `pattern` and `replacement` must be constants so the regex can be
/// compiled once at plan-compile time.
#[derive(Debug, Clone)]
pub struct RegexpJitReplace {
    pub input: Box<Expression>,
    pub pattern: String,
    pub replacement: String,
}

impl Display for RegexpJitReplace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "regexp_jit_replace({}, '{}', '{}')",
            self.input, self.pattern, self.replacement
        )
    }
}

impl RegexpJitReplace {
    pub fn compile(&self, parameters: &compile::BoundParameters) -> Result<ExprFn, compile::Error> {
        // Require JIT compilation up front: a pattern PCRE2 can't JIT-compile
        // fails the plan rather than running on a slower engine.
        build_jit_regex(&self.pattern).map_err(|source| {
            compile::Error::InvalidJitRegexPattern {
                pattern: self.pattern.clone(),
                source,
            }
        })?;

        let pattern = self.pattern.clone();
        let template = Arc::new(parse_replacement(&self.replacement));
        let input_builder = self.input.compile(parameters)?;

        Ok(Box::new(move || {
            let template = template.clone();
            // A fresh regex per worker avoids shared-state contention in the row loop.
            let regex = build_jit_regex(&pattern).expect("pattern validated at plan compile");
            // One reusable capture buffer per worker, refilled by every match so
            // the row loop allocates no per-row PCRE2 match data.
            let mut locations = regex.capture_locations();
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                let result = replace_column(strings, |s| {
                    replace_first(&regex, &mut locations, s, &template)
                });
                ExprResult::Array(result)
            }) as ExprEvalFn
        }))
    }
}

/// Build a byte-mode PCRE2 regex JIT-compiled to native code. `.jit(true)`
/// errors when the pattern can't be JIT-compiled or JIT is unavailable.
fn build_jit_regex(pattern: &str) -> Result<Pcre2Regex, pcre2::Error> {
    Pcre2RegexBuilder::new().jit(true).build(pattern)
}

/// Replace the first match of `regex` in `s`, applying `template`. The match
/// spans are read into `locations` (reused across calls), and group bytes are
/// sliced straight out of `s`. Returns `None` when nothing matched, so the
/// caller passes `s` through unchanged.
fn replace_first(
    regex: &Pcre2Regex,
    locations: &mut CaptureLocations,
    s: &[u8],
    template: &[ReplacementSegment],
) -> Option<Vec<u8>> {
    regex.captures_read(locations, s).ok().flatten()?;
    let (start, end) = locations.get(0).unwrap();
    Some(splice(s, (start, end), template, |n| {
        locations
            .get(n)
            .map(|(group_start, group_end)| &s[group_start..group_end])
    }))
}

/// A parsed replacement template: a sequence of literal byte runs and `\N`
/// group references. Parsed once (PostgreSQL-style `\N`/`\\`/`\$`). PCRE2 has
/// no native replace, so this and [`splice`] expand the template by hand
/// (unlike [`RegexpReplace`](super::regexp::RegexpReplace), which lets the
/// `regex` crate do it).
enum ReplacementSegment {
    Literal(Vec<u8>),
    Group(usize),
}

fn parse_replacement(replacement: &str) -> Vec<ReplacementSegment> {
    let mut segments = Vec::new();
    let mut literal: Vec<u8> = Vec::new();
    let push_char = |buf: &mut Vec<u8>, c: char| {
        let mut tmp = [0u8; 4];
        buf.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
    };
    let mut chars = replacement.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                // `\N` is a group reference (`\0` and PostgreSQL's `\&` both
                // name the whole match).
                Some(d @ '0'..='9') => {
                    if !literal.is_empty() {
                        segments.push(ReplacementSegment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(ReplacementSegment::Group(d as usize - '0' as usize));
                }
                Some('&') => {
                    if !literal.is_empty() {
                        segments.push(ReplacementSegment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(ReplacementSegment::Group(0));
                }
                Some('\\') => literal.push(b'\\'),
                // An escaped dollar is a literal `$` ($ is never special here).
                Some('$') => literal.push(b'$'),
                Some(other) => {
                    literal.push(b'\\');
                    push_char(&mut literal, other);
                }
                None => literal.push(b'\\'),
            },
            other => push_char(&mut literal, other),
        }
    }
    if !literal.is_empty() {
        segments.push(ReplacementSegment::Literal(literal));
    }
    segments
}

/// Build the output for one match: text before the match, the expanded
/// `template`, then text after. `group(n)` returns the bytes captured by group
/// `n` (group 0 is the whole match), or `None` if it didn't participate.
fn splice<'a>(
    input: &[u8],
    span: (usize, usize),
    template: &[ReplacementSegment],
    group: impl Fn(usize) -> Option<&'a [u8]>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    out.extend_from_slice(&input[..span.0]);
    for segment in template {
        match segment {
            ReplacementSegment::Literal(bytes) => out.extend_from_slice(bytes),
            ReplacementSegment::Group(n) => {
                if let Some(bytes) = group(*n) {
                    out.extend_from_slice(bytes);
                }
            }
        }
    }
    out.extend_from_slice(&input[span.1..]);
    out
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
            r"SELECT regexp_jit_replace(url, '^https?://(?:www\.)?([^/]+)/.*', '\1') FROM urls",
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
            "SELECT regexp_jit_replace(url, 'a', 'X') FROM urls",
        );

        assert_eq!(*only_column(&rows[0]), "bXnana");
    }

    #[rstest]
    fn optional_group_absent_expands_empty(mut testing_planner: TestingPlanner) {
        urls(&mut testing_planner, vec!["ac", "abc"]);

        let mut rows = run(
            &mut testing_planner,
            r"SELECT regexp_jit_replace(url, 'a(b)?c', 'X\1Y') FROM urls",
        );

        let got = rows
            .iter_mut()
            .map(|r| only_column(r).as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        // The optional group is absent for "ac" (so `\1` expands to nothing) and
        // captures "b" for "abc"; both rows share the reused capture buffer.
        assert_eq!(got, vec!["XY", "XbY"]);
    }
}
