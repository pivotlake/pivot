//! [`Like`] — SQL `LIKE` / `NOT LIKE` against a constant pattern.
//!
//! DuckDB's optimizer rewrites the single-wildcard patterns into dedicated
//! functions before the plan reaches pivot (`'a%'` into `prefix`, `'%a%'` into
//! `contains`), so what binds to `~~` here is the general shape: several `%`
//! wildcards, or a `_`.
//!
//! A pattern of literals separated by `%` runs on [`SegmentMatcher`], which
//! searches each literal once per underlying string buffer. A `_` constrains a
//! position rather than a substring and has no such decomposition, so those
//! patterns fall back to a regex translated from the pattern once, here at
//! compile time.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow::compute::kernels::boolean::not;
use arrow_array::builder::BooleanBufferBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, RecordBatch};
use dispatch::SegmentMatcher;
use regex::bytes::{Regex as BytesRegex, RegexBuilder as BytesRegexBuilder};
use std::fmt::{self, Display};
use std::sync::Arc;

/// SQL `input LIKE pattern`, or `NOT LIKE` when `negated`.
#[derive(Debug, Clone)]
pub struct Like {
    pub input: Box<Expression>,
    pub pattern: String,
    /// `NOT LIKE`. The match runs once either way; only the mask is inverted.
    pub negated: bool,
}

impl Display for Like {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let operator = if self.negated { "NOT LIKE" } else { "LIKE" };
        write!(f, "{} {} '{}'", self.input, operator, self.pattern)
    }
}

/// A `LIKE` pattern split on its `%` wildcards: the literal the value must
/// start with, the literals that must appear in order in between, and the
/// literal it must end with. Either anchor is empty when the pattern begins or
/// ends with a `%`.
struct LikeSegments {
    prefix: String,
    segments: Vec<String>,
    suffix: String,
}

impl Like {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self.input.as_ref() {
            Expression::Ref(r) if r.return_type == Type::Utf8 => {}
            expr => {
                return Err(compile::Error::UnsupportedExpressionForLikeInput(
                    expr.clone(),
                ));
            }
        }
        let input_builder = self.input.compile()?;
        let negated = self.negated;

        match split_pattern(&self.pattern) {
            Some(pattern) => Ok(Box::new(move || {
                let mut input_expr = input_builder();
                let segments: Vec<&[u8]> = pattern.segments.iter().map(|s| s.as_bytes()).collect();
                let mut matcher = SegmentMatcher::new(
                    pattern.prefix.as_bytes(),
                    &segments,
                    pattern.suffix.as_bytes(),
                );
                Box::new(move |batch: &RecordBatch| {
                    let input = input_expr(batch);
                    let (array, _) = input.as_datum().get();
                    let column = array.as_string_view();
                    let matched = matcher.run(column);
                    // A null row is neither like nor unlike a pattern, so the
                    // input's validity carries over and survives the negation.
                    let matched =
                        BooleanArray::new(matched.values().clone(), column.nulls().cloned());
                    ExprResult::Array(match negated {
                        true => Arc::new(not(&matched).expect("negating a boolean array")),
                        false => Arc::new(matched) as ArrayRef,
                    })
                }) as ExprEvalFn
            })),
            None => {
                let regex = translate_pattern_to_regex(&self.pattern);
                // Built once here so an unusable pattern fails the query rather
                // than every worker, then again per worker below.
                build_regex(&regex).map_err(|source| compile::Error::InvalidLikePattern {
                    pattern: self.pattern.clone(),
                    source,
                })?;
                Ok(Box::new(move || {
                    let mut input_expr = input_builder();
                    let regex = build_regex(&regex).expect("the pattern compiled at plan time");
                    Box::new(move |batch: &RecordBatch| {
                        let input = input_expr(batch);
                        let (array, _) = input.as_datum().get();
                        let column = array.as_string_view();
                        let mut matched = BooleanBufferBuilder::new(column.len());
                        for value in column.iter() {
                            matched.append(
                                value.is_some_and(|value| regex.is_match(value.as_bytes())),
                            );
                        }
                        // Null rows keep their validity, as in the segment path above.
                        let matched = BooleanArray::new(matched.finish(), column.nulls().cloned());
                        ExprResult::Array(match negated {
                            true => Arc::new(not(&matched).expect("negating a boolean array")),
                            false => Arc::new(matched) as ArrayRef,
                        })
                    }) as ExprEvalFn
                }))
            }
        }
    }
}

/// Split a pattern into the literals around its `%` wildcards, or `None` for a
/// pattern [`SegmentMatcher`] can't express: one holding a `_`, or one with no
/// `%` at all (a plain equality, which DuckDB's optimizer rewrites long before
/// the pattern reaches here).
fn split_pattern(pattern: &str) -> Option<LikeSegments> {
    if pattern.contains('_') || !pattern.contains('%') {
        return None;
    }
    let parts: Vec<&str> = pattern.split('%').collect();
    let (prefix, rest) = parts.split_first().expect("split yields at least one part");
    let (suffix, middle) = rest.split_last().expect("the pattern holds a '%'");
    Some(LikeSegments {
        prefix: prefix.to_string(),
        // Consecutive wildcards leave empty parts, which constrain nothing.
        segments: middle
            .iter()
            .filter(|part| !part.is_empty())
            .map(|part| part.to_string())
            .collect(),
        suffix: suffix.to_string(),
    })
}

/// Translate a `LIKE` pattern into the equivalent anchored regex: `%` matches
/// any run of characters, `_` exactly one, and everything else is a literal.
/// Neither wildcard can be escaped, since the two-argument `LIKE` has no escape
/// character (that is the separate `like_escape` function).
fn translate_pattern_to_regex(pattern: &str) -> String {
    // `\A`/`\z` rather than `^`/`$` so the anchors can only mean the ends of the
    // whole value, and `(?s)` so `.` spans newlines: a wildcard matches any
    // character at all.
    let mut regex = String::from(r"(?s)\A");
    let mut literal = String::new();
    for character in pattern.chars() {
        match character {
            '%' | '_' => {
                regex.push_str(&regex::escape(&literal));
                literal.clear();
                regex.push_str(if character == '%' { ".*" } else { "." });
            }
            _ => literal.push(character),
        }
    }
    regex.push_str(&regex::escape(&literal));
    regex.push_str(r"\z");
    regex
}

/// Unicode matching stays on, unlike the `regexp_*` functions: `_` has to
/// consume one character rather than one byte.
fn build_regex(pattern: &str) -> Result<BytesRegex, regex::Error> {
    BytesRegexBuilder::new(pattern).unicode(true).build()
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn matches_a_multi_wildcard_pattern(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE '%a%i%e%'",
        );

        assert_eq!(rows.len(), 3);
        assert!(
            rows.iter()
                .all(|r| r["name"] == "alice" || r["name"] == "charlie")
        );
    }

    #[rstest]
    fn excludes_a_multi_wildcard_pattern(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name NOT LIKE '%a%i%e%'",
        );

        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter()
                .all(|r| r["name"] == "bob" || r["name"] == "dave")
        );
    }

    #[rstest]
    fn anchors_the_pattern_at_both_ends(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE 'a%e%e'",
        );

        assert_eq!(rows.len(), 0);
    }

    #[rstest]
    fn matches_a_single_character_wildcard(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE 'b_b'",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["name"], "bob");
    }

    #[rstest]
    fn treats_regex_metacharacters_in_the_pattern_as_literals(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT name FROM example_table WHERE name LIKE '%.%_%'",
        );

        assert_eq!(rows.len(), 0);
    }
}
