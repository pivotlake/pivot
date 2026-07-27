//! [`RegexpFullMatch`]: SQL `regexp_full_match(input, pattern)`, the predicate
//! DuckDB's parser lowers `input ~ 'pattern'`, `input !~ 'pattern'` (negated
//! with a `NOT` above) and `SIMILAR TO` into.
//!
//! Matching is Unicode-mode, so `.` spans a codepoint, matching RE2's UTF-8
//! default. Two divergences from RE2 remain, both inherent to evaluating with
//! the `regex` crate: the Perl classes `\d`, `\w`, `\s` and `\b` are
//! Unicode-aware here but ASCII-only in RE2, and a few patterns RE2 accepts as
//! literals (`a{`) are rejected outright, failing the query rather than
//! answering it differently.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BooleanArray, RecordBatch, Scalar};
use regex::Regex;
use std::fmt::{self, Display};
use std::sync::Arc;

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
        // Anchoring is what makes this a *full* match: testing a match's span
        // instead would accept `a|ab` against "abc", where the leftmost
        // alternative matches just "a".
        let anchored = format!(r"\A(?:{})\z", self.pattern);
        Regex::new(&anchored).map_err(|source| compile::Error::InvalidRegexPattern {
            // Report the pattern the query wrote, not the anchored form.
            pattern: self.pattern.clone(),
            source,
        })?;

        let input_builder = self.input.compile()?;

        Ok(Box::new(move || {
            // A fresh `Regex` per worker: a cloned `Regex` funnels every match
            // through a shared pool guarded by atomics, contending in the row loop.
            let regex = Regex::new(&anchored).expect("pattern validated at plan compile");
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, is_scalar) = input.as_datum().get();
                let strings = arr.as_string_view();
                // Values arrive in runs (sessions repeat the same URL), so an
                // equal-to-previous input reuses the prior row's verdict instead
                // of matching again. A null row stays null, which a `WHERE` reads
                // as false and a projection surfaces as NULL.
                let mut last_input: Option<(&str, bool)> = None;
                let matches: BooleanArray = strings
                    .iter()
                    .map(|value| {
                        value.map(|s| match last_input {
                            Some((previous, matched)) if previous == s => matched,
                            _ => {
                                let matched = regex.is_match(s);
                                last_input = Some((s, matched));
                                matched
                            }
                        })
                    })
                    .collect();
                let matches = Arc::new(matches) as ArrayRef;
                // A scalar input yields a one-row answer that stands for the
                // whole batch, so it has to stay a scalar rather than become a
                // one-element mask.
                match is_scalar {
                    true => ExprResult::Scalar(Scalar::new(matches)),
                    false => ExprResult::Array(matches),
                }
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    fn add_urls_table(testing_planner: &mut TestingPlanner) {
        testing_planner.add_table(
            "urls",
            &[(
                "url",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "https://example.com/page",
                    "https://foo.org/x",
                    "no-url-here",
                ])) as ArrayRef,
            )],
        );
    }

    #[rstest]
    fn tilde_matches_the_whole_value(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT url FROM urls WHERE url ~ '.*example.*'",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["url"], "https://example.com/page");
    }

    #[rstest]
    fn null_pattern_is_rejected(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let result = testing_planner.plan("SELECT url FROM urls WHERE url ~ NULL");

        // DuckDB binds this and leaves the NULL in the plan, so the pattern
        // reaches pivot; it must never compile as the literal text "NULL",
        // which would match rows holding that string.
        let error = result
            .expect_err("a NULL pattern must not plan")
            .to_string();
        assert!(error.contains("NULL"), "{error}");
    }

    #[rstest]
    fn tilde_rejects_a_partial_match(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT url FROM urls WHERE url ~ 'example'",
        );

        assert!(rows.is_empty());
    }

    #[rstest]
    fn not_tilde_returns_the_complement(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT url FROM urls WHERE url !~ '.*example.*'",
        );

        assert_eq!(rows.len(), 2);
    }

    #[rstest]
    fn matches_over_an_aggregate_result(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        // A filter DuckDB leaves above the scan is where its regex_range pass
        // adds the BLOB range bounds pivot can't decode.
        let rows = run(
            &mut testing_planner,
            r"SELECT m FROM (SELECT max(url) AS m FROM urls) WHERE m ~ '^no-.*'",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["m"], "no-url-here");
    }

    #[rstest]
    fn similar_to_uses_the_same_match(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT url FROM urls WHERE url SIMILAR TO '.*foo.*'",
        );

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["url"], "https://foo.org/x");
    }

    #[rstest]
    fn dot_spans_a_codepoint_not_a_byte(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "accents",
            &[(
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec!["é"])) as ArrayRef,
            )],
        );

        let rows = run(&mut testing_planner, "SELECT s ~ '.' FROM accents");

        assert!(only_column(&rows[0]).as_bool().unwrap());
    }

    #[rstest]
    fn projects_a_boolean_per_row(mut testing_planner: TestingPlanner) {
        add_urls_table(&mut testing_planner);

        let rows = run(&mut testing_planner, "SELECT url ~ '.*example.*' FROM urls");

        let mut got = rows
            .iter()
            .map(|r| only_column(r).as_bool().unwrap())
            .collect::<Vec<_>>();
        got.sort();
        assert_eq!(got, vec![false, false, true]);
    }
}
