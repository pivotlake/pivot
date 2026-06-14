//! [`RegexpReplace`] — SQL `regexp_replace(input, pattern, replacement)`,
//! first-match replacement with a per-worker memoised replacer.

use super::{Error, Expression, constant_string};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use duckdb_planner::expression as duckdb_expression;
use regex::Regex;
use std::fmt::{self, Display};
use std::sync::Arc;

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

impl TryFrom<duckdb_expression::Function> for RegexpReplace {
    type Error = Error;
    fn try_from(mut f: duckdb_expression::Function) -> Result<Self, Self::Error> {
        // A fourth `options` argument (e.g. 'g' for replace-all) changes the
        // semantics, so only the three-argument first-match form is accepted.
        if f.params.len() != 3 {
            let actual = f.params.len();
            return Err(Error::InvalidParameterCount {
                function: f.function,
                expected: 3,
                actual,
            });
        }
        let replacement = constant_string(
            Expression::try_from(f.params.remove(2))?,
            "regexp_replace: replacement",
        )?;
        let pattern = constant_string(
            Expression::try_from(f.params.remove(1))?,
            "regexp_replace: pattern",
        )?;
        let input = Box::new(Expression::try_from(f.params.remove(0))?);
        Ok(RegexpReplace {
            input,
            pattern,
            replacement,
        })
    }
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

/// Translate a PostgreSQL-style `regexp_replace` replacement string (`\N`
/// group references, `\\` literal backslash, `$` literal) into the `regex`
/// crate's form (`${N}` group references, `$$` literal dollar). Group
/// references are emitted braced so a digit following the reference is not
/// absorbed into the group number.
fn translate_replacement(replacement: &str) -> String {
    let mut out = String::with_capacity(replacement.len());
    let mut chars = replacement.chars();
    while let Some(c) = chars.next() {
        match c {
            '$' => out.push_str("$$"),
            '\\' => match chars.next() {
                Some(d @ '0'..='9') => {
                    out.push_str("${");
                    out.push(d);
                    out.push('}');
                }
                Some('\\') => out.push('\\'),
                // An escaped dollar is a literal `$`, which is special in the
                // regex crate's replacement dialect — emit its `$$` escape so
                // it isn't mis-read as a (here empty, thus deleted) group ref.
                Some('$') => out.push_str("$$"),
                // Any other escape is not meaningful in either dialect; keep
                // the pair as literal text.
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            },
            c => out.push(c),
        }
    }
    out
}

/// Memoised results cap for [`MemoizedReplacer`], per worker: stop inserting
/// past this many entries or this many memoised string bytes (lookups
/// continue), so a high-cardinality column can't grow the memo unboundedly.
/// Sized small — value distributions are Zipfian, so the first tens of
/// thousands of entries carry most of the hit rate, while the map itself
/// (entries plus boxed strings) costs a few times the raw bytes.
const REGEX_MEMO_MAX_ENTRIES: usize = 1 << 19;
const REGEX_MEMO_MAX_BYTES: usize = 48 << 20;

/// Per-worker first-match regex replacement with a bounded result memo.
///
/// Replacement is pure and expensive (capture extraction runs a backtracking
/// engine), while real inputs (URLs, paths) repeat heavily, so each result is
/// memoised. A memo value of `None` means "no match — the input passes through
/// unchanged", so unmatched rows don't store their text a second time. The
/// `regex` lives per worker rather than shared: a cloned `Regex` funnels every
/// thread through one internal cache pool whose atomics dominate the row loop
/// under contention, whereas a per-worker instance gets the pool's owner fast
/// path.
struct MemoizedReplacer {
    regex: Regex,
    replacement: String,
    memo: std::collections::HashMap<Box<str>, Option<Box<str>>, ahash::RandomState>,
    memo_bytes: usize,
}

impl MemoizedReplacer {
    /// Replace the first match of the pattern in `s` (DuckDB semantics without
    /// the `'g'` option). Returns `None` when nothing matched — the caller
    /// appends `s` itself, so the passthrough text is never copied — and
    /// otherwise the replaced text, borrowed from the memo on the hot path.
    fn replace(&mut self, s: &str) -> Option<std::borrow::Cow<'_, str>> {
        use std::borrow::Cow;
        // Probe with `contains_key` first (its borrow ends immediately) so the
        // mutating insert path below doesn't overlap a live borrow — the
        // borrow checker can't yet prove a returned `get` borrow is confined
        // to the hit branch.
        if self.memo.contains_key(s) {
            return self.memo.get(s).unwrap().as_deref().map(Cow::Borrowed);
        }
        // `replace` returns `Cow::Borrowed` when nothing matched.
        let entry = match self.regex.replace(s, self.replacement.as_str()) {
            Cow::Borrowed(_) => None,
            Cow::Owned(o) => Some(o.into_boxed_str()),
        };
        if self.memo.len() < REGEX_MEMO_MAX_ENTRIES && self.memo_bytes < REGEX_MEMO_MAX_BYTES {
            self.memo_bytes += s.len() + entry.as_deref().map(str::len).unwrap_or(0);
            self.memo.insert(s.into(), entry);
            return self.memo.get(s).unwrap().as_deref().map(Cow::Borrowed);
        }
        // Past the cap: serve this result without growing the memo.
        entry.map(|o| Cow::Owned(o.into_string()))
    }
}

impl RegexpReplace {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Validate the pattern once at plan-compile time; each worker then
        // compiles its own `Regex` in the builder below.
        Regex::new(&self.pattern).map_err(|source| compile::Error::InvalidRegexPattern {
            pattern: self.pattern.clone(),
            source,
        })?;
        let pattern = self.pattern.clone();
        let replacement = translate_replacement(&self.replacement);
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let mut replacer = MemoizedReplacer {
                regex: Regex::new(&pattern).expect("pattern validated at plan compile"),
                replacement: replacement.clone(),
                memo: std::collections::HashMap::default(),
                memo_bytes: 0,
            };
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                let mut out = StringViewBuilder::with_capacity(strings.len());
                // Values arrive in runs (sessions repeat the same URL), so an
                // equal-to-previous check skips even the memo hash. `None`
                // output means "no match, pass the input through".
                let mut last_in: Option<&str> = None;
                let mut last_out: Option<String> = None;
                for v in strings.iter() {
                    let Some(s) = v else {
                        out.append_null();
                        continue;
                    };
                    if last_in == Some(s) {
                        out.append_value(last_out.as_deref().unwrap_or(s));
                        continue;
                    }
                    last_in = Some(s);
                    match replacer.replace(s) {
                        Some(replaced) => {
                            out.append_value(&replaced);
                            last_out = Some(replaced.into_owned());
                        }
                        None => {
                            out.append_value(s);
                            last_out = None;
                        }
                    }
                }
                ExprResult::Array(Arc::new(out.finish()) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use arrow_array::{ArrayRef, StringViewArray};
    use crate::types::Type;
    use rstest::rstest;
    use std::sync::Arc;

    fn urls(testing_planner: &mut TestingPlanner, values: Vec<&'static str>) {
        testing_planner.add_table(
            "urls",
            &[("url", Type::Utf8, Arc::new(StringViewArray::from(values)) as ArrayRef)],
        );
    }

    #[rstest]
    fn extracts_domain_group(mut testing_planner: TestingPlanner) {
        urls(
            &mut testing_planner,
            vec!["http://www.example.com/page", "https://foo.org/x", "no-url-here"],
        );

        let mut rows = run(
            &mut testing_planner,
            r"SELECT regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') FROM urls",
        );

        // Group reference `\1` yields the host; the un-matching row (no path)
        // passes through unchanged.
        let mut got = rows
            .iter_mut()
            .map(|r| r["col0"].as_str().unwrap().to_string())
            .collect::<Vec<_>>();
        got.sort();
        assert_eq!(got, vec!["example.com", "foo.org", "no-url-here"]);
    }

    #[rstest]
    fn replaces_first_match_only(mut testing_planner: TestingPlanner) {
        urls(&mut testing_planner, vec!["banana"]);

        let rows = run(&mut testing_planner, "SELECT regexp_replace(url, 'a', 'X') FROM urls");

        assert_eq!(rows[0]["col0"], "bXnana");
    }
}
