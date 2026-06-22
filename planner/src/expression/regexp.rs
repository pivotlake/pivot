//! [`RegexpReplace`] — SQL `regexp_replace(input, pattern, replacement)`,
//! first-match replacement.
//!
//! Two engines back this, chosen once per plan: PCRE2 (JIT-compiled to native
//! code) when it can evaluate the pattern with semantics identical to the
//! fallback, else the `regex` crate. Both run in *byte* mode (Unicode matching
//! off), which (a) matches RE2/ClickHouse byte semantics, (b) handles columns
//! that aren't valid UTF-8, and (c) keeps the two engines in agreement. The
//! single semantic gap — PCRE2's `$` matches before a trailing newline, the
//! `regex` crate's does not — is closed by rewriting a trailing `$` to `\z`.

use super::{Error, Expression, constant_string};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use arrow_array::builder::StringViewBuilder;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch};
use duckdb_planner::expression as duckdb_expression;
use pcre2::bytes::{Regex as Pcre2Regex, RegexBuilder as Pcre2RegexBuilder};
use regex::bytes::{Regex as BytesRegex, RegexBuilder as BytesRegexBuilder};
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

/// A parsed replacement template: a sequence of literal byte runs and `\N`
/// group references. Parsed once (PostgreSQL-style `\N`/`\\`/`\$`); applied by
/// both engines identically, so the replacement is engine-independent.
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
                // `\N` is a group reference (`\0` = the whole match).
                Some(d @ '0'..='9') => {
                    if !literal.is_empty() {
                        segments.push(ReplacementSegment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(ReplacementSegment::Group(d as usize - '0' as usize));
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

/// The two interchangeable byte-mode engines. Built per worker (each gets its
/// own instance, avoiding shared-state contention in the row loop).
enum Engine {
    Pcre2(Pcre2Regex),
    Rust(BytesRegex),
}

impl Engine {
    /// Replace the first match of the pattern in `s`, applying `template`.
    /// Returns `None` when nothing matched — the caller passes `s` through
    /// unchanged, so the passthrough text is never copied.
    fn replace_first(&self, s: &[u8], template: &[ReplacementSegment]) -> Option<Vec<u8>> {
        // Splice: text before the match + the expanded template + text after.
        macro_rules! splice {
            ($whole:expr, $group:expr) => {{
                let whole = $whole;
                let mut out = Vec::with_capacity(s.len());
                out.extend_from_slice(&s[..whole.0]);
                for segment in template {
                    match segment {
                        ReplacementSegment::Literal(bytes) => out.extend_from_slice(bytes),
                        ReplacementSegment::Group(n) => {
                            if let Some(bytes) = $group(*n) {
                                out.extend_from_slice(bytes);
                            }
                        }
                    }
                }
                out.extend_from_slice(&s[whole.1..]);
                Some(out)
            }};
        }
        match self {
            Engine::Rust(re) => {
                let caps = re.captures(s)?;
                let whole = caps.get(0).unwrap();
                splice!((whole.start(), whole.end()), |n| caps
                    .get(n)
                    .map(|m| m.as_bytes()))
            }
            Engine::Pcre2(re) => {
                let caps = re.captures(s).ok().flatten()?;
                let whole = caps.get(0).unwrap();
                splice!((whole.start(), whole.end()), |n| caps
                    .get(n)
                    .map(|m| m.as_bytes()))
            }
        }
    }
}

fn build_rust(pattern: &str) -> Result<BytesRegex, regex::Error> {
    BytesRegexBuilder::new(pattern).unicode(false).build()
}

fn build_pcre2(pattern: &str) -> Result<Pcre2Regex, pcre2::Error> {
    // `.jit(true)` errors when JIT is unavailable, so a failure here routes us
    // to the `regex`-crate fallback rather than PCRE2's (slow) interpreter.
    Pcre2RegexBuilder::new().jit(true).build(pattern)
}

/// Produce a PCRE2 pattern whose `$` semantics match the `regex` crate (and
/// RE2): in non-multiline mode the crate's `$` matches end-of-text only,
/// whereas PCRE2's also matches just before a trailing newline. We rewrite a
/// single *trailing* `$` to `\z`. Returns `None` when `$` appears anywhere
/// else (so the caller declines PCRE2 rather than risk a semantic mismatch).
fn pcre2_equivalent_pattern(pattern: &str) -> Option<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut in_class = false;
    let mut i = 0;
    let mut trailing_dollar = None;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 2; // skip the escaped char
                continue;
            }
            '[' => in_class = true,
            ']' => in_class = false,
            '$' if !in_class => {
                if i + 1 == chars.len() {
                    trailing_dollar = Some(i);
                } else {
                    return None; // `$` not at the end — don't accelerate
                }
            }
            _ => {}
        }
        i += 1;
    }
    match trailing_dollar {
        None => Some(pattern.to_string()),
        Some(idx) => {
            let mut out: String = chars[..idx].iter().collect();
            out.push_str("\\z");
            Some(out)
        }
    }
}

impl RegexpReplace {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Validate against the fallback engine (also the semantic reference).
        build_rust(&self.pattern).map_err(|source| compile::Error::InvalidRegexPattern {
            pattern: self.pattern.clone(),
            source,
        })?;

        // Use PCRE2-JIT when it can evaluate this pattern with the same
        // semantics and JIT is available; otherwise fall back.
        let pcre2_pattern =
            pcre2_equivalent_pattern(&self.pattern).filter(|p| build_pcre2(p).is_ok());
        let use_pcre2 = pcre2_pattern.is_some();
        let pcre2_pattern = pcre2_pattern.unwrap_or_default();
        let rust_pattern = self.pattern.clone();
        let template = Arc::new(parse_replacement(&self.replacement));
        let input_builder = self.input.compile()?;

        Ok(Box::new(move || {
            let template = template.clone();
            let engine = if use_pcre2 {
                match build_pcre2(&pcre2_pattern) {
                    Ok(re) => Engine::Pcre2(re),
                    Err(_) => Engine::Rust(
                        build_rust(&rust_pattern).expect("pattern validated at plan compile"),
                    ),
                }
            } else {
                Engine::Rust(build_rust(&rust_pattern).expect("pattern validated at plan compile"))
            };
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let strings = arr.as_string_view();
                let mut out = StringViewBuilder::with_capacity(strings.len());
                // Values arrive in runs (sessions repeat the same URL), so an
                // equal-to-previous check skips even the match. `None` in
                // `last_out` means "no match — pass the input through".
                let mut last_in: Option<&str> = None;
                let mut last_out: Option<Vec<u8>> = None;
                for value in strings.iter() {
                    let Some(s) = value else {
                        out.append_null();
                        continue;
                    };
                    if last_in == Some(s) {
                        match &last_out {
                            // SAFETY: bytes came from matching/splicing this
                            // column's (already `&str`) values.
                            Some(bytes) => {
                                out.append_value(unsafe { std::str::from_utf8_unchecked(bytes) })
                            }
                            None => out.append_value(s),
                        }
                        continue;
                    }
                    last_in = Some(s);
                    match engine.replace_first(s.as_bytes(), &template) {
                        Some(bytes) => {
                            out.append_value(unsafe { std::str::from_utf8_unchecked(&bytes) });
                            last_out = Some(bytes);
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
    use super::{Engine, build_pcre2, build_rust, parse_replacement, pcre2_equivalent_pattern};
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
            .map(|r| r["col0"].as_str().unwrap().to_string())
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

        assert_eq!(rows[0]["col0"], "bXnana");
    }

    /// The PCRE2 and `regex`-crate engines must produce identical output for any
    /// pattern we route to PCRE2 — this guards the byte-mode and `$`→`\z`
    /// alignment against drift, including multibyte, newline, and invalid-UTF-8
    /// inputs.
    #[test]
    fn engines_agree() {
        let pattern = r"^https?://(?:www\.)?([^/]+)/.*$";
        let rust = Engine::Rust(build_rust(pattern).unwrap());
        let pcre2 =
            Engine::Pcre2(build_pcre2(&pcre2_equivalent_pattern(pattern).unwrap()).unwrap());
        let template = parse_replacement(r"\1");

        let cases: Vec<&[u8]> = vec![
            b"https://www.example.com/path/x",
            b"http://foo.org/x",
            b"http://only-host-no-path",
            b"no-url",
            b"https://www./x",
            b"http://host/a\nb",   // embedded newline in the tail
            b"http://host/path\n", // trailing newline: the PCRE2 `$` quirk
            "http://\u{0441}\u{0430}\u{0439}\u{0442}.\u{0440}\u{0444}/\u{043f}".as_bytes(),
            &[
                104, 116, 116, 112, 58, 47, 47, 120, 46, 99, 111, 109, 47, 0xFF, 0xFE,
            ], // invalid UTF-8
        ];
        for case in cases {
            assert_eq!(
                rust.replace_first(case, &template),
                pcre2.replace_first(case, &template),
                "engines disagree on {case:?}"
            );
        }
    }

    #[test]
    fn trailing_dollar_rewritten_else_declined() {
        assert_eq!(
            pcre2_equivalent_pattern(r"^a([^/]+)/.*$").as_deref(),
            Some(r"^a([^/]+)/.*\z")
        );
        assert_eq!(pcre2_equivalent_pattern("abc").as_deref(), Some("abc"));
        // `$` not at the end → declined (caller uses the fallback engine).
        assert_eq!(pcre2_equivalent_pattern("a$b"), None);
        // Escaped `$` and `$` inside a class are literals, not anchors.
        assert_eq!(pcre2_equivalent_pattern(r"a\$").as_deref(), Some(r"a\$"));
        assert_eq!(pcre2_equivalent_pattern("[a$]").as_deref(), Some("[a$]"));
    }
}
