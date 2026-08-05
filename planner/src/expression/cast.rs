//! [`Cast`] — `CAST(source AS target)`.
//!
//! DuckDB inserts casts to coerce operands to a common type and for explicit
//! user casts; pivot honors them by casting the source to the target type's
//! arrow [`DataType`] at runtime. A temporal target casts to its real arrow type
//! (`Date32`/`Timestamp`), every other to its physical storage type.
//!
//! Two casts cross the variant boundary and are handled here rather than by
//! arrow's kernel: text into a `VARIANT` parses each string as a JSON document,
//! and a `VARIANT` into text renders each document back to JSON. (A cast of a
//! `VARIANT` to a *non-text* type is a typed path read, built as a
//! [`VariantGet`](super::VariantGet), so it never reaches here.)

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow_array::builder::StringViewBuilder;
use arrow_array::{Array, ArrayRef, RecordBatch, Scalar, StructArray};
use arrow_schema::{ArrowError, DataType};
use parquet_variant_compute::{VariantArray, json_to_variant, unshred_variant};
use parquet_variant_json::VariantToJson as ToJson;
use std::fmt::{self, Display};
use std::sync::Arc;

/// A `CAST(source AS target)`. `target` is the pivot type (used for the result
/// type when grouping/typing); `target_arrow` is the arrow type the value is
/// cast into.
#[derive(Debug, Clone)]
pub struct Cast {
    pub target: Type,
    pub(crate) target_arrow: DataType,
    pub(crate) source: Box<Expression>,
}

impl Display for Cast {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cast({} as {})", self.source, self.target)
    }
}

impl Cast {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // Casting text to a variant parses each document as JSON.
        if self.target == Type::Variant {
            return self.compile_json_to_variant();
        }
        // Casting a variant to text renders each document back to JSON. A
        // variant source only reaches a `Cast` for a text target (any other
        // target is a typed read built as a `VariantGet`), so the target check
        // is not repeated.
        if matches!(self.source.result_type(), Ok(Type::Variant)) {
            return self.compile_variant_to_json();
        }

        let target = self.target_arrow.clone();
        // Cast strictly: a value the target type cannot represent must fail
        // the query, not silently become NULL (the lenient TRY_CAST form is
        // rejected while building the plan).
        let options = arrow::compute::CastOptions {
            safe: false,
            ..Default::default()
        };
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let target = target.clone();
            let options = options.clone();
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, is_scalar) = src.as_datum().get();
                // The failure is data-dependent and a compiled expression has
                // no error channel, so it panics and the dataflow fails the
                // query with the kernel's message naming the offending value.
                let out = arrow::compute::cast_with_options(arr, &target, &options)
                    .unwrap_or_else(|e| panic!("{e}"));
                if is_scalar {
                    ExprResult::Scalar(Scalar::new(out))
                } else {
                    ExprResult::Array(out)
                }
            }) as ExprEvalFn
        }))
    }

    /// Text into a variant: parse each string as a JSON document. A compiled
    /// expression has no error channel and a malformed document is
    /// data-dependent, so a bad value panics and the dataflow fails the query
    /// (constant folding parses through [`json_to_canonical_variant`] directly
    /// and reports the error instead).
    fn compile_json_to_variant(&self) -> Result<ExprFn, compile::Error> {
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = source_expr(batch).into_array(batch.num_rows());
                let out = json_to_canonical_variant(&input)
                    .unwrap_or_else(|e| panic!("value is not a JSON document: {e}"));
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
    }

    /// A variant into text: render each document as JSON. Used to serialize a
    /// variant-typed output column for the client.
    fn compile_variant_to_json(&self) -> Result<ExprFn, compile::Error> {
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = source_expr(batch);
                let (arr, _) = input.as_datum().get();
                // `VariantArray` reads any physical layout, shredded or not,
                // so batches from differently-shredded files render the same.
                let variant =
                    VariantArray::try_new(arr).expect("a variant-typed input is a variant struct");
                // Row-wise rendering can't reassemble a shredded OBJECT from
                // its typed leaves (`value()` only handles typed scalars), so
                // fold the typed leaves back into the binary value column
                // first. A no-op for unshredded input.
                let variant = unshred_variant(&variant)
                    .expect("shredded variant folds back to its binary form");

                let mut json = StringViewBuilder::with_capacity(variant.len());
                // One text buffer reused across rows, instead of a fresh
                // `String` per rendered value.
                let mut text = Vec::new();
                for row in 0..variant.len() {
                    if variant.is_null(row) {
                        json.append_null();
                        continue;
                    }
                    // Both failures are data-dependent (a corrupt metadata or
                    // value blob); compiled expressions have no error channel,
                    // so the dataflow catches the panic and fails the query.
                    let value = variant
                        .try_value(row)
                        .unwrap_or_else(|e| panic!("corrupt variant value at row {row}: {e}"));
                    text.clear();
                    value
                        .to_json(&mut text)
                        .unwrap_or_else(|e| panic!("variant value failed to render as JSON: {e}"));
                    json.append_value(std::str::from_utf8(&text).expect("JSON output is UTF-8"));
                }
                ExprResult::Array(Arc::new(json.finish()))
            }) as ExprEvalFn
        }))
    }
}

/// Parse an array of JSON-document strings into a variant of the canonical
/// physical type. `json_to_variant` marks the `value` field non-null (no row's
/// value is), but a `VARIANT` column's physical type has it nullable; restamp
/// the canonical fields so the result is exactly what a column declares and an
/// INSERT accepts. Shared by the text-to-variant cast and by constant folding
/// of a variant literal.
pub(crate) fn json_to_canonical_variant(text: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    let variants = json_to_variant(text)?;
    let structure = variants.into_inner();
    let DataType::Struct(fields) = physical_arrow_type(&Type::Variant) else {
        unreachable!("a variant's physical type is a struct")
    };
    Ok(Arc::new(StructArray::new(
        fields,
        structure.columns().to_vec(),
        structure.nulls().cloned(),
    )))
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, StringArray};
    use rstest::rstest;
    use std::sync::Arc;

    /// A value the target type cannot represent fails the query with an error
    /// naming it, rather than quietly becoming NULL. Not temporal-specific:
    /// any target type rejects a value it cannot represent.
    #[rstest]
    #[case("TIMESTAMP", "not-a-timestamp")]
    #[case("INTEGER", "twelve")]
    fn unconvertible_cast_fails_the_query(
        mut testing_planner: TestingPlanner,
        #[case] target: &str,
        #[case] value: &str,
    ) {
        testing_planner.add_table(
            "events",
            &[(
                "raw",
                Type::Utf8,
                Arc::new(StringArray::from(vec![value])) as ArrayRef,
            )],
        );

        let err = run_expecting_error(
            &mut testing_planner,
            &format!("SELECT CAST(raw AS {target}) FROM events"),
        );

        assert!(
            err.contains(value),
            "the error names the offending value; got: {err}"
        );
    }

    /// TRY_CAST's null-on-failure contract is not implemented, so it is
    /// rejected while building the plan rather than quietly run strict.
    #[rstest]
    fn rejects_try_cast(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "raw",
                Type::Utf8,
                Arc::new(StringArray::from(vec!["twelve"])) as ArrayRef,
            )],
        );

        let err = testing_planner
            .plan("SELECT TRY_CAST(raw AS INTEGER) FROM events")
            .unwrap_err();

        assert!(
            err.to_string().contains("TRY_CAST is not supported"),
            "got: {err}"
        );
    }
}
