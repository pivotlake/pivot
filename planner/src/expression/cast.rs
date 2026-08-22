//! [`Cast`] — `CAST(source AS target)`.
//!
//! DuckDB inserts casts to coerce operands to a common type and for explicit
//! user casts; pivot honors them by casting the source to the target type's
//! arrow [`DataType`] at runtime. A temporal target casts to its real arrow type
//! (`Date32`/`Timestamp`), every other to its physical storage type.
//!
//! Text into a `VARIANT` is handled here rather than by Arrow's kernel and
//! parses each string as a JSON document. Variant-to-scalar casts are typed
//! path reads built as a [`VariantGet`](super::VariantGet). A synthetic cast
//! used only at the query boundary renders an uncast variant output as JSON.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow_array::builder::StringViewBuilder;
use arrow_array::{Array, ArrayRef, RecordBatch, Scalar, StructArray};
use arrow_schema::{ArrowError, DataType};
use parquet_variant_compute::{
    GetOptions, VariantArray, json_to_variant, unshred_variant, variant_get,
};
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
    /// The expression being converted.
    pub fn source(&self) -> &Expression {
        &self.source
    }

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
                let input = source_expr(batch).into_array(batch.num_rows());
                let out = variant_to_json_array(&input, &DataType::Utf8View)
                    .unwrap_or_else(|e| panic!("variant cast failed: {e}"));
                ExprResult::Array(out)
            }) as ExprEvalFn
        }))
    }
}

/// Cast a variant array with Pivot's SQL semantics.
///
/// The Parquet variant kernel's strict mode preserves JSON null as Arrow null
/// and rejects every other invalid conversion. Text uses extraction semantics:
/// strings lose their JSON quotes, while other values use their JSON
/// representation.
pub fn cast_variant_array(input: &ArrayRef, target: &DataType) -> Result<ArrayRef, ArrowError> {
    if !variant_cast_maps_json_null_to_sql_null(target) {
        return variant_to_text_array(input, target);
    }

    let target_field = Arc::new(arrow_schema::Field::new("item", target.clone(), true));
    variant_get(
        input,
        GetOptions::new()
            .with_as_type(Some(target_field))
            .with_cast_options(arrow::compute::CastOptions {
                safe: false,
                ..Default::default()
            }),
    )
    .map_err(|error| match error {
        ArrowError::CastError(message) => {
            ArrowError::CastError(format!("cannot cast variant value to {target}: {message}"))
        }
        error => error,
    })
}

/// Whether JSON null and an absent value have the same result under this SQL
/// cast. Text is the exception: it renders JSON null as `"null"` while an
/// absent value remains SQL NULL.
pub fn variant_cast_maps_json_null_to_sql_null(target: &DataType) -> bool {
    !matches!(
        target,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Render a variant as extraction text: JSON strings become their unquoted
/// contents; every other value uses its JSON representation.
fn variant_to_text_array(input: &ArrayRef, target: &DataType) -> Result<ArrayRef, ArrowError> {
    render_variant_text(input, target, true)
}

/// Render any physical variant layout as JSON text, then restamp it to the
/// requested Arrow string representation.
fn variant_to_json_array(input: &ArrayRef, target: &DataType) -> Result<ArrayRef, ArrowError> {
    render_variant_text(input, target, false)
}

fn render_variant_text(
    input: &ArrayRef,
    target: &DataType,
    unquote_strings: bool,
) -> Result<ArrayRef, ArrowError> {
    let variant = VariantArray::try_new(input)?;
    let variant = unshred_variant(&variant)?;
    let mut json = StringViewBuilder::with_capacity(variant.len());
    let mut text = Vec::new();
    for row in 0..variant.len() {
        if variant.is_null(row) {
            json.append_null();
            continue;
        }
        let value = variant.try_value(row)?;
        if unquote_strings && let Some(value) = value.as_string() {
            json.append_value(value);
            continue;
        }
        text.clear();
        value.to_json(&mut text).map_err(|e| {
            ArrowError::ComputeError(format!("variant value failed to render as JSON: {e}"))
        })?;
        json.append_value(
            std::str::from_utf8(&text)
                .map_err(|e| ArrowError::ComputeError(format!("variant JSON is not UTF-8: {e}")))?,
        );
    }
    let json: ArrayRef = Arc::new(json.finish());
    if target == &DataType::Utf8View {
        return Ok(json);
    }
    arrow::compute::cast_with_options(
        &json,
        target,
        &arrow::compute::CastOptions {
            safe: false,
            ..Default::default()
        },
    )
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
