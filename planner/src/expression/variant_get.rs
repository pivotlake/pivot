//! Reads paths from variant columns.
//!
//! A bare `->` returns a sub-variant. A cast turns the full path into one typed
//! [`VariantGet`], which can read a shredded leaf directly.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow_array::RecordBatch;
use arrow_array::builder::StringViewBuilder;
use arrow_schema::{Field, FieldRef};
use parquet_variant::{VariantPath, VariantPathElement};
use parquet_variant_compute::{GetOptions, VariantArray, variant_get};
use parquet_variant_json::VariantToJson as ToJson;
use std::fmt::{self, Display};
use std::sync::Arc;

/// A path into a JSON document: one object field name per nesting level, so
/// `d->'user'->'id'` reads the path `["user", "id"]`. Empty means the whole
/// document.
pub type JsonPath = Vec<String>;

/// A variant path read, optionally converted to a concrete type.
#[derive(Debug, Clone)]
pub struct VariantGet {
    pub input: Box<Expression>,
    pub path: JsonPath,
    pub as_type: Option<Type>,
}

impl VariantGet {
    /// The pivot type the read yields: the cast's target for a typed read, the
    /// sub-variant otherwise.
    pub fn result_type(&self) -> Type {
        self.as_type.clone().unwrap_or(Type::Variant)
    }

    /// Returns whether a variant can be read as `t`.
    pub(crate) fn supports_cast_to(t: &Type) -> bool {
        !matches!(t, Type::Variant)
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        // Build the parsed path once; the per-batch evaluation only clones the
        // elements. Owned segments make the path `'static`.
        let vpath: VariantPath<'static> = self
            .path
            .iter()
            .map(|segment| VariantPathElement::field(segment.clone()))
            .collect();
        // A typed read asks the kernel directly for the pivot type's physical
        // arrow column.
        let as_field: Option<FieldRef> = self
            .as_type
            .as_ref()
            .map(|ty| Arc::new(Field::new("item", physical_arrow_type(ty), true)));
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let vpath = vpath.clone();
            let as_field = as_field.clone();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                let variant = arr.slice(0, arr.len());

                let options =
                    GetOptions::new_with_path(vpath.clone()).with_as_type(as_field.clone());
                let array = variant_get(&variant, options).expect("variant path extraction");
                ExprResult::Array(array)
            }) as ExprEvalFn
        }))
    }
}

impl Display for VariantGet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", self.input)?;
        for segment in &self.path {
            write!(f, "->'{segment}'")?;
        }
        write!(f, ")")?;
        if let Some(ty) = &self.as_type {
            write!(f, "::{ty}")?;
        }
        Ok(())
    }
}

/// Converts variant output to JSON text.
#[derive(Debug, Clone)]
pub struct VariantToJson {
    pub input: Box<Expression>,
}

impl VariantToJson {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (arr, _) = input.as_datum().get();
                // `VariantArray` reads any physical layout, shredded or not,
                // so batches from differently-shredded files render the same.
                let variant =
                    VariantArray::try_new(arr).expect("a variant-typed input is a variant struct");

                let mut json = StringViewBuilder::with_capacity(variant.len());
                for row in 0..variant.len() {
                    if variant.is_null(row) {
                        json.append_null();
                    } else {
                        let text = variant
                            .value(row)
                            .to_json_string()
                            .expect("a variant value renders as JSON");
                        json.append_value(text);
                    }
                }
                ExprResult::Array(Arc::new(json.finish()))
            }) as ExprEvalFn
        }))
    }
}

impl Display for VariantToJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "to_json({})", self.input)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Date32Type;
    use arrow_array::{ArrayRef, Date32Array, StringArray};
    use arrow_schema::DataType;
    use parquet_variant_compute::{cast_to_variant, json_to_variant};
    use rstest::rstest;
    use std::sync::Arc;

    fn docs_table(planner: &mut TestingPlanner, rows: Vec<&str>) {
        let json: ArrayRef = Arc::new(StringArray::from(rows));
        let docs = json_to_variant(&json).unwrap().into_inner();
        planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);
    }

    /// A cast over an extraction returns the value as a real typed column.
    #[rstest]
    fn casts_an_extracted_path(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"age":30}"#, r#"{"age":25}"#]);

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs ORDER BY a",
        );

        let got: Vec<i64> = rows
            .iter()
            .map(|r| only_column(r).as_i64().unwrap())
            .collect();
        assert_eq!(got, vec![25, 30]);
    }

    /// `d.age` is the same extraction as `d->'age'`: DuckDB binds the dot
    /// syntax to its native `variant_extract`, which pivot intercepts.
    #[rstest]
    fn casts_a_dot_accessed_field(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"age":30}"#, r#"{"age":25}"#]);

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d.age AS BIGINT) AS a FROM docs ORDER BY a",
        );

        let got: Vec<i64> = rows
            .iter()
            .map(|r| only_column(r).as_i64().unwrap())
            .collect();
        assert_eq!(got, vec![25, 30]);
    }

    /// Dot access chains into nested objects, fusing like `->` chains do.
    #[rstest]
    fn casts_a_nested_dot_chain(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"user":{"id":7}}"#, r#"{"user":{"id":3}}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d.user.id AS BIGINT) AS i FROM docs ORDER BY i",
        );

        let got: Vec<i64> = rows
            .iter()
            .map(|r| only_column(r).as_i64().unwrap())
            .collect();
        assert_eq!(got, vec![3, 7]);
    }

    /// A bare extraction in the output renders as JSON text: numbers plain,
    /// strings quoted, a missing path as SQL NULL.
    #[rstest]
    fn bare_extraction_renders_json_text(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"age":30}"#, r#"{"age":"old"}"#, r#"{"name":"bob"}"#],
        );

        let rows = run(&mut testing_planner, "SELECT d.age AS a FROM docs");

        let mut got: Vec<Option<String>> = rows
            .iter()
            .map(|r| r["a"].as_str().map(str::to_string))
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![None, Some("\"old\"".to_string()), Some("30".to_string())]
        );
    }

    /// A bare variant column in the output renders each document as JSON text.
    #[rstest]
    fn bare_variant_column_renders_json_text(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"age":30}"#]);

        let rows = run(&mut testing_planner, "SELECT d FROM docs");

        assert_eq!(only_column(&rows[0]).as_str().unwrap(), r#"{"age":30}"#);
    }

    /// Comparing an extraction without a cast is rejected cleanly: DuckDB
    /// plans it as a variant-to-variant comparison (`variant_normalize` on
    /// both sides), which pivot doesn't execute. The supported form is an
    /// explicit cast, `CAST(d.age AS BIGINT) > 27`.
    #[rstest]
    fn rejects_an_uncast_comparison(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"age":30}"#]);

        let result = testing_planner
            .planner
            .plan("SELECT d FROM docs WHERE d.age > 27");

        assert!(
            result.is_err(),
            "an uncast variant comparison must fail to plan; got: {result:?}"
        );
    }

    /// A chain of `->` descends into nested objects; the cast fuses the whole
    /// chain into one typed read.
    #[rstest]
    fn casts_a_nested_extraction_chain(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"user":{"id":7}}"#, r#"{"user":{"id":3}}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'user'->'id' AS BIGINT) AS i FROM docs ORDER BY i",
        );

        let got: Vec<i64> = rows
            .iter()
            .map(|r| only_column(r).as_i64().unwrap())
            .collect();
        assert_eq!(got, vec![3, 7]);
    }

    /// A path absent from a row reads as NULL, not an error.
    #[rstest]
    fn missing_path_is_null(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"age":30}"#, r#"{"name":"bob"}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs",
        );

        // The JSON writer omits null fields, so the missing row has no "a" key
        // (indexing yields JSON null).
        let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
        got.sort();
        assert_eq!(got, vec![None, Some(30)]);
    }

    /// A value of the wrong type reads as NULL (like a missing path), not an
    /// error.
    #[rstest]
    fn type_mismatch_is_null(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"age":30}"#, r#"{"age":"unknown"}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs",
        );

        let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
        got.sort();
        assert_eq!(got, vec![None, Some(30)]);
    }

    /// Because the cast types the path BIGINT, the GROUP BY is built for an
    /// integer key: equal ages collapse to one group.
    #[rstest]
    fn groups_by_a_cast_path(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"age":30}"#, r#"{"age":30}"#, r#"{"age":25}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'age' AS BIGINT) AS a, count(*) AS n FROM docs GROUP BY a ORDER BY a",
        );

        let got: Vec<(i64, i64)> = rows
            .iter()
            .map(|r| (r["a"].as_i64().unwrap(), r["n"].as_i64().unwrap()))
            .collect();
        assert_eq!(got, vec![(25, 1), (30, 2)]);
    }

    /// An aggregate over a cast extraction keeps the cast (it is the read's
    /// type, not a numeric widening to strip).
    #[rstest]
    fn sums_a_cast_path(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"age":30}"#, r#"{"age":25}"#]);

        let rows = run(
            &mut testing_planner,
            "SELECT SUM(CAST(d->'age' AS BIGINT)) AS s FROM docs",
        );

        assert_eq!(only_column(&rows[0]).as_i64().unwrap(), 55);
    }

    /// Casting to VARCHAR yields pivot's string columns (`Utf8View`), the bare
    /// value rather than JSON text (no quotes).
    #[rstest]
    fn casts_a_path_to_string(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"name":"bob"}"#]);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT CAST(d->'name' AS VARCHAR) AS n FROM docs",
        );

        let col = batches[0].column(0);
        assert_eq!(col.data_type(), &DataType::Utf8View);
        assert_eq!(col.as_string_view().value(0), "bob");
    }

    /// A cast extraction composes with a WHERE comparison; rows whose path is
    /// missing (NULL) don't match.
    #[rstest]
    fn filters_on_a_cast_path(mut testing_planner: TestingPlanner) {
        docs_table(
            &mut testing_planner,
            vec![r#"{"age":30}"#, r#"{"age":25}"#, r#"{"name":"bob"}"#],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs \
             WHERE CAST(d->'age' AS BIGINT) > 27",
        );

        let got: Vec<i64> = rows
            .iter()
            .map(|r| only_column(r).as_i64().unwrap())
            .collect();
        assert_eq!(got, vec![30]);
    }

    /// A temporal target is a typed read too: date-valued variants (as a
    /// shredded file stores them) surface as a real `Date32` column.
    #[rstest]
    fn casts_to_a_date(mut testing_planner: TestingPlanner) {
        let days: ArrayRef = Arc::new(Date32Array::from(vec![19723, 19724]));
        let docs = cast_to_variant(&days).unwrap().into_inner();
        testing_planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT CAST(d AS DATE) AS day FROM docs",
        );

        let col = batches[0].column(0);
        assert_eq!(col.data_type(), &DataType::Date32);
        let mut got: Vec<i32> = col.as_primitive::<Date32Type>().values().to_vec();
        got.sort();
        assert_eq!(got, vec![19723, 19724]);
    }

    /// A value whose variant type doesn't match a temporal target reads as
    /// NULL: a JSON document stores `"2024-01-05"` as a string, not a date.
    #[rstest]
    fn date_cast_of_a_string_value_is_null(mut testing_planner: TestingPlanner) {
        docs_table(&mut testing_planner, vec![r#"{"day":"2024-01-05"}"#]);

        let rows = run(
            &mut testing_planner,
            "SELECT CAST(d->'day' AS DATE) AS day FROM docs",
        );

        assert!(rows[0]["day"].is_null(), "got: {:?}", rows[0]["day"]);
    }

    /// The extracted field name must be a constant string.
    #[rstest]
    fn rejects_a_non_constant_field(mut testing_planner: TestingPlanner) {
        let json: ArrayRef = Arc::new(StringArray::from(vec![r#"{"age":30}"#]));
        let docs = json_to_variant(&json).unwrap().into_inner();
        let keys: ArrayRef = Arc::new(StringArray::from(vec!["age"]));
        testing_planner.add_table(
            "docs",
            &[
                ("d", Type::Variant, Arc::new(docs) as ArrayRef),
                ("k", Type::Utf8, keys),
            ],
        );

        let err = testing_planner
            .planner
            .plan("SELECT CAST(d->k AS BIGINT) FROM docs")
            .unwrap_err();

        assert!(
            err.to_string().contains("constant"),
            "expected a constant-field error, got: {err}"
        );
    }
}
