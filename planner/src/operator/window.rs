//! [`Window`] — window functions computed over the input's rows.
//!
//! One node holds the functions that share an `OVER` clause. It emits its
//! input's columns followed by one column per function.

use crate::compile::Error;
use crate::expression::{CompareType, Expression};
use crate::types::Type;
use arrow::compute::cast;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Datum};
use arrow_schema::DataType;
use dispatch::{OrderBy as DispatchOrderBy, RecordBatchOperatorSpec, WindowSpec};
use std::fmt;

pub use dispatch::{WindowFunction, WindowRowLimit};

/// Window functions sharing one `OVER` clause.
#[derive(Debug)]
pub struct Window {
    /// Input columns of the `PARTITION BY` keys.
    pub partition_by: Vec<usize>,
    /// The `ORDER BY` keys inside `OVER`, over input columns.
    pub order_by: Vec<DispatchOrderBy>,
    /// The functions, each producing one output column after the input's
    /// columns, in this order.
    pub functions: Vec<WindowFunction>,
    /// Set when a filter above this node, such as `QUALIFY row_number() <=
    /// 1000`, was moved into it. The node then emits only the rows for which
    /// that function is within the limit.
    pub row_limit: Option<WindowRowLimit>,
}

impl fmt::Display for Window {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let functions = self
            .functions
            .iter()
            .map(|function| match function {
                WindowFunction::Lag { column, offset }
                | WindowFunction::Lead { column, offset } => {
                    format!("{}(#{column}, {offset})", function.sql_name())
                }
                WindowFunction::RunningSum { column, .. } => format!("sum(#{column})"),
                _ => format!("{}()", function.sql_name()),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let partition_by = self
            .partition_by
            .iter()
            .map(|column| format!("#{column}"))
            .collect::<Vec<_>>()
            .join(", ");
        let order_by = self
            .order_by
            .iter()
            .map(|key| {
                let direction = if key.descending() { "DESC" } else { "ASC" };
                let nulls = if key.nulls_first() { "FIRST" } else { "LAST" };
                format!("#{} {direction} NULLS {nulls}", key.column_idx())
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "Window({functions}, partition: [{partition_by}], order: [{order_by}]"
        )?;
        if let Some(row_limit) = &self.row_limit {
            write!(
                f,
                ", keep: {}() <= {}",
                self.functions[row_limit.function_index].sql_name(),
                row_limit.limit
            )?;
        }
        f.write_str(")")
    }
}

impl Window {
    /// The types of the columns this node appends to its input's columns,
    /// given the input's column types.
    ///
    /// `lag` and `lead` return their argument's type. A running sum returns
    /// what the aggregate `sum` returns for its argument.
    pub fn function_types(&self, input_types: &[Type]) -> Result<Vec<Type>, Error> {
        self.functions
            .iter()
            .map(|function| match function {
                WindowFunction::RowNumber | WindowFunction::Rank | WindowFunction::DenseRank => {
                    Ok(Type::Int64)
                }
                WindowFunction::PercentRank => Ok(Type::Float64),
                WindowFunction::Lag { column, .. } | WindowFunction::Lead { column, .. } => {
                    Ok(input_types[*column].clone())
                }
                WindowFunction::RunningSum { column, .. } => match &input_types[*column] {
                    Type::Int8
                    | Type::Int16
                    | Type::Int32
                    | Type::Int64
                    | Type::UInt8
                    | Type::UInt16
                    | Type::UInt32
                    | Type::UInt64 => Ok(Type::Int128),
                    Type::Float32 | Type::Float64 => Ok(Type::Float64),
                    Type::Decimal { scale, .. } => Ok(Type::Decimal {
                        precision: 38,
                        scale: *scale,
                    }),
                    argument_type => {
                        Err(Error::UnsupportedWindowSumArgument(argument_type.clone()))
                    }
                },
            })
            .collect()
    }

    /// Whether each column this node appends can hold NULLs. The ranking
    /// functions always produce a value. `lag` and `lead` are NULL where the
    /// partition has no row at the offset, and a sum is NULL until its
    /// partition has a non-NULL value.
    pub fn function_nullability(&self) -> Vec<bool> {
        self.functions
            .iter()
            .map(|function| {
                matches!(
                    function,
                    WindowFunction::Lag { .. }
                        | WindowFunction::Lead { .. }
                        | WindowFunction::RunningSum { .. }
                )
            })
            .collect()
    }

    /// The value of `expression` when it is a non-NULL integer constant.
    pub(crate) fn read_integer_constant(expression: &Expression) -> Option<i64> {
        let Expression::Constant(constant) = expression else {
            return None;
        };
        let constant: &dyn Array = constant.get().0;
        if !constant.data_type().is_integer() || constant.is_null(0) {
            return None;
        }
        let constant = cast(constant, &DataType::Int64).ok()?;
        Some(constant.as_primitive::<Int64Type>().value(0))
    }

    /// Read a filter condition as "column `column_idx` is at most `limit`",
    /// returned as `(column_idx, limit)`. A condition reads that way when it
    /// is `column <= N`, `column < N` or `column = 1`, with a limit of at
    /// least 1. On a column that holds one of this operator's functions, each
    /// of those keeps the first rows of every partition, because every
    /// function starts at 1 and never decreases within a partition.
    pub(crate) fn read_row_limit(condition: &Expression) -> Option<(usize, u64)> {
        let Expression::Compare(compare) = condition else {
            return None;
        };
        let Expression::Ref(column) = compare.left.as_ref() else {
            return None;
        };
        let constant = Self::read_integer_constant(compare.right.as_ref())?;
        let limit = match compare.compare_type {
            CompareType::LessEqual => constant,
            CompareType::Less => constant.checked_sub(1)?,
            CompareType::Equal if constant == 1 => 1,
            _ => return None,
        };
        let limit = u64::try_from(limit).ok().filter(|&limit| limit >= 1)?;
        Some((column.column_idx, limit))
    }

    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        input_types: &[Type],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // A variant column's physical layout differs from batch to batch, and
        // the window operator gathers rows of many batches into one.
        if input_types.contains(&Type::Variant) {
            return Err(Error::UnsupportedWindowOverVariant);
        }
        Ok(input.window(WindowSpec {
            partition_by: self.partition_by.clone(),
            order_by: self.order_by.clone(),
            functions: self.functions.clone(),
            row_limit: self.row_limit,
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int64Array, StringViewArray};
    use rstest::rstest;
    use std::sync::Arc;

    /// A `scores(team, score)` table: team "a" scored 10, 20, 20 and 30, team
    /// "b" scored 5 and 7.
    fn add_scores_table(testing_planner: &TestingPlanner) {
        testing_planner.add_table(
            "scores",
            &[
                (
                    "team",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec!["a", "b", "a", "a", "b", "a"])) as ArrayRef,
                ),
                (
                    "score",
                    Type::Int64,
                    Arc::new(Int64Array::from(vec![30, 5, 10, 20, 7, 20])) as ArrayRef,
                ),
            ],
        );
    }

    /// Runs `sql` and returns each row as `(team, score, the other integer
    /// columns in name order)`, sorted, so a test does not depend on the order
    /// rows are emitted in.
    fn run_sorted(testing_planner: &mut TestingPlanner, sql: &str) -> Vec<(String, i64, Vec<i64>)> {
        let mut rows: Vec<_> = run(testing_planner, sql)
            .iter()
            .map(|row| {
                let mut other_columns: Vec<_> = row
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| *name != "team" && *name != "score")
                    .collect();
                other_columns.sort_by_key(|(name, _)| name.as_str());
                (
                    row["team"].as_str().unwrap().to_string(),
                    row["score"].as_i64().unwrap(),
                    other_columns
                        .into_iter()
                        .map(|(_, value)| value.as_i64().unwrap())
                        .collect(),
                )
            })
            .collect();
        rows.sort();
        rows
    }

    fn expected_row(team: &str, score: i64, other_values: &[i64]) -> (String, i64, Vec<i64>) {
        (team.to_string(), score, other_values.to_vec())
    }

    #[rstest]
    fn row_number_numbers_each_partition_in_order(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score, \
                    row_number() OVER (PARTITION BY team ORDER BY score DESC) AS position \
             FROM scores WHERE score <> 20",
        );

        assert_eq!(
            rows,
            vec![
                expected_row("a", 10, &[2]),
                expected_row("a", 30, &[1]),
                expected_row("b", 5, &[2]),
                expected_row("b", 7, &[1]),
            ]
        );
    }

    #[rstest]
    fn rank_and_dense_rank_treat_ties_differently(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score, \
                    rank() OVER (PARTITION BY team ORDER BY score) AS a_rank, \
                    dense_rank() OVER (PARTITION BY team ORDER BY score) AS b_dense_rank \
             FROM scores WHERE team = 'a'",
        );

        assert_eq!(
            rows,
            vec![
                expected_row("a", 10, &[1, 1]),
                expected_row("a", 20, &[2, 2]),
                expected_row("a", 20, &[2, 2]),
                expected_row("a", 30, &[4, 3]),
            ]
        );
    }

    #[rstest]
    fn percent_rank_scales_the_rank_to_the_partition(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT score, percent_rank() OVER (ORDER BY score) AS standing \
             FROM scores WHERE team = 'a'",
        );

        let mut standings: Vec<(i64, f64)> = rows
            .iter()
            .map(|row| {
                (
                    row["score"].as_i64().unwrap(),
                    row["standing"].as_f64().unwrap(),
                )
            })
            .collect();
        standings.sort_by(|left, right| left.partial_cmp(right).unwrap());
        assert_eq!(
            standings,
            vec![(10, 0.0), (20, 1.0 / 3.0), (20, 1.0 / 3.0), (30, 1.0)]
        );
    }

    /// Runs `sql`, which selects team a's `score` and two more columns, and
    /// returns each row's three values as text, sorted.
    fn run_for_team_a(
        testing_planner: &mut TestingPlanner,
        sql: &str,
        columns: [&str; 2],
    ) -> Vec<(i64, String, String)> {
        let mut rows: Vec<_> = run(testing_planner, sql)
            .iter()
            .map(|row| {
                (
                    row["score"].as_i64().unwrap(),
                    row[columns[0]].to_string(),
                    row[columns[1]].to_string(),
                )
            })
            .collect();
        rows.sort();
        rows
    }

    fn expected_values(score: i64, first: &str, second: &str) -> (i64, String, String) {
        (score, first.to_string(), second.to_string())
    }

    #[rstest]
    fn lag_and_lead_read_other_rows_of_the_partition(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_for_team_a(
            &mut testing_planner,
            "SELECT score, \
                    lag(score) OVER (PARTITION BY team ORDER BY score) AS previous, \
                    lead(score, 2) OVER (PARTITION BY team ORDER BY score) AS after_next \
             FROM scores WHERE team = 'a'",
            ["previous", "after_next"],
        );

        assert_eq!(
            rows,
            vec![
                expected_values(10, "null", "20"),
                expected_values(20, "10", "30"),
                expected_values(20, "20", "null"),
                expected_values(30, "20", "null"),
            ]
        );
    }

    #[rstest]
    fn a_sum_over_an_ordered_window_adds_up_the_rows_so_far(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_for_team_a(
            &mut testing_planner,
            "SELECT score, \
                    sum(score) OVER (ORDER BY score) AS with_ties, \
                    sum(score) OVER (ORDER BY score ROWS UNBOUNDED PRECEDING) AS by_row \
             FROM scores WHERE team = 'a'",
            ["with_ties", "by_row"],
        );

        assert_eq!(
            rows,
            vec![
                expected_values(10, "10", "10"),
                expected_values(20, "50", "30"),
                expected_values(20, "50", "50"),
                expected_values(30, "80", "80"),
            ]
        );
    }

    #[rstest]
    fn a_sum_over_an_unordered_window_is_the_total_of_the_partition(
        mut testing_planner: TestingPlanner,
    ) {
        add_scores_table(&testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT sum(score) OVER (PARTITION BY team) AS team_total FROM scores",
        );

        let mut totals: Vec<String> = rows
            .iter()
            .map(|row| row["team_total"].to_string())
            .collect();
        totals.sort();
        assert_eq!(totals, vec!["12", "12", "80", "80", "80", "80"]);
    }

    #[rstest]
    fn a_sum_over_a_sliding_frame_is_rejected(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let error = testing_planner
            .plan(
                "SELECT sum(score) OVER (ORDER BY score ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) \
                 FROM scores",
            )
            .unwrap_err();

        assert!(error.to_string().contains("sum over a window"), "{error}");
    }

    #[rstest]
    fn functions_with_different_over_clauses_keep_their_select_order(
        mut testing_planner: TestingPlanner,
    ) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score, \
                    rank() OVER (PARTITION BY team ORDER BY score) AS a_rank_in_team, \
                    rank() OVER (ORDER BY score DESC) AS b_rank_overall, \
                    dense_rank() OVER (PARTITION BY team ORDER BY score) AS c_dense_rank_in_team \
             FROM scores WHERE team = 'b' OR score = 30",
        );

        assert_eq!(
            rows,
            vec![
                expected_row("a", 30, &[1, 1, 1]),
                expected_row("b", 5, &[1, 3, 1]),
                expected_row("b", 7, &[2, 2, 2]),
            ]
        );
    }

    #[rstest]
    fn a_computed_partition_key_groups_rows_by_its_value(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score, \
                    row_number() OVER (PARTITION BY score > 15 ORDER BY score) AS position \
             FROM scores WHERE score <> 20",
        );

        assert_eq!(
            rows,
            vec![
                expected_row("a", 10, &[3]),
                expected_row("a", 30, &[1]),
                expected_row("b", 5, &[1]),
                expected_row("b", 7, &[2]),
            ]
        );
    }

    #[rstest]
    fn qualify_keeps_the_first_rows_of_each_partition(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score FROM scores \
             QUALIFY row_number() OVER (PARTITION BY team ORDER BY score) <= 1",
        );

        assert_eq!(
            rows,
            vec![expected_row("a", 10, &[]), expected_row("b", 5, &[])]
        );
    }

    #[rstest]
    #[case::qualify(
        "SELECT team, score FROM scores \
         QUALIFY row_number() OVER (PARTITION BY team ORDER BY score) <= 2"
    )]
    #[case::filter_on_a_subquery(
        "SELECT team, score FROM ( \
             SELECT team, score, \
                    row_number() OVER (PARTITION BY team ORDER BY score) AS position \
             FROM scores \
         ) WHERE position < 3"
    )]
    fn a_limit_on_row_number_moves_from_the_filter_into_the_window(
        mut testing_planner: TestingPlanner,
        #[case] sql: &str,
    ) {
        add_scores_table(&testing_planner);

        let plan = testing_planner.plan(sql).unwrap().to_string();

        assert!(plan.contains("keep: row_number() <= 2"), "{plan}");
        assert!(!plan.contains("Filter"), "{plan}");
    }

    #[rstest]
    fn a_limit_on_a_lower_window_is_not_applied_before_the_upper_window(
        mut testing_planner: TestingPlanner,
    ) {
        add_scores_table(&testing_planner);

        let rows = run_sorted(
            &mut testing_planner,
            "SELECT team, score, \
                    row_number() OVER (PARTITION BY team ORDER BY score) AS a_position_in_team, \
                    row_number() OVER (ORDER BY score, team) AS b_position_overall \
             FROM scores \
             QUALIFY a_position_in_team <= 1",
        );

        assert_eq!(
            rows,
            vec![
                expected_row("a", 10, &[1, 3]),
                expected_row("b", 5, &[1, 1])
            ]
        );
    }

    #[rstest]
    fn an_unsupported_window_function_is_named_in_the_error(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let error = testing_planner
            .plan("SELECT ntile(2) OVER (ORDER BY score) FROM scores")
            .unwrap_err()
            .to_string();

        assert!(error.contains("window function NTILE"), "{error}");
    }

    #[rstest]
    fn an_order_by_inside_the_function_call_is_rejected(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let error = testing_planner
            .plan(
                "SELECT rank(ORDER BY score DESC) OVER (PARTITION BY team ORDER BY score) \
                 FROM scores",
            )
            .unwrap_err()
            .to_string();

        assert!(
            error.contains("ORDER BY inside a window function call"),
            "{error}"
        );
    }

    #[rstest]
    fn functions_over_the_same_computed_key_share_one_window(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let plan = testing_planner
            .plan(
                "SELECT rank() OVER (PARTITION BY score > 15 ORDER BY score), \
                        dense_rank() OVER (PARTITION BY score > 15 ORDER BY score) \
                 FROM scores",
            )
            .unwrap()
            .to_string();

        assert_eq!(plan.matches("Window(").count(), 1, "{plan}");
    }

    #[rstest]
    fn rows_can_be_ranked_by_an_aggregate(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT team, rank() OVER (ORDER BY sum(score) DESC) AS position \
             FROM scores GROUP BY team",
        );

        let mut positions: Vec<(String, i64)> = rows
            .iter()
            .map(|row| {
                (
                    row["team"].as_str().unwrap().to_string(),
                    row["position"].as_i64().unwrap(),
                )
            })
            .collect();
        positions.sort();
        assert_eq!(positions, vec![("a".to_string(), 1), ("b".to_string(), 2)]);
    }

    #[rstest]
    fn an_empty_over_clause_numbers_every_row(mut testing_planner: TestingPlanner) {
        add_scores_table(&testing_planner);

        let rows = run(
            &mut testing_planner,
            "SELECT row_number() OVER () AS position FROM scores",
        );

        let mut positions: Vec<i64> = rows
            .iter()
            .map(|row| only_column(row).as_i64().unwrap())
            .collect();
        positions.sort();
        assert_eq!(positions, vec![1, 2, 3, 4, 5, 6]);
    }
}
