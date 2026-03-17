//! Project operator: transforms each batch by applying a closure.
//!
//! The projection closure receives a `&RecordBatch` and returns a new `RecordBatch`.
//! Common uses: column selection (`batch.project(&indices)`), computing derived columns,
//! or reordering columns.

use crate::operations::channels::Sender;
use crate::operations::unary::Unary;
use crate::operations::unary::factory::UnaryFactory;
use arrow_array::RecordBatch;

/// Factory that wraps a projection closure. `P` is the per-worker closure created by the
/// builder passed to [`RecordBatchOperatorSpec::project`](crate::api::RecordBatchOperatorSpec::project).
pub struct ProjectFactory<P>(pub P);

impl<P> UnaryFactory<RecordBatch, RecordBatch> for ProjectFactory<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send + 'static,
{
    type Unary = Project<P>;

    fn build_unary(self) -> Self::Unary {
        Project { projector: self.0 }
    }
}

pub struct Project<P: FnMut(&RecordBatch) -> RecordBatch + Send> {
    projector: P,
}

impl<P> Unary<RecordBatch, RecordBatch> for Project<P>
where
    P: FnMut(&RecordBatch) -> RecordBatch + Send,
{
    fn consume<S: Sender<RecordBatch>>(
        &mut self,
        batch: RecordBatch,
        sender: &mut S,
    ) -> crate::operations::unary::Result<()> {
        sender.send((self.projector)(&batch))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::run_unary;
    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn two_col_batch(ints: &[i32], strs: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let col_a: ArrayRef = Arc::new(Int32Array::from(ints.to_vec()));
        let col_b: ArrayRef = Arc::new(StringArray::from(strs.to_vec()));
        RecordBatch::try_new(schema, vec![col_a, col_b]).unwrap()
    }

    #[test]
    fn select_single_column() {
        let project = Project {
            projector: |b: &RecordBatch| b.project(&[0]).unwrap(),
        };

        let out = run_unary(project, vec![two_col_batch(&[1, 2, 3], &["a", "b", "c"])]);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_columns(), 1);
        let col = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col.values().to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn reorder_columns() {
        let project = Project {
            projector: |b: &RecordBatch| b.project(&[1, 0]).unwrap(),
        };

        let out = run_unary(project, vec![two_col_batch(&[10], &["x"])]);

        assert_eq!(out[0].num_columns(), 2);
        assert_eq!(out[0].schema().field(0).name(), "b");
        assert_eq!(out[0].schema().field(1).name(), "a");
    }

    #[test]
    fn multiple_batches() {
        let project = Project {
            projector: |b: &RecordBatch| b.project(&[1]).unwrap(),
        };

        let out = run_unary(
            project,
            vec![two_col_batch(&[1], &["x"]), two_col_batch(&[2], &["y"])],
        );

        assert_eq!(out.len(), 2);
        let v0 = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let v1 = out[1]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(v0.value(0), "x");
        assert_eq!(v1.value(0), "y");
    }

    #[test]
    fn identity_projection() {
        let project = Project {
            projector: |b: &RecordBatch| b.clone(),
        };

        let out = run_unary(project, vec![two_col_batch(&[1, 2], &["a", "b"])]);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].num_columns(), 2);
        assert_eq!(out[0].num_rows(), 2);
    }
}
