//! Concat: a binary operator that forwards items from both inputs.

use super::{Binary, BinaryFactory, Result};
use crate::operations::channels::Sender;

/// Forwards all items from both inputs unchanged.
pub struct Concat;

impl<T> Binary<T, T, T> for Concat {
    fn consume_left<S: Sender<T>>(&mut self, item: T, sender: &mut S) -> Result<()> {
        sender.send(item)?;
        Ok(())
    }

    fn consume_right<S: Sender<T>>(&mut self, item: T, sender: &mut S) -> Result<()> {
        sender.send(item)?;
        Ok(())
    }
}

/// Factory that creates [`Concat`] operators.
pub struct ConcatFactory;

impl<T: 'static> BinaryFactory<T, T, T> for ConcatFactory {
    type Binary = Concat;
    fn build_binary(self) -> Concat {
        Concat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::unary::test_utils::CollectSender;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn int_batch(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap()
    }

    #[test]
    fn concat_forwards_both_sides() {
        let mut concat = Concat;
        let mut sender = CollectSender::new();

        concat
            .consume_left(int_batch(&[1, 2]), &mut sender)
            .unwrap();
        concat
            .consume_right(int_batch(&[3, 4]), &mut sender)
            .unwrap();
        concat.consume_left(int_batch(&[5]), &mut sender).unwrap();

        assert_eq!(sender.total_rows(), 5);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn concat_finish_returns_true() {
        let mut concat = Concat;
        let mut sender: CollectSender<RecordBatch> = CollectSender::new();
        assert!(concat.finish(&mut sender).unwrap());
    }
}
