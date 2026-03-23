mod hashtable;

use std::collections::HashMap;
use arrow_array::RecordBatch;
use crate::operations::Binary;
use crate::operations::channels::Sender;

struct Join {
    // table: HashMap<>
}

impl Binary<RecordBatch, RecordBatch, RecordBatch> for Join {
    fn consume_left<S: Sender<RecordBatch>>(&mut self, item: RecordBatch, sender: &mut S) -> crate::operations::binary::Result<()> {
        todo!()
    }

    fn consume_right<S: Sender<RecordBatch>>(&mut self, item: RecordBatch, sender: &mut S) -> crate::operations::binary::Result<()> {
        todo!()
    }
}