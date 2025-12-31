use crate::Output;
use arrow_array::RecordBatch;

pub struct StdOutOutput {}

impl Default for StdOutOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl StdOutOutput {
    pub fn new() -> Self {
        Self {}
    }
}
impl Output for StdOutOutput {
    fn write(&mut self, batch: RecordBatch) {
        println!("{:?}", batch);
    }
}
