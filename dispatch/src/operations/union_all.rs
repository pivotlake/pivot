//! A UNION ALL: the rows of several chains, concatenated into one.
//!
//! Every input chain sends its batches into one ordinary work-stealing
//! channel, which a single [`UnionAll`] stage reads the way every stage reads
//! its input. The graph carries an edge from each input's tail to that stage,
//! so it converges on all of them: it finishes only once every input chain
//! has, and a lull on one input is never mistaken for the end.
//!
//! The inputs emit batches of their own schemas, which agree in column types
//! but not necessarily in field names or declared nullability. Downstream
//! stages may take the first batch's schema as the schema of all (a sort's
//! merge does), so the stage restamps every batch with one uniform schema.

use crate::api::{BuildContext, OperatorFactory, OperatorGraphBuilder};
use crate::operations::channels::{ChannelFactory, Sender, StealableChannelFactory};
use crate::operations::unary::{self, Unary, UnaryOperator};
use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// The converging stage of a UNION ALL: forwards each batch under the uniform
/// output schema.
pub struct UnionAll {
    /// Whether each output column can hold NULLs: the declared nullability of
    /// every restamped field, which must be sound for every input.
    nullable: Arc<Vec<bool>>,
    /// Each input schema seen so far, paired with the uniform schema its
    /// batches are restamped to. Inputs reuse their schema across batches, so
    /// a pointer-equality lookup here spares rebuilding the fields per batch.
    restamped: Vec<(SchemaRef, SchemaRef)>,
}

impl UnionAll {
    pub fn new(nullable: Arc<Vec<bool>>) -> Self {
        Self {
            nullable,
            restamped: Vec::new(),
        }
    }

    /// The uniform schema for batches arriving with `input` as their schema.
    ///
    /// Field data types are kept per batch rather than fixed up front: a
    /// variant column's physical layout differs per file, so only its tag
    /// and nullability are uniform, not its struct shape.
    fn uniform_schema(&mut self, input: &SchemaRef) -> SchemaRef {
        if let Some((_, uniform)) = self
            .restamped
            .iter()
            .find(|(seen, _)| Arc::ptr_eq(seen, input))
        {
            return uniform.clone();
        }
        assert_eq!(
            input.fields().len(),
            self.nullable.len(),
            "a UNION ALL input emits exactly the union's columns"
        );
        let fields: Vec<_> = input
            .fields()
            .iter()
            .zip(self.nullable.iter())
            .enumerate()
            .map(|(i, (field, &nullable))| {
                field
                    .as_ref()
                    .clone()
                    .with_name(format!("col{i}"))
                    .with_nullable(nullable)
            })
            .collect();
        let uniform = Arc::new(Schema::new(fields));
        self.restamped.push((input.clone(), uniform.clone()));
        uniform
    }
}

impl Unary<RecordBatch, RecordBatch> for UnionAll {
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        let schema = self.uniform_schema(batch.schema_ref());
        let (_, columns, _) = batch.into_parts();
        let restamped = RecordBatch::try_new(schema, columns)
            .expect("a UNION ALL input's batch matches the union's declared columns");
        sender.send(restamped)?;
        Ok(())
    }
}

/// A UNION ALL: the chains producing the rows, plus the stage they converge on.
pub struct UnionAllFactory {
    /// The head of each input chain, in input order.
    pub inputs: Vec<Box<dyn OperatorFactory<RecordBatch>>>,
    /// This worker's endpoint of the channel every input sends into.
    pub channel_factory: StealableChannelFactory<RecordBatch>,
    pub siblings_left: Arc<AtomicUsize>,
    /// Whether each output column can hold NULLs.
    pub nullable: Arc<Vec<bool>>,
}

impl OperatorFactory<RecordBatch> for UnionAllFactory {
    /// Build the converging stage over the first input's chain, then every
    /// other input as a side graph feeding that same stage.
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<RecordBatch>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        let (tx, rx) = self.channel_factory.build();
        let mut inputs = self.inputs.into_iter();
        let first = inputs.next().expect("a UNION ALL has at least one input");
        let mut graph =
            first
                .build(Box::new(tx.clone()), context)
                .with(Box::new(UnaryOperator::new(
                    UnionAll::new(self.nullable),
                    rx,
                    sender,
                    self.siblings_left,
                )));
        for input in inputs {
            let side = input.build(Box::new(tx.clone()), context);
            graph = graph.with_input_side_graph(side);
        }
        graph
    }
}
