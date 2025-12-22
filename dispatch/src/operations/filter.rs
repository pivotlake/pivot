use crate::operations::{Identifier, Operation};
use arrow_array::builder::StringViewBuilder;
use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch,
    builder::{
        ArrayBuilder, BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, Int32Builder,
        Int64Builder, StringBuilder, UInt32Builder, UInt64Builder,
    },
};
use arrow_schema::{DataType, SchemaRef};

macro_rules! append_masked_prim {
    ($arr:expr, $builder_any:expr, $ArrayTy:ty, $BuilderTy:ty, $mask:expr, $nrows:expr) => {{
        let a = $arr.as_any().downcast_ref::<$ArrayTy>().unwrap();
        let b = $builder_any.downcast_mut::<$BuilderTy>().unwrap();
        for i in 0..$nrows {
            if !$mask.value(i) {
                continue;
            }
            if a.is_null(i) {
                b.append_null();
            } else {
                b.append_value(a.value(i));
            }
        }
    }};
}

macro_rules! append_masked_utf8 {
    ($arr:expr, $builder_any:expr, $mask:expr, $nrows:expr) => {{
        let a = $arr
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let b = $builder_any
            .downcast_mut::<arrow_array::builder::StringBuilder>()
            .unwrap();
        for i in 0..$nrows {
            if !$mask.value(i) {
                continue;
            }
            if a.is_null(i) {
                b.append_null();
            } else {
                b.append_value(a.value(i));
            }
        }
    }};
}

macro_rules! append_masked_utf8_view {
    ($arr:expr, $builder_any:expr, $mask:expr, $nrows:expr) => {{
        let a = $arr
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
            .unwrap();
        let b = $builder_any
            .downcast_mut::<arrow_array::builder::StringViewBuilder>()
            .unwrap();

        for i in 0..$nrows {
            if !$mask.value(i) {
                continue;
            }
            if a.is_null(i) {
                b.append_null();
            } else {
                b.append_value(a.value(i)); // &str
            }
        }
    }};
}

macro_rules! append_masked_binary {
    ($arr:expr, $builder_any:expr, $mask:expr, $nrows:expr) => {{
        let a = $arr
            .as_any()
            .downcast_ref::<arrow_array::BinaryArray>()
            .unwrap();
        let b = $builder_any
            .downcast_mut::<arrow_array::builder::BinaryBuilder>()
            .unwrap();
        for i in 0..$nrows {
            if !$mask.value(i) {
                continue;
            }
            if a.is_null(i) {
                b.append_null();
            } else {
                b.append_value(a.value(i)); // &[u8]
            }
        }
    }};
}

macro_rules! append_masked_match {
    ($dt:expr, $arr:expr, $builder_any:expr, $mask:expr, $nrows:expr) => {{
        match $dt {
            DataType::Boolean => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::BooleanArray,
                arrow_array::builder::BooleanBuilder,
                $mask,
                $nrows
            ),
            DataType::Int32 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::Int32Array,
                arrow_array::builder::Int32Builder,
                $mask,
                $nrows
            ),
            DataType::Int64 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::Int64Array,
                arrow_array::builder::Int64Builder,
                $mask,
                $nrows
            ),
            DataType::UInt32 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::UInt32Array,
                arrow_array::builder::UInt32Builder,
                $mask,
                $nrows
            ),
            DataType::UInt64 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::UInt64Array,
                arrow_array::builder::UInt64Builder,
                $mask,
                $nrows
            ),
            DataType::Float32 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::Float32Array,
                arrow_array::builder::Float32Builder,
                $mask,
                $nrows
            ),
            DataType::Float64 => append_masked_prim!(
                $arr,
                $builder_any,
                arrow_array::Float64Array,
                arrow_array::builder::Float64Builder,
                $mask,
                $nrows
            ),
            DataType::Utf8 => append_masked_utf8!($arr, $builder_any, $mask, $nrows),
            DataType::Utf8View => append_masked_utf8_view!($arr, $builder_any, $mask, $nrows),
            DataType::Binary => append_masked_binary!($arr, $builder_any, $mask, $nrows),
            other => panic!("Filter builder not implemented for datatype: {other:?}"),
        }
    }};
}
pub struct Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    id: Identifier,
    func: F,
    schema: Option<SchemaRef>,
    builders: Option<Vec<Box<dyn ArrayBuilder>>>,
}

impl<F> Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    pub fn new(id: Identifier, func: F) -> Self {
        Self {
            id,
            func,
            schema: None,
            builders: None,
        }
    }

    fn ensure_initialized(&mut self, batch: &RecordBatch) {
        if self.schema.is_none() {
            self.schema = Some(batch.schema());
        }
        if self.builders.is_none() {
            let mut builders = Vec::with_capacity(batch.schema().fields().len());
            for f in batch.schema().fields() {
                let b: Box<dyn ArrayBuilder> = match f.data_type() {
                    DataType::Boolean => Box::new(BooleanBuilder::new()),
                    DataType::Int32 => Box::new(Int32Builder::new()),
                    DataType::Int64 => Box::new(Int64Builder::new()),
                    DataType::UInt32 => Box::new(UInt32Builder::new()),
                    DataType::UInt64 => Box::new(UInt64Builder::new()),
                    DataType::Float32 => Box::new(Float32Builder::new()),
                    DataType::Float64 => Box::new(Float64Builder::new()),
                    DataType::Utf8 => Box::new(StringBuilder::new()),
                    DataType::Utf8View => Box::new(StringViewBuilder::new()),
                    DataType::Binary => Box::new(BinaryBuilder::new()),
                    other => panic!("Filter builder not implemented for datatype: {other:?}"),
                };
                builders.push(b);
            }
            self.builders = Some(builders);
        }
    }
}

impl<F> Operation for Filter<F>
where
    F: Fn(&RecordBatch) -> BooleanArray + Send + Sync + 'static,
{
    fn id(&self) -> Identifier {
        self.id
    }

    fn consume_output_batch(&mut self) -> Option<RecordBatch> {
        let builders = self.builders.as_mut()?;

        if builders.iter().all(|b| b.is_empty()) {
            return None;
        }

        let cols: Vec<ArrayRef> = builders.iter_mut().map(|b| b.finish()).collect();
        Some(
            RecordBatch::try_new(self.schema.as_ref()?.clone(), cols)
                .expect("RecordBatch::try_new failed"),
        )
    }

    fn run(&mut self, batch: &RecordBatch) {
        self.ensure_initialized(batch);

        let mask = (self.func)(batch);
        debug_assert_eq!(mask.len(), batch.num_rows());

        if mask.true_count() == 0 {
            return;
        }

        let builders = self.builders.as_mut().unwrap();

        for (builder, arr) in builders.iter_mut().zip(batch.columns()) {
            let dt = arr.data_type().clone();
            let b_any = builder.as_any_mut();
            append_masked_match!(dt, arr, b_any, &mask, batch.num_rows());
        }
    }

    fn finish(&mut self) -> Option<RecordBatch> {
        self.consume_output_batch()
    }
}
