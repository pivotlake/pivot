//! Shreds a variant column into typed leaves on slab memory.
//!
//! [`shred_into_slabs`] is the write-side counterpart of
//! `parquet_variant_compute::shred_variant`, and produces the same array: the
//! `{metadata, value, typed_value{..}}` struct the reader expects, with the
//! same nullability at every level. The difference is where the memory lives.
//! Shredding builds one pair of columns (a typed leaf and a binary fallback)
//! for every shredded path, for every row of the row group, and the encode
//! workers do it for a row group each at the same time. Those columns are the
//! biggest transient allocation of a compaction, and on the global allocator
//! they sit beside the buffer pool's budget rather than inside it. Here every
//! per-row buffer is a slab, so the work is accounted for and pre-faulted like
//! the rest of the dataflow.
//!
//! Only the leaf types [`super::infer`] can choose are shredded: `Int64`,
//! `Float64`, `Utf8View`, and nested objects. A value that does not fit its
//! path's type keeps its bytes in the fallback column, exactly as the
//! reference implementation does; the differential tests below hold the two
//! to array equality.
//!
//! Routing a row's fields to their columns never decodes a field name as a
//! string. A document refers to its fields by their ids in its metadata
//! dictionary, and the dictionary's names are compared as bytes: a row whose
//! dictionary is about the size of the document (the common case, since a
//! dictionary is normally built from the document's own keys) gets one merge
//! of the dictionary against the sorted column names, after which every field
//! routes by indexing a table with its id; a sparse document over an oversized
//! dictionary instead looks each of its fields up in a byte-keyed map. See
//! [`merge_routes_cheaper`].

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use ahash::RandomState;

use arrow_array::builder::make_view;
use arrow_array::types::{BinaryViewType, ByteViewType, Float64Type, Int64Type, StringViewType};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, GenericByteViewArray, PrimitiveArray, StructArray,
};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow_schema::ArrowError;
use arrow_schema::{DataType, Field, Fields};
use dispatch::arrays::{ArrayBuilder, PrimitiveBuilder, SlabColumn, ValidityBuilder};
use dispatch::memory::{BUFFER_SIZE, SlabAllocator};
use parquet_variant::{
    ObjectBuilder, ParentState, ReadOnlyMetadataBuilder, ValueBuilder, Variant, VariantMetadata,
};
use parquet_variant_compute::VariantArray;

use super::super::error::{WriteError, WriteResult};

/// Bytes of out-of-line view values copied into one slab block before the next
/// block is opened.
const VALUE_BLOCK_BYTES: usize = 256 * 1024;

/// Shred `array` into `shredding`, the `typed_value` type chosen for it, with
/// every per-row buffer on slabs from `allocator`.
pub(super) fn shred_into_slabs(
    array: &VariantArray,
    shredding: &DataType,
    allocator: &mut SlabAllocator,
) -> WriteResult<VariantArray> {
    let rows = array.len();
    let mut root = FieldShredder::new(shredding, rows, true, allocator)?;
    for row in 0..rows {
        if array.is_null(row) {
            root.append_null();
        } else {
            root.append_value(array.value(row), allocator)?;
        }
    }
    let mut shared = SharedNullColumn::new(rows);
    let (value, typed_value, nulls) = root.finish(allocator, &mut shared)?;

    let metadata: ArrayRef = Arc::new(array.metadata_field().clone());
    let fields = column_fields(Some(typed_value.data_type()));
    let inner = StructArray::try_new(fields, vec![metadata, Arc::new(value), typed_value], nulls)?;
    Ok(VariantArray::try_new(&inner)?)
}

/// The fields of a variant column holding `typed_value`, or of a plain
/// `{metadata, value}` column without one. Every shredded column this module
/// builds, and every layout it rewrites one into, uses exactly these, so
/// columns of one layout share one Arrow type wherever they were built.
pub(super) fn column_fields(typed_value: Option<&DataType>) -> Fields {
    let mut fields = vec![
        Field::new("metadata", DataType::BinaryView, false),
        Field::new("value", DataType::BinaryView, true),
    ];
    if let Some(typed_value) = typed_value {
        fields.push(Field::new("typed_value", typed_value.clone(), true));
    }
    Fields::from(fields)
}

/// The fields of one shredded position below the top level: its fallback
/// `value` beside its `typed_value`.
pub(super) fn pair_fields(typed_value: &DataType) -> Fields {
    Fields::from(vec![
        Field::new("value", DataType::BinaryView, true),
        Field::new("typed_value", typed_value.clone(), true),
    ])
}

/// The field an object's `typed_value` struct holds for the child `name`
/// shredded into `typed_value`. A field of an object is never null itself,
/// only absent through its parent, so the pair is not nullable.
pub(super) fn child_field(name: &str, typed_value: &DataType) -> Field {
    Field::new(name, DataType::Struct(pair_fields(typed_value)), false)
}

/// The fields of an object's `typed_value` struct for the children `fields`
/// of a shredding type.
pub(super) fn object_fields(fields: &Fields) -> Fields {
    fields
        .iter()
        .map(|field| child_field(field.name(), &typed_value_type(field.data_type())))
        .collect()
}

/// The Arrow type of the `typed_value` column a position shredded into
/// `shredding` holds: a scalar as it is, an object as the struct of its
/// children's pairs.
pub(super) fn typed_value_type(shredding: &DataType) -> DataType {
    match shredding {
        DataType::Struct(fields) => DataType::Struct(object_fields(fields)),
        scalar => scalar.clone(),
    }
}

/// One shredded position: its fallback `value` column, its `typed_value`, and
/// the validity of the struct holding both. At the top level that validity is
/// the column's own; a field of an object is never null itself, only absent
/// through its parent, so its struct rows are all valid.
struct FieldShredder {
    fallback: FallbackColumn,
    typed: TypedShredder,
    presence: Validity,
    top_level: bool,
}

enum TypedShredder {
    Int64(PrimitiveLeaf<Int64Type>),
    Float64(PrimitiveLeaf<Float64Type>),
    Utf8View(ByteViewColumn<StringViewType>),
    Object(ObjectShredder),
}

impl FieldShredder {
    fn new(
        data_type: &DataType,
        rows: usize,
        top_level: bool,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<Self> {
        let typed = match data_type {
            DataType::Int64 => TypedShredder::Int64(PrimitiveLeaf::with_capacity(allocator, rows)),
            DataType::Float64 => {
                TypedShredder::Float64(PrimitiveLeaf::with_capacity(allocator, rows))
            }
            DataType::Utf8View => {
                TypedShredder::Utf8View(ByteViewColumn::with_capacity(allocator, rows))
            }
            DataType::Struct(fields) => {
                TypedShredder::Object(ObjectShredder::new(fields, rows, allocator)?)
            }
            other => return Err(WriteError::UnsupportedType(other.clone())),
        };
        Ok(Self {
            fallback: FallbackColumn::new(rows),
            typed,
            presence: Validity::with_capacity(allocator, rows),
            top_level,
        })
    }

    /// A row with nothing at this position: a null document at the top level,
    /// or a field the document does not have.
    fn append_null(&mut self) {
        self.presence.append(!self.top_level);
        self.fallback.append_null();
        match &mut self.typed {
            TypedShredder::Int64(leaf) => leaf.append_null(),
            TypedShredder::Float64(leaf) => leaf.append_null(),
            TypedShredder::Utf8View(column) => column.append_null(),
            TypedShredder::Object(object) => object.append_null(),
        }
    }

    fn append_value(
        &mut self,
        value: Variant<'_, '_>,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<()> {
        self.presence.append(true);
        match &mut self.typed {
            TypedShredder::Int64(leaf) => match value.as_int64() {
                Some(typed) => {
                    leaf.append(typed);
                    self.fallback.append_null();
                }
                None => {
                    leaf.append_null();
                    self.fallback.append_variant(&value, allocator);
                }
            },
            TypedShredder::Float64(leaf) => match value.as_f64() {
                Some(typed) => {
                    leaf.append(typed);
                    self.fallback.append_null();
                }
                None => {
                    leaf.append_null();
                    self.fallback.append_variant(&value, allocator);
                }
            },
            TypedShredder::Utf8View(column) => match value.as_string() {
                Some(typed) => {
                    column.append_value(typed.as_bytes(), allocator);
                    self.fallback.append_null();
                }
                None => {
                    column.append_null();
                    self.fallback.append_variant(&value, allocator);
                }
            },
            TypedShredder::Object(object) => {
                object.append_value(value, &mut self.fallback, allocator)?
            }
        }
        Ok(())
    }

    fn finish(
        self,
        allocator: &mut SlabAllocator,
        shared: &mut SharedNullColumn,
    ) -> WriteResult<(
        GenericByteViewArray<BinaryViewType>,
        ArrayRef,
        Option<NullBuffer>,
    )> {
        let value = self.fallback.finish(allocator, shared);
        let typed_value: ArrayRef = match self.typed {
            TypedShredder::Int64(leaf) => leaf.finish(),
            TypedShredder::Float64(leaf) => leaf.finish(),
            TypedShredder::Utf8View(column) => Arc::new(column.finish()),
            TypedShredder::Object(object) => Arc::new(object.finish(allocator, shared)?),
        };
        Ok((value, typed_value, self.presence.finish()))
    }
}

/// The shredded fields of an object position, and the validity of the
/// `typed_value` struct they form: null on the rows where the position held
/// something other than an object.
struct ObjectShredder {
    children: Vec<(String, FieldShredder)>,
    /// The children's names as bytes in byte order, each with its position in
    /// `children`: one side of the per-row merge against a dictionary.
    sorted_names: Vec<(Vec<u8>, usize)>,
    /// Name bytes to position in `children`, for the rows that route field by
    /// field.
    by_name: HashMap<Vec<u8>, usize, RandomState>,
    /// Which children the row being appended has provided, so the rest can be
    /// marked absent; reused across rows.
    seen: Vec<bool>,
    /// Where each entry of the row's dictionary routes, by field id; rebuilt
    /// for every row that merges, reused so no row allocates.
    child_by_field_id: Vec<Option<usize>>,
    typed_nulls: Validity,
}

/// Whether a row is cheaper to route by merging its whole dictionary against
/// the column names than by looking its fields up one by one.
///
/// The merge costs one comparison per dictionary entry plus one per column;
/// the lookups cost a hash and a compare per field the document has. When a
/// dictionary is built from the document's own keys, which is how variant
/// builders write one, the two counts are the same and the merge wins by
/// touching nothing per field afterwards. A writer that shares one large
/// dictionary across many sparse documents inverts that, so beyond this ratio
/// of entries to fields the row falls back to lookups.
fn merge_routes_cheaper(dictionary_entries: usize, fields: usize) -> bool {
    const ENTRIES_PER_FIELD_LIMIT: usize = 4;
    dictionary_entries <= fields.saturating_mul(ENTRIES_PER_FIELD_LIMIT)
}

impl ObjectShredder {
    fn new(fields: &Fields, rows: usize, allocator: &mut SlabAllocator) -> WriteResult<Self> {
        let children = fields
            .iter()
            .map(|field| {
                let child = FieldShredder::new(field.data_type(), rows, false, allocator)?;
                Ok((field.name().clone(), child))
            })
            .collect::<WriteResult<Vec<_>>>()?;
        let mut sorted_names: Vec<(Vec<u8>, usize)> = children
            .iter()
            .enumerate()
            .map(|(position, (name, _))| (name.as_bytes().to_vec(), position))
            .collect();
        sorted_names.sort();
        let by_name = sorted_names.iter().cloned().collect();
        Ok(Self {
            seen: vec![false; children.len()],
            child_by_field_id: Vec::new(),
            children,
            sorted_names,
            by_name,
            typed_nulls: Validity::with_capacity(allocator, rows),
        })
    }

    fn append_null(&mut self) {
        self.typed_nulls.append(false);
        for (_, child) in &mut self.children {
            child.append_null();
        }
    }

    /// Route the row's fields: a shredded field goes to its child, anything
    /// else is kept, as an object of the leftovers, in `fallback`. A value
    /// that is not an object at all goes to `fallback` whole.
    fn append_value(
        &mut self,
        value: Variant<'_, '_>,
        fallback: &mut FallbackColumn,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<()> {
        let Some(object) = value.as_object() else {
            fallback.append_variant(&value, allocator);
            self.append_null();
            return Ok(());
        };

        let metadata = value.metadata();
        self.seen.fill(false);
        let mut leftovers: Vec<(u32, Variant<'_, '_>)> = Vec::new();
        if merge_routes_cheaper(metadata.len(), object.len()) {
            self.route_dictionary(metadata)?;
            for (field_id, field_value) in object.iter_with_field_ids() {
                let route = self
                    .child_by_field_id
                    .get(field_id as usize)
                    .copied()
                    .ok_or_else(|| field_id_out_of_dictionary(field_id, metadata))?;
                match route {
                    Some(position) => {
                        self.children[position]
                            .1
                            .append_value(field_value, allocator)?;
                        self.seen[position] = true;
                    }
                    None => leftovers.push((field_id, field_value)),
                }
            }
        } else {
            for (field_id, field_value) in object.iter_with_field_ids() {
                let name = metadata.name_bytes(field_id as usize)?;
                match self.by_name.get(name) {
                    Some(&position) => {
                        self.children[position]
                            .1
                            .append_value(field_value, allocator)?;
                        self.seen[position] = true;
                    }
                    None => leftovers.push((field_id, field_value)),
                }
            }
        }
        for (position, seen) in self.seen.iter().enumerate() {
            if !seen {
                self.children[position].1.append_null();
            }
        }
        if leftovers.is_empty() {
            fallback.append_null();
        } else {
            let bytes = leftover_object(metadata, &leftovers);
            fallback.append_bytes(&bytes, allocator);
        }
        self.typed_nulls.append(true);
        Ok(())
    }

    /// Fill `child_by_field_id` for `metadata`: one merge of its entries, in
    /// byte order, against the sorted column names. An unsorted dictionary
    /// (the format allows one) is walked in the order its names sort into,
    /// so the ids the table is indexed by stay the dictionary's own.
    fn route_dictionary(&mut self, metadata: &VariantMetadata<'_>) -> WriteResult<()> {
        let entries = metadata.len();
        self.child_by_field_id.clear();
        self.child_by_field_id.resize(entries, None);
        let order = metadata.dictionary_order()?;

        let mut column = 0;
        for &field_id in order.ids_by_name() {
            let name = order.name(field_id);
            while column < self.sorted_names.len() && self.sorted_names[column].0.as_slice() < name
            {
                column += 1;
            }
            if column < self.sorted_names.len() && self.sorted_names[column].0 == name {
                self.child_by_field_id[field_id as usize] = Some(self.sorted_names[column].1);
            }
        }
        Ok(())
    }

    fn finish(
        self,
        allocator: &mut SlabAllocator,
        shared: &mut SharedNullColumn,
    ) -> WriteResult<StructArray> {
        let mut fields = Vec::with_capacity(self.children.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.children.len());
        for (name, child) in self.children {
            let (value, typed_value, nulls) = child.finish(allocator, shared)?;
            let pair = pair_fields(typed_value.data_type());
            fields.push(child_field(&name, typed_value.data_type()));
            let field = StructArray::try_new(pair, vec![Arc::new(value), typed_value], nulls)?;
            arrays.push(Arc::new(field));
        }
        Ok(StructArray::try_new(
            Fields::from(fields),
            arrays,
            self.typed_nulls.finish(),
        )?)
    }
}

/// The variant bytes of an object holding just `fields`, named by their ids in
/// `metadata` (they are fields of the document being shredded). Nested values
/// are copied byte for byte, so they keep referring to that same dictionary.
fn leftover_object(metadata: &VariantMetadata<'_>, fields: &[(u32, Variant<'_, '_>)]) -> Vec<u8> {
    let mut value_builder = ValueBuilder::new();
    let mut metadata_builder = ReadOnlyMetadataBuilder::new(metadata);
    let state = ParentState::variant(&mut value_builder, &mut metadata_builder);
    let mut object = ObjectBuilder::new(state, false);
    for (field_id, value) in fields {
        object.insert_bytes_by_field_id(*field_id, value.clone());
    }
    object.finish();
    value_builder.into_inner()
}

fn field_id_out_of_dictionary(field_id: u32, metadata: &VariantMetadata<'_>) -> WriteError {
    WriteError::Arrow(ArrowError::InvalidArgumentError(format!(
        "variant field id {field_id} is outside its {}-entry metadata dictionary",
        metadata.len()
    )))
}

/// The fallback `value` column of one position. Most positions never fall
/// back, so the column is only materialized by the first row that does; a
/// column that never was is one shared all-null array at the end.
struct FallbackColumn {
    rows: usize,
    /// Rows appended before the column was materialized, all null.
    leading_nulls: usize,
    column: Option<ByteViewColumn<BinaryViewType>>,
}

impl FallbackColumn {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            leading_nulls: 0,
            column: None,
        }
    }

    fn append_null(&mut self) {
        match &mut self.column {
            Some(column) => column.append_null(),
            None => self.leading_nulls += 1,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8], allocator: &mut SlabAllocator) {
        let column = self.column.get_or_insert_with(|| {
            let mut column = ByteViewColumn::with_capacity(allocator, self.rows);
            for _ in 0..self.leading_nulls {
                column.append_null();
            }
            column
        });
        column.append_value(bytes, allocator);
    }

    /// Keep `value` whole: its own bytes for a scalar, the raw bytes of an
    /// object or list, which stay valid against the row's metadata.
    fn append_variant(&mut self, value: &Variant<'_, '_>, allocator: &mut SlabAllocator) {
        let mut value_builder = ValueBuilder::new();
        let mut metadata_builder = ReadOnlyMetadataBuilder::new(value.metadata());
        let state = ParentState::variant(&mut value_builder, &mut metadata_builder);
        ValueBuilder::append_variant_bytes(state, value.clone());
        self.append_bytes(&value_builder.into_inner(), allocator);
    }

    fn finish(
        self,
        allocator: &mut SlabAllocator,
        shared: &mut SharedNullColumn,
    ) -> GenericByteViewArray<BinaryViewType> {
        match self.column {
            Some(column) => column.finish(),
            None => shared.all_null(allocator),
        }
    }
}

/// The one all-null array every position without values shares: a zeroed
/// buffer wide enough for any leaf and an all-null bitmap, built on first
/// use. A zeroed 16-byte view is a valid empty inline value, and a zeroed
/// 8-byte integer or float is a valid placeholder, so one buffer serves every
/// leaf type under the bitmap that marks every row null.
pub(super) struct SharedNullColumn {
    rows: usize,
    parts: Option<(Buffer, NullBuffer)>,
}

impl SharedNullColumn {
    pub(super) fn new(rows: usize) -> Self {
        Self { rows, parts: None }
    }

    fn parts(&mut self, allocator: &mut SlabAllocator) -> (Buffer, NullBuffer) {
        let rows = self.rows;
        self.parts
            .get_or_insert_with(|| {
                let views = SlabColumn::<u128> {
                    values: allocator.create_slab_buffer(rows, true),
                    len: rows,
                };
                let mut nulls = ValidityBuilder::with_capacity(allocator, rows);
                nulls.append_n(rows, false);
                (
                    views.into_buffer(),
                    NullBuffer::new(BooleanBuffer::new(nulls.into_buffer(), 0, rows)),
                )
            })
            .clone()
    }

    pub(super) fn all_null(
        &mut self,
        allocator: &mut SlabAllocator,
    ) -> GenericByteViewArray<BinaryViewType> {
        let rows = self.rows;
        let (views, nulls) = self.parts(allocator);
        // SAFETY: every view is the zeroed view, a valid empty inline value,
        // and the bitmap marks every row null.
        unsafe {
            GenericByteViewArray::new_unchecked(
                ScalarBuffer::new(views, 0, rows),
                Vec::new(),
                Some(nulls),
            )
        }
    }

    /// An all-null array of `data_type`: a leaf over the shared parts, or an
    /// object whose children are all null too under the shared bitmap.
    pub(super) fn all_null_of(
        &mut self,
        data_type: &DataType,
        allocator: &mut SlabAllocator,
    ) -> WriteResult<ArrayRef> {
        let rows = self.rows;
        Ok(match data_type {
            DataType::BinaryView => Arc::new(self.all_null(allocator)),
            DataType::Utf8View => {
                let (views, nulls) = self.parts(allocator);
                // SAFETY: as in `all_null`; an empty string is valid UTF-8.
                Arc::new(unsafe {
                    GenericByteViewArray::<StringViewType>::new_unchecked(
                        ScalarBuffer::new(views, 0, rows),
                        Vec::new(),
                        Some(nulls),
                    )
                })
            }
            DataType::Int64 => {
                let (values, nulls) = self.parts(allocator);
                Arc::new(PrimitiveArray::<Int64Type>::new(
                    ScalarBuffer::new(values, 0, rows),
                    Some(nulls),
                ))
            }
            DataType::Float64 => {
                let (values, nulls) = self.parts(allocator);
                Arc::new(PrimitiveArray::<Float64Type>::new(
                    ScalarBuffer::new(values, 0, rows),
                    Some(nulls),
                ))
            }
            DataType::Struct(fields) => {
                let children = fields
                    .iter()
                    .map(|field| self.all_null_of(field.data_type(), allocator))
                    .collect::<WriteResult<Vec<_>>>()?;
                let (_, nulls) = self.parts(allocator);
                Arc::new(StructArray::try_new(fields.clone(), children, Some(nulls))?)
            }
            other => return Err(WriteError::UnsupportedType(other.clone())),
        })
    }
}

/// A typed leaf: its values on a slab, absent rows holding a placeholder that
/// the validity marks dead.
struct PrimitiveLeaf<T: ArrowPrimitiveType> {
    values: PrimitiveBuilder<T>,
    validity: Validity,
}

impl<T: ArrowPrimitiveType> PrimitiveLeaf<T> {
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            values: PrimitiveBuilder::with_capacity(allocator, rows),
            validity: Validity::with_capacity(allocator, rows),
        }
    }

    fn append(&mut self, value: T::Native) {
        self.values.push(&value, 1);
        self.validity.append(true);
    }

    fn append_null(&mut self) {
        self.values.push(&T::Native::default(), 1);
        self.validity.append(false);
    }

    fn finish(self) -> ArrayRef {
        self.values.into_array(self.validity.into_buffer_if_nulls())
    }
}

/// A byte-view column built value by value: the 16-byte views on one slab,
/// out-of-line bytes copied into slab blocks of their own.
pub(super) struct ByteViewColumn<V: ByteViewType> {
    views: SlabColumn<u128>,
    validity: Validity,
    /// Sealed blocks, in the order the views number them.
    blocks: Vec<Buffer>,
    /// The block being filled and its capacity; it becomes the next sealed
    /// block, so its id is `blocks.len()` while it is open.
    open_block: Option<(SlabColumn<u8>, usize)>,
    view_type: PhantomData<V>,
}

impl<V: ByteViewType> ByteViewColumn<V> {
    pub(super) fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            views: SlabColumn::with_capacity(allocator, rows),
            validity: Validity::with_capacity(allocator, rows),
            blocks: Vec::new(),
            open_block: None,
            view_type: PhantomData,
        }
    }

    pub(super) fn append_null(&mut self) {
        self.views.push(0);
        self.validity.append(false);
    }

    pub(super) fn append_value(&mut self, bytes: &[u8], allocator: &mut SlabAllocator) {
        let view = if bytes.len() <= 12 {
            make_view(bytes, 0, 0)
        } else {
            let (block, offset) = self.copy_out_of_line(bytes, allocator);
            make_view(bytes, block, offset)
        };
        self.views.push(view);
        self.validity.append(true);
    }

    /// Copy `bytes` into the open block, sealing it first when they do not
    /// fit, and return where they landed.
    fn copy_out_of_line(&mut self, bytes: &[u8], allocator: &mut SlabAllocator) -> (u32, u32) {
        // A value no slab can hold takes a block of its own on the heap.
        if bytes.len() > BUFFER_SIZE {
            self.seal_open_block();
            self.blocks.push(Buffer::from(bytes.to_vec()));
            return ((self.blocks.len() - 1) as u32, 0);
        }
        let fits = self
            .open_block
            .as_ref()
            .is_some_and(|(block, capacity)| capacity - block.len() >= bytes.len());
        if !fits {
            self.seal_open_block();
            let capacity = VALUE_BLOCK_BYTES.max(bytes.len());
            self.open_block = Some((SlabColumn::with_capacity(allocator, capacity), capacity));
        }
        let (block, _) = self.open_block.as_mut().expect("a block was just opened");
        let offset = block.len();
        block.spare_mut(bytes.len()).copy_from_slice(bytes);
        (self.blocks.len() as u32, offset as u32)
    }

    fn seal_open_block(&mut self) {
        if let Some((block, _)) = self.open_block.take() {
            self.blocks.push(block.into_buffer());
        }
    }

    pub(super) fn finish(mut self) -> GenericByteViewArray<V> {
        self.seal_open_block();
        let rows = self.views.len();
        let views = ScalarBuffer::new(self.views.into_buffer(), 0, rows);
        let nulls = self.validity.finish();
        // SAFETY: every view was made by `make_view` over the bytes it names,
        // at the block and offset they were copied to, or is the zeroed view
        // of a null row. The string flavour only ever receives `str` bytes.
        unsafe { GenericByteViewArray::new_unchecked(views, self.blocks, nulls) }
    }
}

/// A validity bitmap on a slab that, like Arrow's own builder, yields no
/// buffer at all when every row turned out valid.
struct Validity {
    bits: ValidityBuilder,
    rows: usize,
    nulls: usize,
}

impl Validity {
    fn with_capacity(allocator: &mut SlabAllocator, rows: usize) -> Self {
        Self {
            bits: ValidityBuilder::with_capacity(allocator, rows),
            rows: 0,
            nulls: 0,
        }
    }

    fn append(&mut self, present: bool) {
        self.bits.append_n(1, present);
        self.rows += 1;
        if !present {
            self.nulls += 1;
        }
    }

    fn into_buffer_if_nulls(self) -> Option<Buffer> {
        (self.nulls > 0).then(|| self.bits.into_buffer())
    }

    fn finish(self) -> Option<NullBuffer> {
        let rows = self.rows;
        self.into_buffer_if_nulls()
            .map(|bits| NullBuffer::new(BooleanBuffer::new(bits, 0, rows)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::Field;
    use dispatch::memory::init_test_free_pool;
    use parquet_variant_compute::{json_to_variant, shred_variant};

    /// A variant column with one document per row; `None` is a null row.
    fn documents(rows: &[Option<&str>]) -> VariantArray {
        let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
        json_to_variant(&json).unwrap()
    }

    /// A variant column whose rows are built with `VariantBuilder`, so a test
    /// controls each row's dictionary: the names it holds and their order.
    fn rows_with_dictionaries(rows: Vec<(Vec<u8>, Vec<u8>)>) -> VariantArray {
        use arrow_array::BinaryViewArray;

        let metadata: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            rows.iter().map(|(metadata, _)| metadata.as_slice()),
        ));
        let value: ArrayRef = Arc::new(BinaryViewArray::from_iter_values(
            rows.iter().map(|(_, value)| value.as_slice()),
        ));
        let fields = Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]);
        VariantArray::try_new(&StructArray::try_new(fields, vec![metadata, value], None).unwrap())
            .unwrap()
    }

    /// One document `{a: 1, b: "two", zz: 3}` whose dictionary is pre-seeded
    /// with `seeded` names in the order given.
    fn document_with_seeded_dictionary(seeded: &[&str]) -> (Vec<u8>, Vec<u8>) {
        use parquet_variant::VariantBuilder;

        let mut builder = VariantBuilder::new().with_field_names(seeded.iter().copied());
        let mut object = builder.new_object();
        object.insert("a", 1i64);
        object.insert("b", "two");
        object.insert("zz", 3i64);
        object.finish();
        builder.finish()
    }

    fn object(fields: Vec<(&str, DataType)>) -> DataType {
        DataType::Struct(
            fields
                .into_iter()
                .map(|(name, data_type)| Field::new(name, data_type, true))
                .collect(),
        )
    }

    /// Shred `array` both ways and hand back the pair.
    fn both(array: &VariantArray, shredding: &DataType) -> (StructArray, StructArray) {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let ours = shred_into_slabs(array, shredding, &mut allocator).unwrap();
        let reference = shred_variant(array, shredding).unwrap();
        (ours.into_inner(), reference.into_inner())
    }

    /// Every way a document can meet the schema: matching leaves, absent
    /// fields, extra fields, a leaf of the wrong type, a nested object that is
    /// sometimes a scalar, JSON nulls, and rows that are not objects at all.
    #[test]
    fn shreds_documents_exactly_like_the_reference() {
        let array = documents(&[
            Some(
                r#"{"id": 1, "name": "ab", "ratio": 0.5, "user": {"age": 30, "city": "a city name longer than twelve bytes"}}"#,
            ),
            Some(
                r#"{"id": 9007199254740993, "name": "a name longer than twelve bytes", "ratio": 2, "extra": [1, 2], "user": {"age": "old"}}"#,
            ),
            Some(r#"{"id": "not a number", "ratio": "x", "user": 5, "other": {"k": null}}"#),
            Some(r#"{"name": null, "user": {}}"#),
            None,
            Some(r#"7"#),
            Some(r#"[1, "two"]"#),
            Some(r#"{"id": 2, "name": "cd", "ratio": 1.5, "user": {"age": 31, "city": "b"}}"#),
        ]);
        let shredding = object(vec![
            ("id", DataType::Int64),
            ("name", DataType::Utf8View),
            ("ratio", DataType::Float64),
            (
                "user",
                object(vec![("age", DataType::Int64), ("city", DataType::Utf8View)]),
            ),
        ]);

        let (ours, reference) = both(&array, &shredding);

        assert_eq!(ours.data_type(), reference.data_type());
        assert_eq!(ours, reference);
    }

    /// The widened schema the planner derives by shredding zero rows must be
    /// the schema the rows are later shredded into.
    #[test]
    fn agrees_with_the_reference_on_zero_rows() {
        let array = documents(&[Some(r#"{"id": 1}"#)]);
        let sliced = array.into_inner().slice(0, 0);
        let empty = VariantArray::try_new(&sliced).unwrap();
        let shredding = object(vec![("id", DataType::Int64), ("tags", DataType::Utf8View)]);

        let (ours, reference) = both(&empty, &shredding);

        assert_eq!(ours.data_type(), reference.data_type());
        assert_eq!(ours, reference);
    }

    /// A dictionary far larger than the document routes field by field, and
    /// gets the same answer as the reference.
    #[test]
    fn a_sparse_document_over_an_oversized_dictionary_shreds_like_the_reference() {
        let seeded: Vec<String> = (0..64).map(|i| format!("unused_{i:02}")).collect();
        let seeded: Vec<&str> = seeded.iter().map(String::as_str).collect();
        let array = rows_with_dictionaries(vec![
            document_with_seeded_dictionary(&seeded),
            document_with_seeded_dictionary(&seeded),
        ]);
        assert!(!merge_routes_cheaper(array.value(0).metadata().len(), 3));
        let shredding = object(vec![("a", DataType::Int64), ("b", DataType::Utf8View)]);

        let (ours, reference) = both(&array, &shredding);

        assert_eq!(ours, reference);
    }

    /// A dictionary whose entries are not in byte order is merged through a
    /// sorted permutation, and gets the same answer as the reference.
    #[test]
    fn an_unsorted_dictionary_shreds_like_the_reference() {
        let array = rows_with_dictionaries(vec![document_with_seeded_dictionary(&["zz", "b"])]);
        assert!(!array.value(0).metadata().is_sorted());
        assert!(merge_routes_cheaper(array.value(0).metadata().len(), 3));
        let shredding = object(vec![("a", DataType::Int64), ("b", DataType::Utf8View)]);

        let (ours, reference) = both(&array, &shredding);

        assert_eq!(ours, reference);
    }

    #[test]
    fn a_dictionary_built_from_the_document_merges() {
        assert!(merge_routes_cheaper(50, 50));
        assert!(merge_routes_cheaper(200, 50));
        assert!(!merge_routes_cheaper(201, 50));
        assert!(!merge_routes_cheaper(1, 0));
    }

    /// Paths that never fall back cost one shared all-null column, not one
    /// views buffer each.
    #[test]
    fn positions_without_fallbacks_share_one_null_column() {
        let array = documents(&[Some(r#"{"a": 1, "b": 2}"#), Some(r#"{"a": 3, "b": 4}"#)]);
        let shredding = object(vec![("a", DataType::Int64), ("b", DataType::Int64)]);
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);

        let shredded = shred_into_slabs(&array, &shredding, &mut allocator).unwrap();

        let typed_value = shredded.typed_value_field().unwrap().as_struct();
        let fallback = |name: &str| {
            typed_value
                .column_by_name(name)
                .unwrap()
                .as_struct()
                .column_by_name("value")
                .unwrap()
                .as_binary_view()
                .views()
                .as_ptr()
        };
        assert_eq!(fallback("a"), fallback("b"));
    }
}
