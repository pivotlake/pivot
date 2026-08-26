//! Rewriting a shredded variant column from one layout to another without
//! rebuilding its documents.
//!
//! A compaction reads files that each shredded a variant column their own
//! way and writes files that decide their layout afresh. Folding every
//! document back together on the way in and taking it apart again on the way
//! out costs more than everything else a merge does, and nearly all of that
//! work moves a value from one typed leaf to the same typed leaf. The two
//! steps here move only what changes.
//!
//! [`widen`] brings a batch from its file's layout to the union of the input
//! files' layouts, so batches of every file share one Arrow type and can be
//! gathered together. A leaf the union adds is filled from the leftover of
//! the rows that carry its path there, and a path the file shredded as
//! another kind than the union settled on goes back to the leftover to be
//! filled in from there: a leaf never stands beside a leftover that still
//! names its field, which is not a shredded column and is not something the
//! folds below can read.
//!
//! [`reshred`] brings a run of rows from the union layout to the output
//! layout. A leaf the output keeps is passed through as it is. A leaf the
//! output drops is folded back into the leftover of the rows that held it. A
//! leaf the output adds is filled from the leftover of the rows that carry the
//! field there.
//!
//! Both are the same operation at every object position: some children fold
//! back into the position's leftover, some are pulled out of it. Folding is
//! the unshred kernel over just those children, on just the rows that hold
//! them; pulling is the slab shredder over just the leftover. Neither touches
//! a row that has nothing to move.

use std::sync::Arc;

use arrow_array::types::{BinaryViewType, Float64Type, Int64Type};
use arrow_array::{Array, ArrayRef, BinaryViewArray, PrimitiveArray, StringViewArray, StructArray};
use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields};
use dispatch::arrays::take::{concat_chunks, take, take_chunked};
use dispatch::memory::SlabAllocator;
use parquet_variant::{
    ObjectBuilder, ParentState, ReadOnlyMetadataBuilder, ValueBuilder, Variant, VariantMetadata,
};
use parquet_variant_compute::{VariantArray, unshred_variant};

use super::super::error::{WriteError, WriteResult};
use super::shred::{
    self, ByteViewColumn, SharedNullColumn, column_fields, object_fields, pair_fields,
    typed_value_type,
};

/// Bring `array`, a variant column in one file's layout, to the `union`
/// layout (a `typed_value` type). Leaves the union keeps as the file has
/// them are shared with the input; the rest are filled from the rows'
/// leftovers.
pub(super) fn widen(
    array: &VariantArray,
    union: &DataType,
    allocator: &mut SlabAllocator,
) -> WriteResult<VariantArray> {
    let metadata = array.metadata_field();
    let mut shared = SharedNullColumn::new(array.len());
    // A null document's own columns need not be null (a struct may be null
    // over valid children), and everything below reads a position's presence
    // off its columns, so a null row is made null in them here, once.
    let value = array
        .value_field()
        .map(|value| masked_by_rows(value, array.nulls()));
    let typed = array
        .typed_value_field()
        .map(|typed| masked_by_rows_typed(typed, array.nulls()))
        .transpose()?;
    let node = Node {
        value: value.as_ref(),
        typed: typed.as_ref(),
    };
    let (value, typed_value) = widen_node(metadata, node, union, allocator, &mut shared)?;
    assemble_column(metadata, value, Some(typed_value), array.nulls().cloned())
}

/// `value` null wherever `rows` is.
fn masked_by_rows(value: &BinaryViewArray, rows: Option<&NullBuffer>) -> BinaryViewArray {
    let Some(rows) = rows else {
        return value.clone();
    };
    let nulls = NullBuffer::new(&present(value) & rows.inner());
    let (views, buffers, _) = value.clone().into_parts();
    // SAFETY: the views and buffers are the input's own, already validated.
    unsafe { BinaryViewArray::new_unchecked(views, buffers, Some(nulls)) }
}

/// `typed` null wherever `rows` is.
fn masked_by_rows_typed(typed: &ArrayRef, rows: Option<&NullBuffer>) -> WriteResult<ArrayRef> {
    let Some(rows) = rows else {
        return Ok(typed.clone());
    };
    let nulls = NullBuffer::new(&present(typed.as_ref()) & rows.inner());
    Ok(match typed.as_any().downcast_ref::<StructArray>() {
        Some(object) => Arc::new(StructArray::try_new(
            object.fields().clone(),
            object.columns().to_vec(),
            Some(nulls),
        )?),
        None => {
            let data = typed.to_data().into_builder().nulls(Some(nulls)).build()?;
            arrow_array::make_array(data)
        }
    })
}

/// Bring `array`, a variant column in the union layout, to `shredding`, the
/// `typed_value` type chosen for the file. An empty struct means no path is
/// typed, and the result is the plain `{metadata, value}` column.
pub(super) fn reshred(
    array: &VariantArray,
    shredding: &DataType,
    allocator: &mut SlabAllocator,
) -> WriteResult<VariantArray> {
    let DataType::Struct(fields) = shredding else {
        return Err(WriteError::UnsupportedType(shredding.clone()));
    };
    let metadata = array.metadata_field();
    let mut shared = SharedNullColumn::new(array.len());
    let node = Node {
        value: array.value_field(),
        typed: array.typed_value_field(),
    };
    let target = (!fields.is_empty()).then_some(shredding);
    let (value, typed_value) = reshred_node(metadata, node, target, allocator, &mut shared)?;
    assemble_column(metadata, value, typed_value, array.nulls().cloned())
}

/// One shredded position's columns, whichever of them the layout holds: the
/// leftover `value` and the `typed_value`.
struct Node<'a> {
    value: Option<&'a BinaryViewArray>,
    typed: Option<&'a ArrayRef>,
}

impl<'a> Node<'a> {
    fn of_pair(pair: &'a StructArray) -> WriteResult<Self> {
        let value = match pair.column_by_name("value") {
            Some(value) => Some(
                value
                    .as_any()
                    .downcast_ref::<BinaryViewArray>()
                    .ok_or_else(|| WriteError::UnsupportedType(value.data_type().clone()))?,
            ),
            None => None,
        };
        Ok(Self {
            value,
            typed: pair.column_by_name("typed_value"),
        })
    }

    fn value_or_null(
        &self,
        allocator: &mut SlabAllocator,
        shared: &mut SharedNullColumn,
    ) -> BinaryViewArray {
        match self.value {
            Some(value) => value.clone(),
            None => shared.all_null(allocator),
        }
    }
}

fn widen_node(
    metadata: &BinaryViewArray,
    node: Node<'_>,
    target: &DataType,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<(BinaryViewArray, ArrayRef)> {
    let DataType::Struct(target_fields) = target else {
        return Ok(match node.typed {
            Some(typed) if typed.data_type() == target => {
                (node.value_or_null(allocator, shared), typed.clone())
            }
            Some(typed) => (
                fold_back(metadata, node.value, typed.clone(), allocator, shared)?,
                shared.all_null_of(target, allocator)?,
            ),
            None => (
                node.value_or_null(allocator, shared),
                shared.all_null_of(target, allocator)?,
            ),
        });
    };

    let object = node
        .typed
        .and_then(|typed| typed.as_any().downcast_ref::<StructArray>());
    // A scalar where the union has an object folds back whole.
    let mut value = match (node.typed, object) {
        (Some(typed), None) => fold_back(metadata, node.value, typed.clone(), allocator, shared)?,
        _ => node.value_or_null(allocator, shared),
    };
    if let Some(object) = object {
        value = fold_back_dropped(metadata, value, object, target_fields, allocator, shared)?;
    }

    let kept: Vec<Option<&StructArray>> = target_fields
        .iter()
        .map(|target_field| object.and_then(|object| kept_child(object, target_field)))
        .collect();
    // A leaf beside a leftover that still names its field is not a shredded
    // column, so the paths the union keeps that this file did not shred as
    // that kind are pulled out of the leftover here, into the leaves the
    // union gives them. The paths the file did shred as that kind are left
    // where they are, so a file already in the union layout is not shredded
    // again.
    let added: Fields = target_fields
        .iter()
        .zip(&kept)
        .filter(|(_, kept)| kept.is_none())
        .map(|(target_field, _)| target_field.clone())
        .collect();
    let pulled = if added.is_empty() || value.null_count() == value.len() {
        None
    } else {
        let (leftover, typed) = pull(metadata, value, &DataType::Struct(added), allocator, shared)?;
        value = leftover;
        Some(
            typed
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("an object shredding type yields a struct")
                .clone(),
        )
    };

    let mut children: Vec<ArrayRef> = Vec::with_capacity(target_fields.len());
    for (target_field, kept) in target_fields.iter().zip(&kept) {
        let pair = match kept {
            Some(pair) => {
                let (value, typed) = widen_node(
                    metadata,
                    Node::of_pair(pair)?,
                    target_field.data_type(),
                    allocator,
                    shared,
                )?;
                assemble_pair(value, typed)?
            }
            None => match pulled
                .as_ref()
                .and_then(|pulled| pulled.column_by_name(target_field.name()))
            {
                Some(pair) => pair
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .expect("a shredded child is a struct")
                    .clone(),
                None => all_null_pair(target_field.data_type(), allocator, shared)?,
            },
        };
        children.push(Arc::new(pair));
    }
    let nulls = match (object, &pulled) {
        (None, None) => Some(
            shared
                .all_null(allocator)
                .nulls()
                .cloned()
                .expect("all rows null"),
        ),
        (Some(object), None) => object.nulls().cloned(),
        (None, Some(pulled)) => pulled.nulls().cloned(),
        (Some(object), Some(pulled)) => Some(NullBuffer::new(&present(object) | &present(pulled))),
    };
    let typed = StructArray::try_new(object_fields(target_fields), children, nulls)?;
    Ok((value, Arc::new(typed)))
}

fn reshred_node(
    metadata: &BinaryViewArray,
    node: Node<'_>,
    target: Option<&DataType>,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<(BinaryViewArray, Option<ArrayRef>)> {
    let Some(target) = target else {
        // Nothing is typed here any more: whatever was folds back.
        let value = match node.typed {
            Some(typed) => fold_back(metadata, node.value, typed.clone(), allocator, shared)?,
            None => node.value_or_null(allocator, shared),
        };
        return Ok((value, None));
    };

    let DataType::Struct(target_fields) = target else {
        if let Some(typed) = node.typed
            && typed.data_type() == target
        {
            return Ok((node.value_or_null(allocator, shared), Some(typed.clone())));
        }
        // Another kind was typed here: it goes back to the leftover, and the
        // leftover is shredded afresh into the type wanted now.
        let value = match node.typed {
            Some(typed) => fold_back(metadata, node.value, typed.clone(), allocator, shared)?,
            None => node.value_or_null(allocator, shared),
        };
        if value.null_count() == value.len() {
            return Ok((value, Some(shared.all_null_of(target, allocator)?)));
        }
        let (value, typed) = pull(metadata, value, target, allocator, shared)?;
        return Ok((value, Some(typed)));
    };

    let object = node
        .typed
        .and_then(|typed| typed.as_any().downcast_ref::<StructArray>());
    let mut value = match (node.typed, object) {
        (Some(typed), None) => fold_back(metadata, node.value, typed.clone(), allocator, shared)?,
        _ => node.value_or_null(allocator, shared),
    };
    if let Some(object) = object {
        value = fold_back_dropped(metadata, value, object, target_fields, allocator, shared)?;
    }

    // The rows that carry a kept child in this leftover give it up now.
    let pulled = if value.null_count() < value.len() {
        let (leftover, typed) = pull(metadata, value, target, allocator, shared)?;
        value = leftover;
        Some(
            typed
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("an object shredding type yields a struct")
                .clone(),
        )
    } else {
        None
    };

    let mut children: Vec<ArrayRef> = Vec::with_capacity(target_fields.len());
    for target_field in target_fields {
        let carried = match object.and_then(|object| kept_child(object, target_field)) {
            Some(pair) => {
                let (value, typed) = reshred_node(
                    metadata,
                    Node::of_pair(pair)?,
                    Some(target_field.data_type()),
                    allocator,
                    shared,
                )?;
                Some(assemble_pair(
                    value,
                    typed.expect("a typed target yields a typed column"),
                )?)
            }
            None => None,
        };
        let pulled_child = pulled
            .as_ref()
            .and_then(|pulled| pulled.column_by_name(target_field.name()))
            .map(|pair| {
                pair.as_any()
                    .downcast_ref::<StructArray>()
                    .expect("a shredded child is a struct")
                    .clone()
            });
        children.push(merge_pairs(
            carried,
            pulled_child,
            target_field.data_type(),
            allocator,
            shared,
        )?);
    }
    let nulls = match (object, &pulled) {
        (None, None) => Some(
            shared
                .all_null(allocator)
                .nulls()
                .cloned()
                .expect("all rows null"),
        ),
        (Some(object), None) => object.nulls().cloned(),
        (None, Some(pulled)) => pulled.nulls().cloned(),
        (Some(object), Some(pulled)) => Some(NullBuffer::new(&present(object) | &present(pulled))),
    };
    let typed = StructArray::try_new(object_fields(target_fields), children, nulls)?;
    Ok((value, Some(Arc::new(typed))))
}

/// Shred `value`, a leftover column, into `target`: the rows that hold
/// something the target types give it up, and keep the rest as their
/// leftover.
///
/// Only the rows with a leftover are shredded, packed together, and the
/// result is spread back over all rows: a child no row gave anything to
/// comes out as the shared all-null column, so the pass costs the rows that
/// hold a leftover times the children they fill, never the rows of the run
/// times the children of the layout.
fn pull(
    metadata: &BinaryViewArray,
    value: BinaryViewArray,
    target: &DataType,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<(BinaryViewArray, ArrayRef)> {
    let rows = value.len();
    let held = present(&value);
    let indices: Vec<u32> = held.set_indices().map(|row| row as u32).collect();
    let (packed_metadata, packed_value) = if indices.len() == rows {
        (metadata.clone(), value)
    } else {
        let metadata: ArrayRef = Arc::new(metadata.clone());
        let value: ArrayRef = Arc::new(value);
        (
            as_binary_view(take(allocator, &metadata, &indices)?),
            as_binary_view(take(allocator, &value, &indices)?),
        )
    };
    let leftover = assemble_column(&packed_metadata, packed_value, None, None)?;
    let shredded = shred::shred_into_slabs(&leftover, target, allocator)?;
    let packed_leftover: ArrayRef = Arc::new(
        shredded
            .value_field()
            .expect("the shredder keeps a value column")
            .clone(),
    );
    let packed_typed = shredded
        .typed_value_field()
        .expect("the shredder builds the typed column");
    if indices.len() == rows {
        return Ok((as_binary_view(packed_leftover), packed_typed.clone()));
    }

    // Row `row` of the run is packed row `k` when it held a leftover, and
    // the filler's own row otherwise.
    let mut mapping = Vec::with_capacity(rows);
    let mut packed_row = 0;
    for row in 0..rows {
        if held.value(row) {
            mapping.push((0, packed_row));
            packed_row += 1;
        } else {
            mapping.push((1, row as u32));
        }
    }
    let leftover = as_binary_view(spread_leaf(&packed_leftover, &mapping, allocator, shared)?);
    let typed = spread_typed(packed_typed, &mapping, allocator, shared)?;
    Ok((leftover, typed))
}

/// `packed`, a `typed_value` column over the rows that held a leftover,
/// spread over the run by `mapping`. An object's validity spreads with it;
/// a child's pair stays all-valid, as a pair is.
fn spread_typed(
    packed: &ArrayRef,
    mapping: &[(u32, u32)],
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<ArrayRef> {
    let Some(object) = packed.as_any().downcast_ref::<StructArray>() else {
        return spread_leaf(packed, mapping, allocator, shared);
    };
    if object.null_count() == object.len() {
        return shared.all_null_of(packed.data_type(), allocator);
    }
    let mut children: Vec<ArrayRef> = Vec::with_capacity(object.num_columns());
    for column in object.columns() {
        let pair = column
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("a shredded child is a pair");
        let value = spread_leaf(
            pair.column_by_name("value").expect("a pair has a value"),
            mapping,
            allocator,
            shared,
        )?;
        let typed = spread_typed(
            pair.column_by_name("typed_value")
                .expect("a pair has a typed_value"),
            mapping,
            allocator,
            shared,
        )?;
        children.push(Arc::new(StructArray::try_new(
            pair.fields().clone(),
            vec![value, typed],
            None,
        )?));
    }
    let packed_validity = present(object);
    let nulls = BooleanBuffer::collect_bool(mapping.len(), |row| match mapping[row] {
        (0, packed_row) => packed_validity.value(packed_row as usize),
        _ => false,
    });
    Ok(Arc::new(StructArray::try_new(
        object.fields().clone(),
        children,
        Some(NullBuffer::new(nulls)),
    )?))
}

/// `packed`, a leaf over the rows that held a leftover, spread over the run
/// by `mapping`: the shared all-null column when no row gave it anything.
fn spread_leaf(
    packed: &ArrayRef,
    mapping: &[(u32, u32)],
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<ArrayRef> {
    let filler = shared.all_null_of(packed.data_type(), allocator)?;
    if packed.null_count() == packed.len() {
        return Ok(filler);
    }
    Ok(take_chunked(allocator, &[packed.clone(), filler], mapping)?)
}

fn as_binary_view(array: ArrayRef) -> BinaryViewArray {
    array
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .expect("a binary view column")
        .clone()
}

/// Whether the child `field` of an object is one `target_fields` keeps as the
/// same kind: an object for an object, a scalar for the same scalar type.
/// Anything else folds back.
fn is_kept(field: &FieldRef, column: &ArrayRef, target_fields: &Fields) -> bool {
    let Some((_, target)) = target_fields.find(field.name()) else {
        return false;
    };
    typed_type_of_pair(column).is_some_and(|typed| same_kind(typed, target.data_type()))
}

/// The child of `object` that `target_field` keeps, if the object has it as
/// the same kind.
fn kept_child<'a>(object: &'a StructArray, target_field: &FieldRef) -> Option<&'a StructArray> {
    let (index, field) = object.fields().find(target_field.name())?;
    let column = &object.columns()[index];
    is_kept(field, column, &Fields::from(vec![target_field.clone()]))
        .then(|| column.as_any().downcast_ref::<StructArray>())
        .flatten()
}

fn typed_type_of_pair(pair: &ArrayRef) -> Option<&DataType> {
    let DataType::Struct(fields) = pair.data_type() else {
        return None;
    };
    fields
        .find("typed_value")
        .map(|(_, field)| field.data_type())
}

fn same_kind(existing: &DataType, target: &DataType) -> bool {
    match (existing, target) {
        (DataType::Struct(_), DataType::Struct(_)) => true,
        _ => existing == target,
    }
}

/// Fold the children of `object` that `target_fields` does not keep back
/// into `value`, the object's leftover. Scalar children fold together, one
/// pass over the rows that hold any of them; an object child, whose values
/// are a subtree, folds through the unshred kernel on its own.
fn fold_back_dropped(
    metadata: &BinaryViewArray,
    mut value: BinaryViewArray,
    object: &StructArray,
    target_fields: &Fields,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<BinaryViewArray> {
    let mut scalars: Vec<FoldedScalar<'_>> = Vec::new();
    for (field, column) in object.fields().iter().zip(object.columns()) {
        if is_kept(field, column, target_fields) {
            continue;
        }
        match FoldedScalar::of(field, column) {
            Some(scalar) => scalars.push(scalar),
            None => value = fold_back_child(metadata, &value, field, column, allocator, shared)?,
        }
    }
    if scalars.is_empty() {
        return Ok(value);
    }
    fold_back_scalars(metadata, &value, &scalars, allocator)
}

/// A dropped scalar child as the fold reads it: its name, its typed leaf,
/// and its fallback.
struct FoldedScalar<'a> {
    name: &'a [u8],
    typed: ScalarLeaf<'a>,
    fallback: Option<&'a BinaryViewArray>,
    held: BooleanBuffer,
}

enum ScalarLeaf<'a> {
    Int64(&'a PrimitiveArray<Int64Type>),
    Float64(&'a PrimitiveArray<Float64Type>),
    Utf8View(&'a StringViewArray),
}

impl<'a> FoldedScalar<'a> {
    /// The child `pair` named `field` as a scalar, or `None` when it is an
    /// object or a leaf of a type the fold does not rebuild.
    fn of(field: &'a FieldRef, pair: &'a ArrayRef) -> Option<Self> {
        let pair_struct = pair.as_any().downcast_ref::<StructArray>()?;
        let typed = pair_struct.column_by_name("typed_value")?;
        let typed = match typed.data_type() {
            DataType::Int64 => ScalarLeaf::Int64(typed.as_any().downcast_ref()?),
            DataType::Float64 => ScalarLeaf::Float64(typed.as_any().downcast_ref()?),
            DataType::Utf8View => ScalarLeaf::Utf8View(typed.as_any().downcast_ref()?),
            _ => return None,
        };
        let fallback = pair_struct
            .column_by_name("value")
            .and_then(|value| value.as_any().downcast_ref::<BinaryViewArray>());
        Some(Self {
            name: field.name().as_bytes(),
            typed,
            fallback,
            held: pair_present(pair),
        })
    }

    /// The value this child holds on `row`, if any: its typed value, else
    /// its fallback's bytes.
    fn value_on(&self, row: usize, metadata: VariantMetadata<'a>) -> Option<Variant<'a, 'a>> {
        if !self.held.value(row) {
            return None;
        }
        let typed = match self.typed {
            ScalarLeaf::Int64(leaf) => leaf.is_valid(row).then(|| Variant::from(leaf.value(row))),
            ScalarLeaf::Float64(leaf) => leaf.is_valid(row).then(|| Variant::from(leaf.value(row))),
            ScalarLeaf::Utf8View(leaf) => {
                leaf.is_valid(row).then(|| Variant::from(leaf.value(row)))
            }
        };
        typed.or_else(|| {
            self.fallback
                .filter(|fallback| fallback.is_valid(row))
                .map(|fallback| Variant::new_with_metadata(metadata, fallback.value(row)))
        })
    }
}

/// Rebuild the leftover of every row that holds any of `scalars`: the old
/// leftover's fields and the folded values, each under its id in the row's
/// dictionary. Other rows keep their leftover as it was.
fn fold_back_scalars(
    metadata: &BinaryViewArray,
    value: &BinaryViewArray,
    scalars: &[FoldedScalar<'_>],
    allocator: &mut SlabAllocator,
) -> WriteResult<BinaryViewArray> {
    let rows = value.len();
    let mut held = BooleanBuffer::new_unset(rows);
    for scalar in scalars {
        held = &held | &scalar.held;
    }
    if held.count_set_bits() == 0 {
        return Ok(value.clone());
    }
    let mut column = ByteViewColumn::<BinaryViewType>::with_capacity(allocator, rows);
    let mut bytes = Vec::new();
    for row in 0..rows {
        if !held.value(row) {
            if value.is_valid(row) {
                column.append_value(value.value(row), allocator);
            } else {
                column.append_null();
            }
            continue;
        }
        let dictionary = VariantMetadata::new(metadata.value(row));
        let mut value_builder = ValueBuilder::new();
        let mut metadata_builder = ReadOnlyMetadataBuilder::new(&dictionary);
        let state = ParentState::variant(&mut value_builder, &mut metadata_builder);
        let mut object = ObjectBuilder::new(state, false);
        if value.is_valid(row) {
            let leftover = Variant::new_with_metadata(dictionary.clone(), value.value(row));
            let Some(fields) = leftover.as_object() else {
                return Err(WriteError::Arrow(ArrowError::InvalidArgumentError(
                    "a position holding typed fields has a leftover that is not an object"
                        .to_string(),
                )));
            };
            for (field_id, field_value) in fields.iter_with_field_ids() {
                object.insert_bytes_by_field_id(field_id, field_value);
            }
        }
        for scalar in scalars {
            let Some(folded) = scalar.value_on(row, dictionary.clone()) else {
                continue;
            };
            let field_id = field_id_of(&dictionary, scalar.name)?;
            object.insert_bytes_by_field_id(field_id, folded);
        }
        object.finish();
        bytes.clear();
        bytes.extend_from_slice(&value_builder.into_inner());
        column.append_value(&bytes, allocator);
    }
    Ok(column.finish())
}

/// The id `name` has in `dictionary`. The row's document had the field, so
/// its dictionary names it.
fn field_id_of(dictionary: &VariantMetadata<'_>, name: &[u8]) -> WriteResult<u32> {
    for id in 0..dictionary.len() {
        if dictionary.name_bytes(id)? == name {
            return Ok(id as u32);
        }
    }
    Err(WriteError::Arrow(ArrowError::InvalidArgumentError(
        format!(
            "a folded field is missing from its row's {}-entry dictionary",
            dictionary.len()
        ),
    )))
}

/// Fold the child `pair` (the shredded position `field` of an object) back
/// into `value`, the object's leftover, on the rows that hold it.
fn fold_back_child(
    metadata: &BinaryViewArray,
    value: &BinaryViewArray,
    field: &FieldRef,
    pair: &ArrayRef,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<BinaryViewArray> {
    let held = pair_present(pair);
    let typed = StructArray::try_new(
        Fields::from(vec![field.clone()]),
        vec![pair.clone()],
        Some(NullBuffer::new(held.clone())),
    )?;
    fold_back_rows(
        metadata,
        Some(value),
        Arc::new(typed),
        held,
        allocator,
        shared,
    )
}

/// Fold `typed`, the whole typed column at a position, back into `value`,
/// that position's leftover, on the rows that hold it.
fn fold_back(
    metadata: &BinaryViewArray,
    value: Option<&BinaryViewArray>,
    typed: ArrayRef,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<BinaryViewArray> {
    let held = present(typed.as_ref());
    fold_back_rows(metadata, value, typed, held, allocator, shared)
}

/// Rebuild the leftover of the rows `held` marks as the document `typed`
/// and the old leftover make together, and keep every other row's leftover
/// as it was.
fn fold_back_rows(
    metadata: &BinaryViewArray,
    value: Option<&BinaryViewArray>,
    typed: ArrayRef,
    held: BooleanBuffer,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<BinaryViewArray> {
    if held.count_set_bits() == 0 {
        return Ok(match value {
            Some(value) => value.clone(),
            None => shared.all_null(allocator),
        });
    }
    let rows = typed.len();
    let mut fields = vec![Field::new("metadata", DataType::BinaryView, false)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(metadata.clone())];
    if let Some(value) = value {
        fields.push(Field::new("value", DataType::BinaryView, true));
        columns.push(Arc::new(value.clone()));
    }
    fields.push(Field::new("typed_value", typed.data_type().clone(), true));
    columns.push(typed);
    // Only the rows with something to fold are rebuilt; the wrapper's nulls
    // keep the unshred off the rest.
    let wrapper = StructArray::try_new(
        Fields::from(fields),
        columns,
        Some(NullBuffer::new(held.clone())),
    )?;
    let unshredded = unshred_variant(&VariantArray::try_new(&wrapper)?)?;
    let folded: ArrayRef = Arc::new(
        unshredded
            .value_field()
            .expect("an unshredded column has a value field")
            .clone(),
    );
    let folded = concat_chunks(allocator, std::slice::from_ref(&folded))?;
    let merged = match value {
        None => folded,
        Some(value) => {
            let mapping: Vec<(u32, u32)> = (0..rows)
                .map(|row| (if held.value(row) { 0 } else { 1 }, row as u32))
                .collect();
            take_chunked(allocator, &[folded, Arc::new(value.clone())], &mapping)?
        }
    };
    Ok(merged
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .expect("a gathered binary view column")
        .clone())
}

/// One child of the output, from the rows that carried it typed and the rows
/// that gave it up from their leftover. A row holds it on at most one side.
fn merge_pairs(
    carried: Option<StructArray>,
    pulled: Option<StructArray>,
    typed_value: &DataType,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<ArrayRef> {
    Ok(match (carried, pulled) {
        (None, None) => Arc::new(all_null_pair(typed_value, allocator, shared)?),
        (Some(carried), None) => Arc::new(carried),
        (None, Some(pulled)) => Arc::new(pulled),
        (Some(carried), Some(pulled)) => {
            let carried: ArrayRef = Arc::new(carried);
            let pulled: ArrayRef = Arc::new(pulled);
            let held = pair_present(&carried);
            if pair_present(&pulled).count_set_bits() == 0 {
                carried
            } else if held.count_set_bits() == 0 {
                pulled
            } else {
                let mapping: Vec<(u32, u32)> = (0..carried.len())
                    .map(|row| (if held.value(row) { 0 } else { 1 }, row as u32))
                    .collect();
                take_chunked(allocator, &[carried, pulled], &mapping)?
            }
        }
    })
}

/// The rows on which `array` holds something: its validity, all rows when it
/// has none.
fn present(array: &dyn Array) -> BooleanBuffer {
    match array.nulls() {
        Some(nulls) => nulls.inner().clone(),
        None => BooleanBuffer::new_set(array.len()),
    }
}

/// The rows on which a shredded position holds something, in either of its
/// columns.
fn pair_present(pair: &ArrayRef) -> BooleanBuffer {
    let pair = pair
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("a shredded position is a struct");
    let mut held = BooleanBuffer::new_unset(pair.len());
    for column in pair.columns() {
        held = &held | &present(column.as_ref());
    }
    held
}

fn assemble_pair(value: BinaryViewArray, typed: ArrayRef) -> WriteResult<StructArray> {
    Ok(StructArray::try_new(
        pair_fields(typed.data_type()),
        vec![Arc::new(value), typed],
        None,
    )?)
}

/// An absent position shredded into `shredding`: both columns null on every
/// row.
fn all_null_pair(
    shredding: &DataType,
    allocator: &mut SlabAllocator,
    shared: &mut SharedNullColumn,
) -> WriteResult<StructArray> {
    let typed_value = typed_value_type(shredding);
    let value: ArrayRef = Arc::new(shared.all_null(allocator));
    let typed = shared.all_null_of(&typed_value, allocator)?;
    Ok(StructArray::try_new(
        pair_fields(&typed_value),
        vec![value, typed],
        None,
    )?)
}

fn assemble_column(
    metadata: &BinaryViewArray,
    value: BinaryViewArray,
    typed_value: Option<ArrayRef>,
    nulls: Option<NullBuffer>,
) -> WriteResult<VariantArray> {
    let fields = column_fields(typed_value.as_ref().map(|typed| typed.data_type()));
    let mut columns: Vec<ArrayRef> = vec![Arc::new(metadata.clone()), Arc::new(value)];
    columns.extend(typed_value);
    Ok(VariantArray::try_new(&StructArray::try_new(
        fields, columns, nulls,
    )?)?)
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use dispatch::memory::init_test_free_pool;
    use parquet_variant_compute::json_to_variant;
    use parquet_variant_json::VariantToJson;

    use super::*;

    fn documents(rows: &[Option<&str>]) -> VariantArray {
        let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
        json_to_variant(&json).unwrap()
    }

    fn object(fields: Vec<(&str, DataType)>) -> DataType {
        DataType::Struct(
            fields
                .into_iter()
                .map(|(name, data_type)| Field::new(name, data_type, true))
                .collect(),
        )
    }

    fn shredded(
        rows: &[Option<&str>],
        layout: &DataType,
        allocator: &mut SlabAllocator,
    ) -> VariantArray {
        shred::shred_into_slabs(&documents(rows), layout, allocator).unwrap()
    }

    /// The rows of `arrays` dealt round-robin into one column, as a sort
    /// interleaves the files of a merge.
    fn interleaved(arrays: &[VariantArray], allocator: &mut SlabAllocator) -> VariantArray {
        let chunks: Vec<ArrayRef> = arrays
            .iter()
            .map(|array| Arc::new(array.inner().clone()) as ArrayRef)
            .collect();
        let mapping = round_robin(arrays.iter().map(|array| array.len()).collect());
        VariantArray::try_new(&take_chunked(allocator, &chunks, &mapping).unwrap()).unwrap()
    }

    fn round_robin(lengths: Vec<usize>) -> Vec<(u32, u32)> {
        let mut mapping = Vec::new();
        let mut row = 0;
        while mapping.len() < lengths.iter().sum::<usize>() {
            for (chunk, &length) in lengths.iter().enumerate() {
                if row < length {
                    mapping.push((chunk as u32, row as u32));
                }
            }
            row += 1;
        }
        mapping
    }

    fn round_robin_rows<'a>(files: &[&[Option<&'a str>]]) -> Vec<Option<&'a str>> {
        let lengths = files.iter().map(|file| file.len()).collect();
        round_robin(lengths)
            .into_iter()
            .map(|(chunk, row)| files[chunk as usize][row as usize])
            .collect()
    }

    /// Every row's document with the kind of every value, so that a value
    /// read back as another kind is a difference. An integer's width and a
    /// string's encoding are not: a value re-encodes at the width of the leaf
    /// it passed through.
    fn typed_documents(array: &VariantArray) -> Vec<Option<String>> {
        let whole = unshred_variant(array).unwrap();
        (0..whole.len())
            .map(|row| {
                whole.is_valid(row).then(|| {
                    let mut text = format!("{:?}", whole.value(row));
                    for width in ["Int8(", "Int16(", "Int32(", "Int64("] {
                        text = text.replace(width, "Int(");
                    }
                    text.replace("ShortString(ShortString(", "String((")
                })
            })
            .collect()
    }

    fn rendered(array: &VariantArray) -> Vec<Option<String>> {
        let whole = unshred_variant(array).unwrap();
        (0..whole.len())
            .map(|row| {
                whole.is_valid(row).then(|| {
                    let mut text = Vec::new();
                    whole.value(row).to_json(&mut text).unwrap();
                    String::from_utf8(text).unwrap()
                })
            })
            .collect()
    }

    /// The same typed leaves and the same documents as shredding the whole
    /// documents into `layout` gives. A fallback's bytes may differ (a folded
    /// integer is re-encoded at its leaf's width), so fallbacks are compared
    /// by which rows they hold and the documents rendered.
    fn assert_matches_shredding_whole(
        ours: &VariantArray,
        rows: &[Option<&str>],
        layout: &DataType,
        allocator: &mut SlabAllocator,
    ) {
        let reference = shredded(rows, layout, allocator);
        assert_eq!(ours.data_type(), reference.data_type());
        assert_same_leaves(ours.inner(), reference.inner());
        assert_eq!(rendered(ours), rendered(&reference));
    }

    fn assert_same_leaves(ours: &StructArray, reference: &StructArray) {
        assert_eq!(present(ours), present(reference), "struct validity");
        for (field, (our_column, reference_column)) in ours
            .fields()
            .iter()
            .zip(ours.columns().iter().zip(reference.columns()))
        {
            match (
                field.name().as_str(),
                our_column.as_any().downcast_ref::<StructArray>(),
            ) {
                ("value" | "metadata", _) => {
                    assert_eq!(
                        present(our_column.as_ref()),
                        present(reference_column.as_ref()),
                        "{} validity",
                        field.name()
                    )
                }
                (_, Some(object)) => assert_same_leaves(
                    object,
                    reference_column
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .unwrap(),
                ),
                _ => assert_eq!(our_column, reference_column, "{} leaf", field.name()),
            }
        }
    }

    /// Layouts and documents at random: two files of their own layouts,
    /// widened to a union, gathered together and reshredded into an output
    /// layout. Every step comes out as taking its input's documents apart
    /// and shredding them whole would, leaf for leaf.
    #[test]
    fn a_run_of_any_layouts_comes_out_as_shredding_it_whole_would() {
        init_test_free_pool(64);
        let mut allocator = SlabAllocator::new(false);
        let mut seed: u64 = 0x2545F4914F6CDD1D;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let scalar = |pick: u64| match pick % 3 {
            0 => DataType::Int64,
            1 => DataType::Float64,
            _ => DataType::Utf8View,
        };
        let docs_of = |rand: &mut dyn FnMut() -> u64| {
            let mut docs: Vec<Option<String>> = Vec::new();
            for _ in 0..2 + rand() % 5 {
                if rand() % 11 == 0 {
                    docs.push(None);
                    continue;
                }
                let mut fields: Vec<String> = Vec::new();
                for name in ["a", "b", "c"] {
                    match rand() % 6 {
                        0 => continue,
                        1 => fields.push(format!("\"{name}\": {}", rand() % 100)),
                        2 => fields.push(format!("\"{name}\": {}.5", rand() % 100)),
                        3 => fields.push(format!("\"{name}\": \"s{}\"", rand() % 100)),
                        4 => fields.push(format!("\"{name}\": true")),
                        _ => {
                            let mut inner: Vec<String> = Vec::new();
                            for child in ["x", "y"] {
                                match rand() % 4 {
                                    0 => continue,
                                    1 => inner.push(format!("\"{child}\": {}", rand() % 100)),
                                    2 => inner.push(format!("\"{child}\": \"t{}\"", rand() % 100)),
                                    _ => inner.push(format!("\"{child}\": {}.25", rand() % 100)),
                                }
                            }
                            fields.push(format!("\"{name}\": {{{}}}", inner.join(", ")));
                        }
                    }
                }
                docs.push(Some(format!("{{{}}}", fields.join(", "))));
            }
            docs
        };
        let layout_of = |rand: &mut dyn FnMut() -> u64| {
            let mut fields: Vec<(&str, DataType)> = Vec::new();
            for name in ["a", "b", "c"] {
                match rand() % 5 {
                    0 => continue,
                    1 => {
                        let mut inner: Vec<(&str, DataType)> = Vec::new();
                        for child in ["x", "y"] {
                            if rand() % 3 != 0 {
                                inner.push((child, scalar(rand())));
                            }
                        }
                        if !inner.is_empty() {
                            fields.push((name, object(inner)));
                        }
                    }
                    _ => fields.push((name, scalar(rand()))),
                }
            }
            (!fields.is_empty()).then(|| object(fields))
        };

        for case in 0..500u64 {
            let mut rand = || next();
            let first = docs_of(&mut rand);
            let second = docs_of(&mut rand);
            let (Some(first_layout), Some(second_layout), Some(union), Some(output)) = (
                layout_of(&mut rand),
                layout_of(&mut rand),
                layout_of(&mut rand),
                layout_of(&mut rand),
            ) else {
                continue;
            };
            fn text(docs: &[Option<String>]) -> Vec<Option<&str>> {
                docs.iter().map(|doc| doc.as_deref()).collect()
            }
            let (first_rows, second_rows) = (text(&first), text(&second));
            let case = format!(
                "case {case}: {first_layout:?} + {second_layout:?} -> {union:?} -> {output:?} \
                 over {first_rows:?} and {second_rows:?}"
            );

            println!("{case}");
            let mut widened = Vec::new();
            for (rows, layout) in [(&first_rows, &first_layout), (&second_rows, &second_layout)] {
                let file = shredded(rows, layout, &mut allocator);
                let wide = match widen(&file, &union, &mut allocator) {
                    Ok(wide) => wide,
                    Err(error) => panic!("{case}: widen: {error}"),
                };
                let whole = unshred_variant(&file).unwrap();
                let rebuilt = shred::shred_into_slabs(&whole, &union, &mut allocator).unwrap();
                assert_eq!(
                    typed_documents(&wide),
                    typed_documents(&rebuilt),
                    "{case}: widen"
                );
                assert_same_leaves(wide.inner(), rebuilt.inner());
                widened.push(wide);
            }
            let merged = interleaved(&widened, &mut allocator);
            let reshredded = match reshred(&merged, &output, &mut allocator) {
                Ok(reshredded) => reshredded,
                Err(error) => panic!("{case}: reshred: {error}"),
            };

            // What the merge would have come out with had it taken the run's
            // documents apart and shredded them whole: the same column, leaf
            // for leaf.
            let whole = unshred_variant(&merged).unwrap();
            let rebuilt = shred::shred_into_slabs(&whole, &output, &mut allocator).unwrap();
            assert_eq!(reshredded.data_type(), rebuilt.data_type(), "{case}: type");
            assert_eq!(
                typed_documents(&reshredded),
                typed_documents(&rebuilt),
                "{case}: documents"
            );
            assert_same_leaves(reshredded.inner(), rebuilt.inner());
        }
    }

    /// Two files of one layout, widened to it and reshredded into it: the
    /// typed leaves come through as the very same arrays.
    #[test]
    fn an_unchanged_layout_shares_its_leaves() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let layout = object(vec![("id", DataType::Int64), ("name", DataType::Utf8View)]);
        let rows = [
            Some(r#"{"id": 1, "name": "a", "x": true}"#),
            Some(r#"{"id": 2}"#),
        ];
        let file = shredded(&rows, &layout, &mut allocator);

        let widened = widen(&file, &layout, &mut allocator).unwrap();
        let reshredded = reshred(&widened, &layout, &mut allocator).unwrap();

        assert_eq!(widened.data_type(), file.data_type());
        let leaf = |array: &VariantArray| {
            let typed = array
                .typed_value_field()
                .unwrap()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone();
            let pair = typed
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone();
            pair.column_by_name("typed_value").unwrap().clone()
        };
        assert!(Arc::ptr_eq(&leaf(&file), &leaf(&reshredded)));
        assert_matches_shredding_whole(&reshredded, &rows, &layout, &mut allocator);
    }

    /// A path one file did not shred is filled from that file's rows'
    /// leftovers, and the other file's rows keep their typed values.
    #[test]
    fn a_path_one_file_left_in_its_leftover_is_pulled_out() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let both = object(vec![("id", DataType::Int64), ("n", DataType::Int64)]);
        let only_id = object(vec![("id", DataType::Int64)]);
        let a = [
            Some(r#"{"id": 1, "n": 10}"#),
            Some(r#"{"id": 2, "n": 20, "z": "q"}"#),
        ];
        let b = [Some(r#"{"id": 3, "n": 30}"#), Some(r#"{"id": 4}"#), None];

        let widened_a = widen(&shredded(&a, &both, &mut allocator), &both, &mut allocator).unwrap();
        let widened_b = widen(
            &shredded(&b, &only_id, &mut allocator),
            &both,
            &mut allocator,
        )
        .unwrap();
        let merged = interleaved(&[widened_a, widened_b], &mut allocator);
        let reshredded = reshred(&merged, &both, &mut allocator).unwrap();

        assert_matches_shredding_whole(
            &reshredded,
            &round_robin_rows(&[&a, &b]),
            &both,
            &mut allocator,
        );
    }

    /// A path the output drops goes back into the leftover of the rows that
    /// held it, beside whatever the leftover already had.
    #[test]
    fn a_dropped_path_folds_back_into_the_leftover() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let both = object(vec![("id", DataType::Int64), ("n", DataType::Int64)]);
        let only_id = object(vec![("id", DataType::Int64)]);
        let rows = [
            Some(r#"{"id": 1, "n": 10, "z": "q"}"#),
            Some(r#"{"id": 2}"#),
            Some(r#"{"n": "not a number"}"#),
            Some(r#""a bare string""#),
        ];

        let widened = widen(
            &shredded(&rows, &both, &mut allocator),
            &both,
            &mut allocator,
        )
        .unwrap();
        let reshredded = reshred(&widened, &only_id, &mut allocator).unwrap();

        assert_matches_shredding_whole(&reshredded, &rows, &only_id, &mut allocator);
    }

    /// Several dropped children fold together, with each row keeping only
    /// what it held, beside leftover fields and non-conforming values.
    #[test]
    fn several_dropped_paths_fold_back_together() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let wide = object(vec![
            ("id", DataType::Int64),
            ("n", DataType::Int64),
            ("s", DataType::Utf8View),
            ("f", DataType::Float64),
        ]);
        let only_id = object(vec![("id", DataType::Int64)]);
        let rows = [
            Some(r#"{"id": 1, "n": 10, "s": "a", "f": 1.5}"#),
            Some(r#"{"id": 2, "n": 20, "z": [1]}"#),
            Some(r#"{"id": 3, "s": "c", "f": "not a float"}"#),
            Some(r#"{"id": 4}"#),
            None,
            Some(r#"{"n": {"nested": true}, "q": null}"#),
        ];

        let widened = widen(
            &shredded(&rows, &wide, &mut allocator),
            &wide,
            &mut allocator,
        )
        .unwrap();
        let reshredded = reshred(&widened, &only_id, &mut allocator).unwrap();

        assert_matches_shredding_whole(&reshredded, &rows, &only_id, &mut allocator);
    }

    /// Two files that typed a path as different types: widening folds the
    /// type the union did not settle on back into the leftover, and the
    /// output types the path once, with the other rows in its fallback.
    #[test]
    fn a_type_the_union_did_not_settle_on_folds_back() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let as_int = object(vec![("x", DataType::Int64)]);
        let as_text = object(vec![("x", DataType::Utf8View)]);
        let a = [Some(r#"{"x": 1, "k": 1}"#), Some(r#"{"x": 2}"#)];
        let b = [Some(r#"{"x": "one"}"#), Some(r#"{"x": "two", "k": 2}"#)];

        let widened_a = widen(
            &shredded(&a, &as_int, &mut allocator),
            &as_text,
            &mut allocator,
        )
        .unwrap();
        let widened_b = widen(
            &shredded(&b, &as_text, &mut allocator),
            &as_text,
            &mut allocator,
        )
        .unwrap();
        let merged = interleaved(&[widened_a, widened_b], &mut allocator);
        let reshredded = reshred(&merged, &as_text, &mut allocator).unwrap();

        assert_matches_shredding_whole(
            &reshredded,
            &round_robin_rows(&[&a, &b]),
            &as_text,
            &mut allocator,
        );
    }

    /// A change of type in the output itself: the union typed a path as one
    /// type and the output wants another, so the typed rows fold back and the
    /// leftover is shredded into the new type.
    #[test]
    fn a_path_retyped_by_the_output_is_reshredded_from_its_leftover() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let as_int = object(vec![("x", DataType::Int64)]);
        let as_text = object(vec![("x", DataType::Utf8View)]);
        let rows = [
            Some(r#"{"x": 1}"#),
            Some(r#"{"x": "one"}"#),
            Some(r#"{"x": "two"}"#),
        ];

        let widened = widen(
            &shredded(&rows, &as_int, &mut allocator),
            &as_int,
            &mut allocator,
        )
        .unwrap();
        let reshredded = reshred(&widened, &as_text, &mut allocator).unwrap();

        assert_matches_shredding_whole(&reshredded, &rows, &as_text, &mut allocator);
    }

    /// Nested objects move children level by level: a child added or dropped
    /// under `user` touches `user`'s own leftover, not the document's.
    #[test]
    fn a_nested_object_moves_children_at_its_own_level() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let narrow = object(vec![("user", object(vec![("id", DataType::Int64)]))]);
        let wide = object(vec![(
            "user",
            object(vec![("id", DataType::Int64), ("name", DataType::Utf8View)]),
        )]);
        let a = [
            Some(r#"{"user": {"id": 1, "name": "a", "t": 1}}"#),
            Some(r#"{"user": 7}"#),
        ];
        let b = [
            Some(r#"{"user": {"id": 2, "name": "b"}}"#),
            Some(r#"{"other": 1}"#),
        ];

        let widened_a = widen(
            &shredded(&a, &narrow, &mut allocator),
            &wide,
            &mut allocator,
        )
        .unwrap();
        let widened_b = widen(&shredded(&b, &wide, &mut allocator), &wide, &mut allocator).unwrap();
        let merged = interleaved(&[widened_a, widened_b], &mut allocator);
        let widened_out = reshred(&merged, &wide, &mut allocator).unwrap();
        let narrowed_out = reshred(&merged, &narrow, &mut allocator).unwrap();

        let rows = round_robin_rows(&[&a, &b]);
        assert_matches_shredding_whole(&widened_out, &rows, &wide, &mut allocator);
        assert_matches_shredding_whole(&narrowed_out, &rows, &narrow, &mut allocator);
    }

    /// An object where the output wants a scalar, and a scalar where it wants
    /// an object, both fold back whole and are shredded afresh.
    #[test]
    fn a_change_of_shape_folds_the_position_back_whole() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let as_object = object(vec![("p", object(vec![("q", DataType::Int64)]))]);
        let as_scalar = object(vec![("p", DataType::Int64)]);
        let rows = [
            Some(r#"{"p": {"q": 1}}"#),
            Some(r#"{"p": 5}"#),
            Some(r#"{"p": {"q": 2, "r": 3}}"#),
        ];

        let from_object = widen(
            &shredded(&rows, &as_object, &mut allocator),
            &as_object,
            &mut allocator,
        )
        .unwrap();
        let from_scalar = widen(
            &shredded(&rows, &as_scalar, &mut allocator),
            &as_scalar,
            &mut allocator,
        )
        .unwrap();
        let to_scalar = reshred(&from_object, &as_scalar, &mut allocator).unwrap();
        let to_object = reshred(&from_scalar, &as_object, &mut allocator).unwrap();

        assert_matches_shredding_whole(&to_scalar, &rows, &as_scalar, &mut allocator);
        assert_matches_shredding_whole(&to_object, &rows, &as_object, &mut allocator);
    }

    /// A file that never shredded the column widens to all-null leaves with
    /// its documents intact in the leftover, and reshreds like any other.
    #[test]
    fn an_unshredded_file_widens_to_all_null_leaves() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let layout = object(vec![("id", DataType::Int64)]);
        let a = [Some(r#"{"id": 1}"#), None, Some(r#"{"id": "text"}"#)];
        let b = [Some(r#"{"id": 2, "k": 0}"#)];

        let widened_a = widen(&documents(&a), &layout, &mut allocator).unwrap();
        let widened_b = widen(
            &shredded(&b, &layout, &mut allocator),
            &layout,
            &mut allocator,
        )
        .unwrap();
        let merged = interleaved(&[widened_a, widened_b], &mut allocator);
        let reshredded = reshred(&merged, &layout, &mut allocator).unwrap();

        assert_matches_shredding_whole(
            &reshredded,
            &round_robin_rows(&[&a, &b]),
            &layout,
            &mut allocator,
        );
    }

    /// Typing nothing yields the plain pair with every document whole.
    #[test]
    fn typing_nothing_yields_the_plain_pair() {
        init_test_free_pool(16);
        let mut allocator = SlabAllocator::new(false);
        let layout = object(vec![
            ("id", DataType::Int64),
            ("user", object(vec![("n", DataType::Utf8View)])),
        ]);
        let rows = [
            Some(r#"{"id": 1, "user": {"n": "a", "z": 0}, "k": [1, 2]}"#),
            None,
            Some(r#"{"id": 2}"#),
        ];

        let widened = widen(
            &shredded(&rows, &layout, &mut allocator),
            &layout,
            &mut allocator,
        )
        .unwrap();
        let plain = reshred(&widened, &DataType::Struct(Fields::empty()), &mut allocator).unwrap();

        assert!(plain.typed_value_field().is_none());
        assert_eq!(plain.data_type(), &DataType::Struct(column_fields(None)));
        assert_eq!(rendered(&plain), rendered(&documents(&rows)));
    }
}
