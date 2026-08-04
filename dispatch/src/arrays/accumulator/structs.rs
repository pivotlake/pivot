//! Accumulating an Arrow struct column.
//!
//! A struct holds no values of its own: its children do, and it contributes
//! only the validity saying which rows have a struct at all. So this
//! accumulates through one child accumulator per field, exactly as Parquet
//! stores a group, and a child gets whichever strategy its own type calls for.
//!
//! Slicing a struct slices its children with it, so a child's rows are at the
//! same positions as the struct's own and the same [`SourceSelection`] goes straight
//! down.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{ArrowError, Fields};

use super::chunked::ChunkedColumn;
use super::column::{AppendSource, ColumnAccumulator, SourceSelection};
use super::validity::ValidityMask;
use super::{ValueStorage, create_column_accumulator};
use crate::memory::SlabAllocator;

/// A struct column, accumulated as its children plus its own validity.
pub(super) struct StructColumn {
    fields: Fields,
    children: Vec<Box<dyn ColumnAccumulator>>,
    validity: ValidityMask,
}

impl StructColumn {
    pub(super) fn new(
        fields: &Fields,
        capacity: usize,
        storage: ValueStorage,
        allocator: &mut SlabAllocator,
    ) -> Self {
        Self {
            fields: fields.clone(),
            children: fields
                .iter()
                .map(|field| {
                    create_column_accumulator(field.data_type(), capacity, storage, allocator)
                })
                .collect(),
            validity: ValidityMask::new(capacity),
        }
    }
}

impl ColumnAccumulator for StructColumn {
    fn append(
        &mut self,
        source: AppendSource<'_>,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        match source {
            AppendSource::Batch { column, selection } => {
                let column = column.as_struct();
                self.validity
                    .append(column.nulls(), selection, destination_start);
                for (child, values) in self.children.iter_mut().zip(column.columns()) {
                    child.append(
                        AppendSource::Batch {
                            column: values,
                            selection,
                        },
                        destination_start,
                        allocator,
                    );
                }
            }
            AppendSource::Chunked { column, ids, shift } => {
                let ChunkedColumn::Struct { nulls, children } = column else {
                    unreachable!("a struct accumulator receives a struct chunked column");
                };
                self.validity
                    .append_by_ids(ids, shift, destination_start, |batch| nulls[batch].as_ref());
                for (child, values) in self.children.iter_mut().zip(children) {
                    child.append(
                        AppendSource::Chunked {
                            column: values,
                            ids,
                            shift,
                        },
                        destination_start,
                        allocator,
                    );
                }
            }
        }
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let nulls = self.validity.take(len);
        let columns = self
            .children
            .iter_mut()
            .map(|child| child.take_array(len, allocator))
            .collect::<Result<Vec<_>, _>>()?;
        if columns.is_empty() {
            // A struct with no fields has no child to take its length from.
            return Ok(Arc::new(StructArray::new_empty_fields(len, nulls)));
        }
        Ok(Arc::new(StructArray::new(
            self.fields.clone(),
            columns,
            nulls,
        )))
    }
}
