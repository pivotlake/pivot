//! Decodes one projected output column, from its Parquet leaves to the array
//! the batch carries.
//!
//! A [`ColumnDecoder`] owns a [`LeafDecoder`] per leaf column chunk it reads,
//! the field those leaves fold back into, and the transform that finishes the
//! folded array. A plain column reads all of its leaves and needs no transform.
//! A pushed-down variant extract reads only the leaves its path needs, and its
//! transform casts the typed leaf or pulls the path out of the reconstructed
//! variant.

use crate::parquet::reading::decoding::ScanEqualityPredicate;
use crate::parquet::reading::decoding::leaf_decoders;
use crate::parquet::reading::decoding::leaf_decoders::{
    BytesViewDecoder, LeafDecoder, PrimitiveLeafDecoder, decimal_decoder,
};
use crate::parquet::types::leaves::{
    leaf_range, plan_variant_extract, reconstruct_column_from_leaves, try_extract_typed_leaf,
};
use crate::parquet::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::parquet::types::page::DecompressedPage;
use arrow_array::types::{
    BinaryViewType, Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int16Type,
    Int32Type, Int64Type, StringViewType, TimestampMicrosecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, TimeUnit};
use dispatch::VariantExtract;
use dispatch::memory::SlabAllocator;
use parquet_variant::{VariantPath, VariantPathElement};
use parquet_variant_compute::{GetOptions, variant_get};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unsupported column type: {0:?}")]
    UnsupportedColumnType(DataType),
    #[error("{0}")]
    LeafDecoder(#[from] leaf_decoders::Error),
    #[error("{0}")]
    Arrow(#[from] ArrowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Finishes a pushed-down variant extract after its leaves are decoded.
///
/// Plain output columns do not need a transform.
enum OutputTransform {
    /// Casts a directly decoded typed leaf to the requested output type.
    Cast(DataType),
    /// Extracts a path from a reconstructed variant.
    ///
    /// A requested `as_type` produces a scalar. Without one, this produces the
    /// sub-variant at the path.
    Extract {
        path: Arc<[String]>,
        as_type: Option<DataType>,
    },
}

impl OutputTransform {
    fn apply(&self, column: &ArrayRef) -> Result<ArrayRef> {
        match self {
            OutputTransform::Cast(as_type) => Ok(arrow_cast::cast(column, as_type)?),
            OutputTransform::Extract { path, as_type } => {
                Ok(extract_variant_path(column, path, as_type)?)
            }
        }
    }
}

/// Extracts `path` from a reconstructed variant column.
///
/// A requested `as_type` produces a scalar. Without one, this returns the
/// sub-variant at the path.
fn extract_variant_path(
    column: &ArrayRef,
    path: &[String],
    as_type: &Option<DataType>,
) -> Result<ArrayRef, ArrowError> {
    let variant_path: VariantPath<'_> = path
        .iter()
        .map(|segment| VariantPathElement::field(segment.as_str()))
        .collect();
    let target_field = as_type
        .as_ref()
        .map(|data_type| Arc::new(Field::new("item", data_type.clone(), true)));
    variant_get(
        column,
        GetOptions::new_with_path(variant_path).with_as_type(target_field),
    )
}

/// Decodes one output column of a row group's batches.
///
/// The leaf decoders are ordered exactly as
/// [`projected_leaves`](crate::parquet::types::leaves::projected_leaves)
/// resolves this column, which is how a fetched leaf finds the decoder that
/// consumes it.
pub struct ColumnDecoder {
    /// One decoder per leaf column chunk read for this column, in Parquet's
    /// depth-first order.
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
    /// The decoded leaf arrays fold back under this field. It is the output
    /// field for a plain column, and for an extract it is whatever the read
    /// leaves reconstruct: the typed leaf, a pruned variant, or the whole
    /// variant struct.
    pre_transform_field: FieldRef,
    /// Finishes the folded array. `None` for a plain column.
    transform: Option<OutputTransform>,
    /// The field this column contributes to the batch schema.
    output_field: FieldRef,
    /// Whether this column carries a pushed-down equality constant whose
    /// column chunk is sound to prune the row group by (all data pages
    /// dictionary encoded).
    prunable: bool,
}

impl ColumnDecoder {
    /// Builds a decoder that reads all of `column`'s leaves.
    pub fn for_column(
        column: usize,
        leaf_fields: &[FieldRef],
        metadata: &QueryRowGroupMetadata,
        eq_predicates: &[ScanEqualityPredicate],
    ) -> Result<Self> {
        let fields = metadata.get_metadata().schema.fields();
        let file_leaves: Vec<usize> = leaf_range(fields, column).collect();
        let mut decoder = Self::from_leaves(
            &file_leaves,
            fields[column].clone(),
            None,
            fields[column].clone(),
            leaf_fields,
            metadata.columns(),
        )?;
        // A whole-column read answers a comparison on the column itself, never
        // one that reaches into a variant path.
        if let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path.is_empty())
        {
            decoder.install_eq_constant(predicate, &file_leaves, metadata.columns());
        }
        Ok(decoder)
    }

    /// Builds a decoder that reads only what a pushed-down `extract` on variant
    /// `column` needs in this file.
    ///
    /// A scalar extract reads only a complete shredded typed leaf when possible.
    /// It otherwise reconstructs the whole variant before extracting the scalar.
    /// A bare extract reconstructs only the shredded path subtree when possible,
    /// or the whole variant when the path is not shredded.
    pub fn for_extract(
        column: usize,
        extract: &VariantExtract,
        leaf_fields: &[FieldRef],
        metadata: &QueryRowGroupMetadata,
        eq_predicates: &[ScanEqualityPredicate],
    ) -> Result<Self> {
        let fields = metadata.get_metadata().schema.fields();
        let column_name = fields[column].name();
        let create_output_field =
            |data_type: DataType| Arc::new(Field::new(column_name, data_type, true));
        // The read leaves reconstruct `pre_transform_field`, which `transform`
        // then finishes into `output_field`.
        let (file_leaves, pre_transform_field, transform, output_field) = match &extract.as_type {
            Some(as_type) => {
                match try_extract_typed_leaf(fields, metadata, column, &extract.path) {
                    Some(typed_leaf) => {
                        let leaf_type = leaf_fields[typed_leaf].data_type();
                        (
                            vec![typed_leaf],
                            create_output_field(leaf_type.clone()),
                            (*leaf_type != *as_type)
                                .then(|| OutputTransform::Cast(as_type.clone())),
                            create_output_field(as_type.clone()),
                        )
                    }
                    None => (
                        leaf_range(fields, column).collect(),
                        fields[column].clone(),
                        Some(OutputTransform::Extract {
                            path: extract.path.clone().into(),
                            as_type: Some(as_type.clone()),
                        }),
                        create_output_field(as_type.clone()),
                    ),
                }
            }
            None => {
                let (file_leaves, pre_transform_field) =
                    match plan_variant_extract(fields, column, &extract.path) {
                        Some(plan) => (plan.file_leaves, plan.nest_field),
                        None => (leaf_range(fields, column).collect(), fields[column].clone()),
                    };
                // The output type is whatever `variant_get` yields for the path
                // over this variant shape. Resolving it once from an empty array
                // fixes the row group's output schema before any batch is
                // decoded.
                let empty_variant = arrow_array::new_empty_array(pre_transform_field.data_type());
                let output_type = extract_variant_path(&empty_variant, &extract.path, &None)?
                    .data_type()
                    .clone();
                (
                    file_leaves,
                    pre_transform_field,
                    Some(OutputTransform::Extract {
                        path: extract.path.clone().into(),
                        as_type: None,
                    }),
                    create_output_field(output_type),
                )
            }
        };

        let mut decoder = Self::from_leaves(
            &file_leaves,
            pre_transform_field,
            transform,
            output_field,
            leaf_fields,
            metadata.columns(),
        )?;
        // A pushed extract answers a comparison that names the very path it
        // reads, on the same variant column.
        if let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path == extract.path)
        {
            decoder.install_eq_constant(predicate, &file_leaves, metadata.columns());
        }
        Ok(decoder)
    }

    /// Builds the leaf decoders for `file_leaves` and assembles the column
    /// around them. The result carries no equality constant yet.
    fn from_leaves(
        file_leaves: &[usize],
        pre_transform_field: FieldRef,
        transform: Option<OutputTransform>,
        output_field: FieldRef,
        leaf_fields: &[FieldRef],
        column_chunks: &[ColumnChunkMeta],
    ) -> Result<Self> {
        let leaf_decoders = file_leaves
            .iter()
            .map(|&leaf| create_leaf_decoder(leaf_fields[leaf].data_type(), &column_chunks[leaf]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            leaf_decoders,
            pre_transform_field,
            transform,
            output_field,
            prunable: false,
        })
    }

    /// Installs a pushed-down equality `predicate` on this column, and records
    /// whether the row group is then sound to prune by it.
    ///
    /// Both uses of the constant read the column's leaf directly:
    /// [`dict_excludes_eq_constant`](Self::dict_excludes_eq_constant) answers
    /// from that leaf's dictionary, and
    /// [`fast_filter_record_batch`](Self::fast_filter_record_batch) compares the
    /// emitted batch's own array against the dictionary's view of the constant.
    /// A column that casts or reconstructs its leaves emits something other
    /// than the leaf, so it can use neither. The constant therefore goes in only
    /// when the column emits exactly one leaf unchanged.
    ///
    /// It also goes in only when every data page of that chunk is dictionary
    /// encoded. The decoder uses the constant to skip building a dictionary that
    /// excludes it, which is sound only when an excluded dictionary prunes the
    /// whole row group. A PLAIN fallback page could hold the constant even if
    /// the dictionary does not, so such a chunk is still scanned and its
    /// dictionary must be built to decode it.
    ///
    /// A constant whose type does not match the leaf is dropped by the leaf
    /// decoder, which forgoes the pushdown; the query's `Filter` still applies
    /// the comparison.
    fn install_eq_constant(
        &mut self,
        predicate: &ScanEqualityPredicate,
        file_leaves: &[usize],
        column_chunks: &[ColumnChunkMeta],
    ) {
        let [leaf] = file_leaves else { return };
        if self.transform.is_some() || !column_chunks[*leaf].data_pages_all_dictionary {
            return;
        }
        self.leaf_decoders[0].set_eq_constant(&predicate.value);
        self.prunable = true;
    }

    /// Returns the field this column contributes to the batch schema.
    pub fn output_field(&self) -> &FieldRef {
        &self.output_field
    }

    /// Returns how many leaf column chunks this column reads.
    pub fn leaf_count(&self) -> usize {
        self.leaf_decoders.len()
    }

    /// Returns how many rows this column can decode from the pages buffered so
    /// far.
    pub fn available(&self) -> usize {
        self.leaf_decoders
            .iter()
            .map(|leaf| leaf.available())
            .min()
            .expect("a column reads at least one leaf")
    }

    /// Buffers a decompressed page for the leaf at `leaf` within this column.
    pub fn insert_page(
        &mut self,
        leaf: usize,
        page: DecompressedPage,
        allocator: &mut SlabAllocator,
    ) {
        self.leaf_decoders[leaf].insert_page(page, allocator);
    }

    /// Decodes the next `size` rows into this column's output array.
    pub fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let leaf_arrays = self
            .leaf_decoders
            .iter_mut()
            .map(|leaf| leaf.read(allocator, size).map_err(Error::from))
            .collect::<Result<Vec<_>>>()?;
        let column =
            reconstruct_column_from_leaves(&self.pre_transform_field, &mut leaf_arrays.into_iter());
        match &self.transform {
            Some(transform) => transform.apply(&column),
            None => Ok(column),
        }
    }

    /// Whether a pushed-down equality constant on this column is sound to prune
    /// the whole row group by. Only such a column is worth asking
    /// [`dict_excludes_eq_constant`](Self::dict_excludes_eq_constant).
    pub fn is_prunable(&self) -> bool {
        self.prunable
    }

    /// Whether a loaded dictionary is known to exclude this column's
    /// pushed-down equality constant, meaning no row can match.
    pub fn dict_excludes_eq_constant(&self) -> bool {
        self.leaf_decoders
            .iter()
            .any(|leaf| leaf.dict_excludes_eq_constant())
    }

    /// Drops rows that cannot pass this column's pushed-down equality constant,
    /// where `column` is its position in `batch`. Most leaf decoders leave the
    /// batch untouched; dictionary encoded string columns filter it with a
    /// cheap view comparison.
    pub fn fast_filter_record_batch(&self, batch: RecordBatch, column: usize) -> RecordBatch {
        self.leaf_decoders.iter().fold(batch, |batch, leaf| {
            leaf.fast_filter_record_batch(batch, column)
        })
    }
}

/// Creates a leaf decoder for `data_type`.
///
/// The chunk metadata disambiguates a decimal's physical storage.
fn create_leaf_decoder(
    data_type: &DataType,
    chunk: &ColumnChunkMeta,
) -> Result<Box<dyn LeafDecoder>> {
    let max_def_level = chunk.max_def_level;
    macro_rules! primitive {
        ($t:ty) => {
            Box::new(PrimitiveLeafDecoder::<$t>::new(max_def_level)) as Box<dyn LeafDecoder>
        };
    }
    match data_type {
        DataType::UInt8 => Ok(primitive!(UInt8Type)),
        DataType::UInt16 => Ok(primitive!(UInt16Type)),
        DataType::UInt32 => Ok(primitive!(UInt32Type)),
        DataType::UInt64 => Ok(primitive!(UInt64Type)),
        DataType::Int16 => Ok(primitive!(Int16Type)),
        DataType::Int32 => Ok(primitive!(Int32Type)),
        DataType::Int64 => Ok(primitive!(Int64Type)),
        DataType::Date32 => Ok(primitive!(Date32Type)),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Ok(primitive!(TimestampMicrosecondType))
        }
        DataType::Float32 => Ok(primitive!(Float32Type)),
        DataType::Float64 => Ok(primitive!(Float64Type)),
        DataType::Decimal64(precision, scale) => {
            Ok(decimal_decoder::<Decimal64Type>(chunk, *precision, *scale)?)
        }
        DataType::Decimal128(precision, scale) => Ok(decimal_decoder::<Decimal128Type>(
            chunk, *precision, *scale,
        )?),
        // The byte-view decoder matches the leaf's declared string or binary
        // type, so no finishing conversion is needed.
        DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => Ok(Box::new(
            BytesViewDecoder::<StringViewType>::new(max_def_level),
        )),
        DataType::BinaryView | DataType::Binary | DataType::LargeBinary => Ok(Box::new(
            BytesViewDecoder::<BinaryViewType>::new(max_def_level),
        )),
        other => Err(Error::UnsupportedColumnType(other.clone())),
    }
}
