//! Helpers shared by more than one expression's `compile` impl. Currently the
//! type-coercing wrapper around arrow's comparison kernels, used by
//! [`Compare`](super::Compare), [`Between`](super::Between), and
//! [`InList`](super::InList).

use arrow_array::{BooleanArray, Datum, Scalar};
use arrow_schema::{ArrowError, DataType};

/// Signature shared by arrow's scalar comparison kernels.
pub(crate) type CmpKernel =
    fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, ArrowError>;

/// Run a comparison kernel, coercing both operands to a common type when their
/// data types differ. The declared logical type (e.g. DATE) and the physical
/// parquet array (e.g. UInt16 day counts) can diverge, and arrow's kernels
/// require matching types.
///
/// The common type must *preserve values*: unconditionally casting to Int64
/// would truncate a fractional operand (`a >= 1.5` would compare against `1`),
/// so we promote to Float64 whenever either side is non-integer and only fall
/// back to Int64 for genuinely integer operands.
pub(crate) fn compare_coerced(
    left: &dyn Datum,
    right: &dyn Datum,
    kernel: CmpKernel,
) -> BooleanArray {
    let (la, l_scalar) = left.get();
    let (ra, r_scalar) = right.get();
    if la.data_type() == ra.data_type() {
        kernel(left, right).unwrap()
    } else {
        let common = if la.data_type().is_integer() && ra.data_type().is_integer() {
            DataType::Int64
        } else {
            DataType::Float64
        };
        let lc = arrow::compute::cast(la, &common).unwrap();
        let rc = arrow::compute::cast(ra, &common).unwrap();
        let ld: Box<dyn Datum> = if l_scalar {
            Box::new(Scalar::new(lc))
        } else {
            Box::new(lc)
        };
        let rd: Box<dyn Datum> = if r_scalar {
            Box::new(Scalar::new(rc))
        } else {
            Box::new(rc)
        };
        kernel(ld.as_ref(), rd.as_ref()).unwrap()
    }
}
