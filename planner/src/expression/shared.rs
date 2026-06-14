//! Helpers shared by more than one expression's `compile` impl. Currently the
//! type-coercing wrapper around arrow's comparison kernels, used by
//! [`Compare`](super::Compare), [`Between`](super::Between), and
//! [`InList`](super::InList).

use arrow_array::{BooleanArray, Datum, Scalar};
use arrow_schema::{ArrowError, DataType};

/// Signature shared by arrow's scalar comparison kernels.
pub(crate) type CmpKernel =
    fn(&dyn Datum, &dyn Datum) -> std::result::Result<BooleanArray, ArrowError>;

/// Run a comparison kernel, coercing both operands to Int64 when their data
/// types differ. The declared logical type (e.g. DATE) and the physical parquet
/// array (e.g. UInt16 day counts) can diverge, and arrow's kernels require
/// matching types; Int64 is a safe common type for every integer/date/timestamp
/// column we compare.
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
        let lc = arrow::compute::cast(la, &DataType::Int64).unwrap();
        let rc = arrow::compute::cast(ra, &DataType::Int64).unwrap();
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
