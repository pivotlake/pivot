//! Reversible conversion between an integer's bit pattern and `u64`.
//!
//! Keys built over integer columns (GROUP BY's packed pairs, the join's packed
//! key tuples) widen each value to a `u64` lane. Equal values of one type map
//! to equal lanes and back, so lane equality is value equality and the
//! original value survives a round trip.

/// Reversible conversion between an integer's bit pattern and `u64`, so a key
/// packs losslessly into fixed `u64` lanes and unpacks back to the original
/// value.
pub trait IntBits: Copy {
    fn to_u64(self) -> u64;
    fn from_u64(bits: u64) -> Self;
}

macro_rules! impl_int_bits {
    ($($t:ty => $u:ty),*) => {
        $(impl IntBits for $t {
            #[inline(always)]
            fn to_u64(self) -> u64 { self as $u as u64 }
            #[inline(always)]
            fn from_u64(bits: u64) -> Self { bits as $u as $t }
        })*
    }
}
impl_int_bits!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, u8 => u8, u16 => u16, u32 => u32, u64 => u64);
