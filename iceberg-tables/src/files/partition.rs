//! One partition of a table, under the key every pass names it by.
//!
//! A partition is the files written under one partition spec with one
//! tuple, so an unpartitioned table is a single partition. Its key is the
//! spec's id and then the tuple, in bytes that are equal exactly for the
//! files of the partition and order as the tuples do, and the partition is
//! read back from them.

use std::sync::Arc;

use iceberg::spec::{Literal, PrimitiveLiteral, PrimitiveType, Struct, TableMetadata};
use parquet_engine::{measure_allocation, measure_shared_allocation};

use crate::files::pruning;
use crate::{Error, Result};

/// One partition of a table: the files written under one partition spec
/// with one tuple.
#[derive(Debug)]
pub struct Partition {
    spec_id: i32,
    tuple: Struct,
    key: Arc<[u8]>,
}

impl Partition {
    /// The partition's key, which every pass names it by: bytes equal for
    /// exactly the files of it, the spec's id, then the tuple, each value
    /// in a form whose bytes order as the values do.
    pub fn key(&self) -> &Arc<[u8]> {
        &self.key
    }

    /// The partition spec the tuple is laid out by.
    pub fn spec_id(&self) -> i32 {
        self.spec_id
    }

    pub fn tuple(&self) -> &Struct {
        &self.tuple
    }

    /// The partition of `tuple` under the spec `spec_id`.
    pub(crate) fn new(spec_id: i32, tuple: Struct) -> Self {
        let key = encode_partition(spec_id, &tuple);
        Self::with_key(spec_id, tuple, key)
    }

    /// The partition of `tuple` under the spec `spec_id`, whose key the
    /// caller encoded already ([`encode_partition`]).
    pub(crate) fn with_key(spec_id: i32, tuple: Struct, key: Arc<[u8]>) -> Self {
        Self {
            spec_id,
            tuple,
            key,
        }
    }

    /// An estimate of the memory the partition holds, itself included, as
    /// one shared allocation: its tuple and its key.
    pub(crate) fn estimate_heap_bytes(&self) -> usize {
        let tuple = self.tuple.fields();
        measure_shared_allocation(size_of::<Self>())
            + measure_shared_allocation(self.key.len())
            + measure_allocation(size_of_val(tuple))
            + tuple
                .iter()
                .map(|value| match value {
                    Some(Literal::Primitive(value)) => estimate_literal_heap_bytes(value),
                    _ => 0,
                })
                .sum::<usize>()
    }

    /// The partition `key` names in the table `metadata` describes: the
    /// spec whose id the key starts with, and the tuple the rest of the key
    /// holds, each value read back as the type the spec gives its field. A
    /// key that is none of the table's is refused.
    pub(crate) fn from_key(key: &[u8], metadata: &TableMetadata) -> Result<Self> {
        let unknown = || Error::UnknownPartition {
            table: metadata.location().to_string(),
        };
        let (spec_bytes, mut rest) = key.split_first_chunk::<4>().ok_or_else(unknown)?;
        let spec_id = (u32::from_be_bytes(*spec_bytes) ^ (1 << 31)) as i32;
        let spec = metadata.partition_spec_by_id(spec_id).ok_or_else(unknown)?;
        let mut values = Vec::with_capacity(spec.fields().len());
        for field in spec.fields() {
            // A tuple shorter than its spec ends the key early.
            let Some((is_set, value_bytes)) = rest.split_first() else {
                break;
            };
            rest = value_bytes;
            if *is_set == 0 {
                values.push(None);
                continue;
            }
            let value_type = pruning::find_partition_value_type(metadata.current_schema(), field)
                .ok_or_else(unknown)?;
            let value = decode_value(&mut rest, &value_type).ok_or_else(unknown)?;
            values.push(Some(Literal::Primitive(value)));
        }
        if !rest.is_empty() {
            return Err(unknown());
        }
        Ok(Self {
            spec_id,
            tuple: Struct::from_iter(values),
            key: Arc::from(key),
        })
    }
}

/// What `literal` holds outside its own value: the bytes of a string or a
/// binary.
fn estimate_literal_heap_bytes(literal: &PrimitiveLiteral) -> usize {
    match literal {
        PrimitiveLiteral::String(value) => measure_allocation(value.len()),
        PrimitiveLiteral::Binary(value) => measure_allocation(value.len()),
        _ => 0,
    }
}

/// A partition tuple under its spec's id, as bytes that are equal exactly for
/// equal tuples of one spec and order as the tuples do: a NULL first, an
/// integer by value, a string by its bytes.
pub(crate) fn encode_partition(partition_spec_id: i32, partition: &Struct) -> Arc<[u8]> {
    let mut key = Vec::new();
    key.extend_from_slice(&(partition_spec_id as u32 ^ (1 << 31)).to_be_bytes());
    for value in partition.fields() {
        let Some(Literal::Primitive(value)) = value else {
            key.push(0);
            continue;
        };
        key.push(1);
        encode_value(&mut key, value);
    }
    Arc::from(key)
}

/// Append `value` to `key` in a form whose bytes order as the values of one
/// type do, and that what follows cannot be read as part of.
fn encode_value(key: &mut Vec<u8>, value: &PrimitiveLiteral) {
    match value {
        PrimitiveLiteral::Boolean(value) => key.push(u8::from(*value)),
        PrimitiveLiteral::Int(value) => {
            key.extend_from_slice(&(*value as u32 ^ (1 << 31)).to_be_bytes());
        }
        PrimitiveLiteral::Long(value) => {
            key.extend_from_slice(&(*value as u64 ^ (1 << 63)).to_be_bytes());
        }
        PrimitiveLiteral::Float(value) => {
            key.extend_from_slice(&order_float_bits(u64::from(value.0.to_bits()) << 32));
        }
        PrimitiveLiteral::Double(value) => {
            key.extend_from_slice(&order_float_bits(value.0.to_bits()));
        }
        PrimitiveLiteral::Int128(value) => {
            key.extend_from_slice(&(*value as u128 ^ (1 << 127)).to_be_bytes());
        }
        PrimitiveLiteral::UInt128(value) => key.extend_from_slice(&value.to_be_bytes()),
        // A length after the bytes keeps one value's bytes apart from
        // the next field's, and a prefix ordered before what extends it.
        PrimitiveLiteral::String(value) => extend_with_bytes(key, value.as_bytes()),
        PrimitiveLiteral::Binary(value) => extend_with_bytes(key, value),
        PrimitiveLiteral::AboveMax => key.push(u8::MAX),
        PrimitiveLiteral::BelowMin => key.push(0),
    }
}

/// The bits of a float, sign in the top bit, as bytes that order as the
/// floats do: a negative value's bits are inverted, a positive one's sign bit
/// set.
fn order_float_bits(bits: u64) -> [u8; 8] {
    let ordered = if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    };
    ordered.to_be_bytes()
}

/// Append `bytes` so that what follows cannot be read as part of them: each
/// zero byte escaped, then a terminator below every escaped byte.
fn extend_with_bytes(key: &mut Vec<u8>, bytes: &[u8]) {
    for byte in bytes {
        match byte {
            0 => key.extend_from_slice(&[0, u8::MAX]),
            byte => key.push(*byte),
        }
    }
    key.extend_from_slice(&[0, 0]);
}

/// Take a value of `value_type` off the front of `key`, as
/// [`encode_value`] wrote it; `None` when the bytes are not one.
fn decode_value(key: &mut &[u8], value_type: &PrimitiveType) -> Option<PrimitiveLiteral> {
    fn take<const N: usize>(key: &mut &[u8]) -> Option<[u8; N]> {
        let (taken, rest) = key.split_first_chunk::<N>()?;
        *key = rest;
        Some(*taken)
    }
    Some(match value_type {
        PrimitiveType::Boolean => PrimitiveLiteral::Boolean(take::<1>(key)?[0] != 0),
        PrimitiveType::Int | PrimitiveType::Date => {
            PrimitiveLiteral::Int((u32::from_be_bytes(take(key)?) ^ (1 << 31)) as i32)
        }
        PrimitiveType::Long
        | PrimitiveType::Time
        | PrimitiveType::Timestamp
        | PrimitiveType::Timestamptz
        | PrimitiveType::TimestampNs
        | PrimitiveType::TimestamptzNs => {
            PrimitiveLiteral::Long((u64::from_be_bytes(take(key)?) ^ (1 << 63)) as i64)
        }
        PrimitiveType::Float => {
            let bits = read_float_bits(take(key)?);
            PrimitiveLiteral::Float(f32::from_bits((bits >> 32) as u32).into())
        }
        PrimitiveType::Double => {
            PrimitiveLiteral::Double(f64::from_bits(read_float_bits(take(key)?)).into())
        }
        PrimitiveType::Decimal { .. } => {
            PrimitiveLiteral::Int128((u128::from_be_bytes(take(key)?) ^ (1 << 127)) as i128)
        }
        PrimitiveType::Uuid => PrimitiveLiteral::UInt128(u128::from_be_bytes(take(key)?)),
        PrimitiveType::String => {
            PrimitiveLiteral::String(String::from_utf8(take_escaped_bytes(key)?).ok()?)
        }
        PrimitiveType::Fixed(_) | PrimitiveType::Binary => {
            PrimitiveLiteral::Binary(take_escaped_bytes(key)?)
        }
    })
}

/// The bits of the float [`order_float_bits`] wrote as `ordered`.
fn read_float_bits(ordered: [u8; 8]) -> u64 {
    let ordered = u64::from_be_bytes(ordered);
    if ordered >> 63 == 1 {
        ordered & !(1 << 63)
    } else {
        !ordered
    }
}

/// Take the bytes [`extend_with_bytes`] wrote off the front of `key`, up to
/// and with their terminator; `None` when the key ends before one.
fn take_escaped_bytes(key: &mut &[u8]) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    loop {
        let (byte, rest) = key.split_first()?;
        *key = rest;
        if *byte != 0 {
            bytes.push(*byte);
            continue;
        }
        let (escape, rest) = key.split_first()?;
        *key = rest;
        match escape {
            0 => return Some(bytes),
            &u8::MAX => bytes.push(0),
            _ => return None,
        }
    }
}
