//! The Confluent Schema Registry **wire format** that frames Avro and Protobuf
//! message values:
//!
//! ```text
//! byte 0      magic, always 0x00
//! bytes 1..5  schema id, 4-byte big-endian u32
//! bytes 5..   payload (Protobuf prefixes a message-index array; see below)
//! ```
//!
//! The consumer reads the schema id, fetches + caches the writer schema from the
//! registry, and decodes the remaining bytes.

use super::DecodeError;

/// A parsed Confluent frame: the writer schema id and the bytes after the
/// 5-byte header.
pub(super) struct Frame<'a> {
    pub(super) schema_id: u32,
    pub(super) body: &'a [u8],
}

/// Split the 5-byte Confluent header (magic `0x00` + big-endian schema id) off a
/// message value.
pub(super) fn parse(payload: &[u8]) -> Result<Frame<'_>, DecodeError> {
    if payload.len() < 5 || payload[0] != 0x00 {
        return Err(DecodeError::Framing(
            "expected magic byte 0x00 followed by a 4-byte schema id".into(),
        ));
    }
    let schema_id = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
    Ok(Frame {
        schema_id,
        body: &payload[5..],
    })
}

/// Strip the Protobuf **message-index** array that follows the schema id, and
/// return the message indexes plus the remaining protobuf message bytes.
///
/// The indexes are a length (zig-zag varint) followed by that many zig-zag
/// varints, identifying which message type within the schema this is. The common
/// single-`[0]` case is optimised to one `0x00` byte (length 0).
pub(super) fn strip_message_index(body: &[u8]) -> Result<(Vec<i64>, &[u8]), DecodeError> {
    // A message-index path is tiny; bound it so a corrupt/hostile length can't
    // request a huge (or, if negative, `~usize::MAX`) allocation.
    const MAX_MESSAGE_INDEX_LEN: i64 = 1024;
    let mut pos = 0;
    let len = read_zigzag(body, &mut pos)?;
    if len == 0 {
        return Ok((vec![0], &body[pos..]));
    }
    if len < 0 || len > MAX_MESSAGE_INDEX_LEN {
        return Err(DecodeError::Framing(format!(
            "implausible message-index length {len}"
        )));
    }
    let mut indexes = Vec::with_capacity(len as usize);
    for _ in 0..len {
        indexes.push(read_zigzag(body, &mut pos)?);
    }
    Ok((indexes, &body[pos..]))
}

/// Read one zig-zag-encoded varint (the Confluent / protobuf signed varint
/// encoding) from `buf` at `*pos`, advancing `*pos`.
fn read_zigzag(buf: &[u8], pos: &mut usize) -> Result<i64, DecodeError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let byte = *buf
            .get(*pos)
            .ok_or_else(|| DecodeError::Framing("truncated message-index varint".into()))?;
        *pos += 1;
        result |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(DecodeError::Framing("message-index varint overflow".into()));
        }
    }
    // zig-zag decode
    Ok(((result >> 1) as i64) ^ -((result & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_magic_byte_and_schema_id() {
        let payload = [0x00, 0x00, 0x00, 0x00, 0x07, b'h', b'i'];

        let frame = parse(&payload).unwrap();

        assert_eq!(frame.schema_id, 7);
        assert_eq!(frame.body, b"hi");
    }

    #[test]
    fn rejects_missing_magic_byte() {
        assert!(parse(&[0x01, 0x00, 0x00, 0x00, 0x01]).is_err());
        assert!(parse(&[0x00, 0x00]).is_err());
    }

    #[test]
    fn single_zero_message_index_means_first_message() {
        let (indexes, rest) = strip_message_index(&[0x00, b'x', b'y']).unwrap();

        assert_eq!(indexes, vec![0]);
        assert_eq!(rest, b"xy");
    }
}
