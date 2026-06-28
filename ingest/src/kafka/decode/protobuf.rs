//! Protobuf decoder: strip the Confluent frame and its message-index array,
//! compile the registry `.proto` schema into a descriptor pool, decode the
//! message reflectively, and render it as a JSON object (the single
//! `arrow-json` construction path).
//!
//! Limitation: the schema is compiled with [`protox_parse`], which resolves no
//! imports, so v1 supports self-contained `.proto` schemas (the common Confluent
//! case). The message-index path selects the message type within the schema.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use prost_reflect::prost::Message as _;
use prost_reflect::prost_types::{DescriptorProto, FileDescriptorSet};
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor, SerializeOptions};
use serde_json::Value;

use super::{DecodeError, MessageDecoder, SchemaRegistry, confluent};

pub(super) struct ProtobufDecoder {
    registry: Arc<SchemaRegistry>,
    /// Compiled descriptor pools, cached by schema id.
    pools: Mutex<HashMap<u32, DescriptorPool>>,
}

impl ProtobufDecoder {
    pub(super) fn new(registry: Arc<SchemaRegistry>) -> Self {
        Self {
            registry,
            pools: Mutex::new(HashMap::new()),
        }
    }

    /// The descriptor pool for schema `id`, compiling the registry `.proto` on
    /// first use. Returns the pool and the parsed [`FileDescriptorProto`] (needed
    /// to resolve the message-index path to a message name).
    fn pool(&self, id: u32) -> Result<DescriptorPool, DecodeError> {
        if let Some(pool) = self.pools.lock().unwrap().get(&id) {
            return Ok(pool.clone());
        }
        let text = self.registry.schema_text(id)?;
        let file = protox_parse::parse("schema.proto", &text)
            .map_err(|e| DecodeError::Protobuf(format!("compiling .proto schema {id}: {e}")))?;
        let set = FileDescriptorSet { file: vec![file] };
        let pool = DescriptorPool::decode(set.encode_to_vec().as_slice())
            .map_err(|e| DecodeError::Protobuf(format!("building descriptor pool {id}: {e}")))?;
        self.pools.lock().unwrap().insert(id, pool.clone());
        Ok(pool)
    }
}

impl MessageDecoder for ProtobufDecoder {
    fn decode(&self, payload: &[u8]) -> Result<Value, DecodeError> {
        let frame = confluent::parse(payload)?;
        let (indexes, message_bytes) = confluent::strip_message_index(frame.body)?;
        let pool = self.pool(frame.schema_id)?;
        let descriptor = resolve_message(&pool, &indexes)?;
        let message = DynamicMessage::decode(descriptor, message_bytes)
            .map_err(|e| DecodeError::Protobuf(format!("decoding message: {e}")))?;
        // Match the table's column names to the proto field names: keep snake_case
        // (`use_proto_field_name`), emit 64-bit ints as JSON numbers rather than
        // strings (so arrow-json builds Int64/Float64), and keep default-valued
        // fields present so a proto `0`/`""` lands as that value, not null.
        let options = SerializeOptions::new()
            .use_proto_field_name(true)
            .stringify_64_bit_integers(false)
            .skip_default_fields(false);
        let json = message
            .serialize_with_options(serde_json::value::Serializer, &options)
            .map_err(|e| DecodeError::Protobuf(format!("protobuf to json: {e}")))?;
        if json.is_object() {
            Ok(json)
        } else {
            Err(DecodeError::NotAnObject)
        }
    }
}

/// Resolve the message-index path (Confluent's index into the file's message
/// types, descending into nested types) to a [`MessageDescriptor`] in `pool`.
fn resolve_message(
    pool: &DescriptorPool,
    indexes: &[i64],
) -> Result<MessageDescriptor, DecodeError> {
    // The pool was built from exactly one file; walk that file's descriptor
    // proto by the index path to find the message's fully qualified name.
    let file = pool
        .files()
        .next()
        .ok_or_else(|| DecodeError::Protobuf("empty descriptor pool".into()))?;
    let file_proto = file.file_descriptor_proto();
    let package = file_proto.package();

    let mut messages: &[DescriptorProto] = &file_proto.message_type;
    let mut name_path: Vec<&str> = Vec::new();
    let mut chosen: Option<&DescriptorProto> = None;
    for &index in indexes {
        let message = messages
            .get(index as usize)
            .ok_or_else(|| DecodeError::Protobuf(format!("message index {index} out of range")))?;
        name_path.push(message.name());
        messages = &message.nested_type;
        chosen = Some(message);
    }
    chosen.ok_or_else(|| DecodeError::Protobuf("empty message-index path".into()))?;

    let full_name = if package.is_empty() {
        name_path.join(".")
    } else {
        format!("{package}.{}", name_path.join("."))
    };
    pool.get_message_by_name(&full_name)
        .ok_or_else(|| DecodeError::Protobuf(format!("message `{full_name}` not in schema")))
}
