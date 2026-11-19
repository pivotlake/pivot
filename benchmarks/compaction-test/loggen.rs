//! Generates synthetic log rows shaped like a wide production logs table and
//! streams them into a table as Arrow IPC through `COPY ... FROM STDIN`.
//!
//! usage: loggen <connection string> <table> <rows> <seed> <threads>

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, StringArray, TimestampMicrosecondArray, UInt8Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use futures::SinkExt;

const BATCH_ROWS: usize = 8192;
/// 2026-07-12T23:00:00Z in microseconds; every table's rows fall in the hour
/// after it, so the tables' key ranges overlap fully.
const WINDOW_START_MICROS: i64 = 1_783_897_200_000_000;
const WINDOW_MICROS: i64 = 3_600_000_000;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
    fn hex(&mut self, digits: usize) -> String {
        let mut text = String::with_capacity(digits);
        while text.len() < digits {
            text.push_str(&format!("{:016x}", self.next()));
        }
        text.truncate(digits);
        text
    }
    fn uuid(&mut self) -> String {
        let hex = self.hex(32);
        format!(
            "{}-{}-{}-{}-{}",
            &hex[0..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        )
    }
}

struct Pod {
    name: String,
    uid: String,
    node: String,
    host_id: String,
    container_id: String,
}

struct Workload {
    name: String,
    namespace: String,
    container: String,
    cluster: &'static str,
    env: &'static str,
    index_tier: String,
    image: String,
    image_tag: String,
    bucket: u8,
    pods: Vec<Pod>,
}

const WORKLOAD_NAMES: [&str; 24] = [
    "karpenter", "coredns", "ingress-nginx", "checkout", "cart", "payments", "catalog", "search",
    "recommendations", "auth", "gateway", "billing", "notifications", "inventory", "shipping",
    "frontend", "otel-collector", "kafka-consumer", "scheduler", "reporting", "fraud", "ledger",
    "profile", "media",
];
const CLUSTERS: [&str; 3] = ["k8s-demo", "k8s-prod-eu", "k8s-prod-us"];
const ENVS: [&str; 2] = ["aws - eks", "gcp - gke"];
const LEVELS: [(&str, u64); 4] = [("info", 70), ("warn", 12), ("error", 10), ("debug", 8)];
const MESSAGES: [&str; 12] = [
    "Reconciler error",
    "request completed",
    "failed to connect to upstream",
    "cache miss for key",
    "starting reconcile",
    "payment authorization declined",
    "slow query detected",
    "retrying after transient failure",
    "consumer lag above threshold",
    "token refreshed",
    "published event",
    "health check failed",
];
const ERRORS: [&str; 8] = [
    "getting amis, getting AMI queries, failed to discover any AMIs for alias \\\"bottlerocket@v1.25.0\\\"",
    "context deadline exceeded while waiting for response headers",
    "connection reset by peer",
    "rpc error: code = Unavailable desc = upstream connect error or disconnect/reset before headers",
    "pq: deadlock detected, process waits for ShareLock on transaction",
    "dial tcp 10.2.44.17:5432: connect: connection refused",
    "unexpected EOF reading response body",
    "no healthy upstream",
];

fn build_workloads() -> Vec<Workload> {
    // The same workloads for every table, so the tables' keys interleave.
    let mut random = Random(0x5eed);
    let mut workloads = Vec::new();
    for (index, name) in WORKLOAD_NAMES.iter().enumerate() {
        for cluster in CLUSTERS {
            let image_tag = random.hex(64);
            let pods = (0..3 + random.below(6))
                .map(|_| Pod {
                    name: format!("{name}-{}-{}", random.hex(10), random.hex(5)),
                    uid: random.uuid(),
                    node: format!(
                        "ip-10-2-{}-{}.ec2.internal",
                        random.below(250),
                        random.below(250)
                    ),
                    host_id: format!("i-{}", random.hex(17)),
                    container_id: random.hex(64),
                })
                .collect();
            workloads.push(Workload {
                name: name.to_string(),
                namespace: if index % 3 == 0 {
                    name.to_string()
                } else {
                    format!("team-{}", index % 7)
                },
                container: if index % 4 == 0 { "controller" } else { "app" }.to_string(),
                cluster,
                env: ENVS[index % 2],
                index_tier: format!("namespace-{name}"),
                image: format!("public.ecr.aws/{name}/{name}:1.{}.{}@sha256", index % 9, index % 5),
                image_tag,
                bucket: (index as u8).wrapping_mul(37) % 16,
                pods,
            });
        }
    }
    workloads
}

/// The table's columns in order, as `(name, is variant)` with the Arrow type
/// of the plain ones.
fn plain_field(name: &str, data_type: DataType) -> Field {
    Field::new(name, data_type, true)
}

struct Columns {
    guid: Vec<u64>,
    index_tier: Vec<String>,
    round_timestamp: Vec<i64>,
    timestamp: Vec<i64>,
    cluster: Vec<&'static str>,
    pod_name: Vec<String>,
    pod_uid: Vec<String>,
    workload: Vec<String>,
    bucket_workload: Vec<u8>,
    namespace: Vec<String>,
    level: Vec<&'static str>,
    container_name: Vec<String>,
    env: Vec<&'static str>,
    series_id: Vec<u64>,
    pattern_id: Vec<u64>,
    pattern: Vec<&'static str>,
    trace_id: Vec<String>,
    span_id: Vec<String>,
    body: Vec<String>,
    string_attributes: Vec<String>,
    float_attributes: Vec<String>,
    string_tags: Vec<String>,
    node_name: Vec<String>,
    content: Vec<&'static str>,
    body_length: Vec<u64>,
    favorite_attributes: Vec<String>,
    favorite_attributes_json: Vec<String>,
    instance_uid: Vec<u64>,
}

fn strings<S: AsRef<str>>(values: &[S]) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values.iter().map(AsRef::as_ref)))
}

fn constant(value: &str, rows: usize) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(std::iter::repeat_n(value, rows)))
}

fn variant<S: AsRef<str>>(documents: &[S]) -> ArrayRef {
    let documents = strings(documents);
    let variants = parquet_variant_compute::json_to_variant(&documents).unwrap();
    Arc::new(variants.into_inner())
}

fn constant_variant(document: &str, rows: usize) -> ArrayRef {
    let documents = constant(document, rows);
    let variants = parquet_variant_compute::json_to_variant(&documents).unwrap();
    Arc::new(variants.into_inner())
}

fn generate_batch(workloads: &[Workload], random: &mut Random, rows: usize) -> Vec<ArrayRef> {
    let mut columns = Columns {
        guid: Vec::with_capacity(rows),
        index_tier: Vec::with_capacity(rows),
        round_timestamp: Vec::with_capacity(rows),
        timestamp: Vec::with_capacity(rows),
        cluster: Vec::with_capacity(rows),
        pod_name: Vec::with_capacity(rows),
        pod_uid: Vec::with_capacity(rows),
        workload: Vec::with_capacity(rows),
        bucket_workload: Vec::with_capacity(rows),
        namespace: Vec::with_capacity(rows),
        level: Vec::with_capacity(rows),
        container_name: Vec::with_capacity(rows),
        env: Vec::with_capacity(rows),
        series_id: Vec::with_capacity(rows),
        pattern_id: Vec::with_capacity(rows),
        pattern: Vec::with_capacity(rows),
        trace_id: Vec::with_capacity(rows),
        span_id: Vec::with_capacity(rows),
        body: Vec::with_capacity(rows),
        string_attributes: Vec::with_capacity(rows),
        float_attributes: Vec::with_capacity(rows),
        string_tags: Vec::with_capacity(rows),
        node_name: Vec::with_capacity(rows),
        content: Vec::with_capacity(rows),
        body_length: Vec::with_capacity(rows),
        favorite_attributes: Vec::with_capacity(rows),
        favorite_attributes_json: Vec::with_capacity(rows),
        instance_uid: Vec::with_capacity(rows),
    };
    for _ in 0..rows {
        // Squaring skews the choice, so a few workloads log most rows.
        let unit = random.below(1_000_000) as f64 / 1_000_000.0;
        let workload = &workloads[(unit * unit * workloads.len() as f64) as usize];
        let pod = &workload.pods[random.below(workload.pods.len() as u64) as usize];
        let timestamp = WINDOW_START_MICROS + random.below(WINDOW_MICROS as u64) as i64;
        let mut level_pick = random.below(100);
        let mut level = LEVELS[0].0;
        for (name, weight) in LEVELS {
            if level_pick < weight {
                level = name;
                break;
            }
            level_pick -= weight;
        }
        let message_index = random.below(MESSAGES.len() as u64) as usize;
        let message = MESSAGES[message_index];
        let error = ERRORS[random.below(ERRORS.len() as u64) as usize];
        let reconcile_id = random.uuid();
        let duration_ms = random.below(5_000) as f64 / 10.0;
        let object_name = format!("object-{}", random.below(5_000));
        let time = format!("2026-07-12T23:{:02}:{:02}.{:03}Z",
            (timestamp - WINDOW_START_MICROS) / 60_000_000,
            (timestamp / 1_000_000) % 60,
            (timestamp / 1_000) % 1_000);
        let traced = random.below(10) < 3;

        let body = format!(
            "{{\"level\":\"{}\",\"time\":\"{time}\",\"logger\":\"{}\",\"message\":\"{message}\",\"commit\":\"688ea21\",\"controller\":\"{}.status\",\"name\":\"{object_name}\",\"reconcileID\":\"{reconcile_id}\",\"duration_ms\":{duration_ms},\"error\":\"{error}\"}}",
            level.to_uppercase(),
            workload.container,
            workload.name,
        );
        let string_attributes = format!(
            "{{\"commit\":\"688ea21\",\"controller\":\"{}.status\",\"error\":\"{error}\",\"logger\":\"{}\",\"name\":\"{object_name}\",\"reconcileID\":\"{reconcile_id}\",\"time\":\"{time}\"}}",
            workload.name, workload.container,
        );
        let string_tags = format!(
            "{{\"container.id\":\"{}\",\"container.image\":\"{}:{}\",\"container.image.name\":\"{}\",\"container.image.tag\":\"{}\",\"host.id\":\"{}\",\"host.name\":\"{}\",\"host.unix.name\":\"{}\"}}",
            pod.container_id, workload.image, workload.image_tag, workload.image,
            workload.image_tag, pod.host_id, pod.node, pod.node,
        );
        let favorite_attributes = format!(
            "{{\"container.id\":\"{}\",\"container.image.name\":\"{}\",\"error\":\"{error}\",\"host.id\":\"{}\",\"host.name\":\"{}\"}}",
            pod.container_id, workload.image, pod.host_id, pod.node,
        );
        let favorite_attributes_json = format!(
            "{{\"container\":{{\"id\":\"{}\",\"name\":\"{}\",\"tag\":\"{}\"}},\"error\":\"{error}\",\"host\":{{\"id\":\"{}\",\"name\":\"{}\",\"unix\":{{\"name\":\"{}\"}}}}}}",
            pod.container_id, workload.image, workload.image_tag, pod.host_id, pod.node, pod.node,
        );

        columns.guid.push(random.next());
        columns.index_tier.push(workload.index_tier.clone());
        columns.round_timestamp.push(timestamp - timestamp % 60_000_000);
        columns.timestamp.push(timestamp);
        columns.cluster.push(workload.cluster);
        columns.pod_name.push(pod.name.clone());
        columns.pod_uid.push(pod.uid.clone());
        columns.workload.push(workload.name.clone());
        columns.bucket_workload.push(workload.bucket);
        columns.namespace.push(workload.namespace.clone());
        columns.level.push(level);
        columns.container_name.push(workload.container.clone());
        columns.env.push(workload.env);
        columns.series_id.push(pod.uid.len() as u64 * 7919 + pod.name.len() as u64);
        columns.pattern_id.push(2_241_814_617 + message_index as u64);
        columns.pattern.push(if message_index % 2 == 0 { "" } else { message });
        columns.trace_id.push(if traced { random.hex(32) } else { String::new() });
        columns.span_id.push(if traced { random.hex(16) } else { String::new() });
        columns.body_length.push((body.len() + string_attributes.len() + string_tags.len()) as u64);
        columns.body.push(body);
        columns.string_attributes.push(string_attributes);
        columns.float_attributes.push(format!("{{\"duration_ms\":{duration_ms}}}"));
        columns.string_tags.push(string_tags);
        columns.node_name.push(pod.node.clone());
        columns.content.push(message);
        columns.favorite_attributes.push(favorite_attributes);
        columns.favorite_attributes_json.push(favorite_attributes_json);
        columns.instance_uid.push(random.next());
    }

    vec![
        Arc::new(UInt64Array::from(columns.guid)),
        strings(&columns.index_tier),
        Arc::new(TimestampMicrosecondArray::from(columns.round_timestamp)),
        Arc::new(TimestampMicrosecondArray::from(columns.timestamp)),
        strings(&columns.cluster),
        constant("k8s", rows),
        constant("k8s", rows),
        strings(&columns.pod_name),
        strings(&columns.pod_uid),
        strings(&columns.workload),
        Arc::new(UInt8Array::from(columns.bucket_workload)),
        strings(&columns.namespace),
        strings(&columns.level),
        constant("json", rows),
        strings(&columns.container_name),
        strings(&columns.env),
        strings(&columns.pod_name),
        Arc::new(BooleanArray::from(vec![None::<bool>; rows])),
        Arc::new(UInt64Array::from(columns.series_id)),
        Arc::new(UInt64Array::from(columns.pattern_id)),
        strings(&columns.pattern),
        strings(&columns.trace_id),
        strings(&columns.span_id),
        strings(&columns.body),
        constant_variant("[]", rows),
        constant_variant("[]", rows),
        constant_variant("[]", rows),
        constant_variant("[]", rows),
        constant_variant("[]", rows),
        constant_variant("[]", rows),
        variant(&columns.string_attributes),
        variant(&columns.float_attributes),
        variant(&columns.string_tags),
        constant_variant("{}", rows),
        strings(&columns.node_name),
        strings(&columns.node_name),
        strings(&columns.workload),
        strings(&columns.pod_uid),
        constant("", rows),
        constant("", rows),
        strings(&columns.content),
        Arc::new(UInt64Array::from(columns.body_length)),
        constant_variant("{}", rows),
        variant(&columns.favorite_attributes),
        variant(&columns.favorite_attributes_json),
        constant("", rows),
        Arc::new(UInt64Array::from(columns.instance_uid)),
        constant_variant("{}", rows),
        constant_variant("[]", rows),
    ]
}

const COLUMN_NAMES: [&str; 49] = [
    "guid", "index_tier", "round_timestamp", "timestamp", "cluster", "source", "env_type",
    "pod_name", "pod_uid", "workload", "bucket_workload", "namespace", "level", "format",
    "container_name", "env", "instance", "is_pii", "series_id", "pattern_id", "pattern",
    "trace_id", "span_id", "body", "flatten_string_keys", "flatten_string_values",
    "flatten_float_keys", "flatten_float_values", "flatten_bool_keys", "flatten_bool_values",
    "string_attributes", "float_attributes", "string_tags", "float_tags", "node_name", "host",
    "service", "instance_id", "function_name", "lambda_arn", "content", "body_length", "metadata",
    "favorite_attributes", "favorite_attributes_json", "session_id", "instance_uid",
    "sensitive_attributes", "sensitive_keys",
];

fn schema_of(columns: &[ArrayRef]) -> SchemaRef {
    let _ = (plain_field, TimeUnit::Microsecond);
    Arc::new(Schema::new(
        COLUMN_NAMES
            .iter()
            .zip(columns)
            .map(|(name, column)| Field::new(*name, column.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ))
}

#[tokio::main]
async fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let connection = &arguments[1];
    let table = &arguments[2];
    let rows: usize = arguments[3].parse().unwrap();
    let seed: u64 = arguments[4].parse().unwrap();
    let threads: usize = arguments[5].parse().unwrap();

    let workloads = Arc::new(build_workloads());
    let schema = schema_of(&generate_batch(&workloads, &mut Random(1), 1));
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<RecordBatch>(threads * 2);
    let batch_count = rows.div_ceil(BATCH_ROWS);
    for thread in 0..threads {
        let workloads = workloads.clone();
        let sender = sender.clone();
        let schema = schema.clone();
        std::thread::spawn(move || {
            let mut random = Random(seed.wrapping_mul(1_000_003).wrapping_add(thread as u64));
            for batch in (thread..batch_count).step_by(threads) {
                let batch_rows = BATCH_ROWS.min(rows - batch * BATCH_ROWS);
                let columns = generate_batch(&workloads, &mut random, batch_rows);
                let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
                if sender.blocking_send(batch).is_err() {
                    return;
                }
            }
        });
    }
    drop(sender);

    let (client, session) = tokio_postgres::connect(connection, tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(session);
    let sink = client
        .copy_in::<_, bytes::Bytes>(&format!("COPY {table} FROM STDIN WITH (FORMAT arrow)"))
        .await
        .unwrap();
    let mut sink = Box::pin(sink);
    let mut writer = arrow::ipc::writer::StreamWriter::try_new(Vec::new(), &schema).unwrap();
    let started = std::time::Instant::now();
    let mut sent_rows = 0usize;
    let mut sent_bytes = 0usize;
    while let Some(batch) = receiver.recv().await {
        writer.write(&batch).unwrap();
        let bytes = std::mem::take(writer.get_mut());
        sent_rows += batch.num_rows();
        sent_bytes += bytes.len();
        sink.send(bytes::Bytes::from(bytes)).await.unwrap();
        if sent_rows % (BATCH_ROWS * 128) < BATCH_ROWS {
            eprintln!(
                "{table}: {sent_rows} rows, {} MiB of arrow, {:.0} s",
                sent_bytes >> 20,
                started.elapsed().as_secs_f64()
            );
        }
    }
    writer.finish().unwrap();
    let bytes = std::mem::take(writer.get_mut());
    sink.send(bytes::Bytes::from(bytes)).await.unwrap();
    let loaded = sink.as_mut().finish().await.unwrap();
    eprintln!(
        "{table}: loaded {loaded} rows, {} MiB of arrow, {:.0} s",
        sent_bytes >> 20,
        started.elapsed().as_secs_f64()
    );
}
