//! Loads newline-delimited JSON into Parquet through pivot's own write pipeline,
//! as one shredded `variant` column.
//!
//! JSONBench ships its dataset as gzipped ndjson, and the benchmark needs it as
//! Parquet that pivot wrote itself, shredding included, since how well the read
//! side does depends on what the write side chose to shred. `INSERT` reads rows
//! from a query, not ndjson from disk, so this is still how the documents get in;
//! the benchmark then `INSERT`s them from here into the table it measures.
//!
//! Each input file is loaded on its own, so peak memory is one file's documents
//! rather than the whole dataset. That also means each output file's shredding is
//! decided from its own rows, which is the same thing a real ingest would do with
//! a flush.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{ArrowError, Schema};
use catalog::parquet::writing::encode_record_batches;
use clap::Parser;
use dispatch::{BUFFER_SIZE, Dispatch, values_input};
use flate2::read::MultiGzDecoder;
use parquet_variant_compute::json_to_variant;

/// Documents per `RecordBatch`. Each item converts on a worker, so this is the
/// unit of parallelism in the pipeline's first stage; a few thousand keeps the
/// batches big enough to be worth a hop and small enough to spread.
const DOCUMENTS_PER_BATCH: usize = 8192;

/// Rows per row group, and row groups per file — the same shape the ingest sink
/// writes, so the benchmark reads files laid out like real ones.
const ROW_GROUP_ROWS: usize = 128 * 1024;
const ROW_GROUPS_PER_FILE: usize = 8;

/// Dispatch's memory ring. Small on purpose: the ring backs the *read* path's
/// file cache, and this only writes, so it needs little more than the pipeline's
/// own buffers. (A server sizes this from total memory; doing that here just
/// prefaults tens of gigabytes and gets the loader OOM-killed.)
const RING_BYTES: usize = 256 * 1024 * 1024;

#[derive(Parser)]
#[command(about = "Load newline-delimited JSON into Parquet as a shredded variant column")]
struct Args {
    /// Input `.json` / `.json.gz` files, or a directory holding them.
    #[arg(long, required = true, num_args = 1..)]
    input: Vec<PathBuf>,

    /// Directory to write the Parquet files into.
    #[arg(long)]
    output: PathBuf,

    /// Name of the variant column holding each document.
    #[arg(long, default_value = "j")]
    column: String,

    /// Worker threads for the write pipeline; defaults to the machine's.
    #[arg(long)]
    workers: Option<usize>,
}

/// One batch of JSON documents, converted on a worker into a single variant
/// column. Parsing to variant is the expensive part, and this is what puts it on
/// the pool rather than on the reader.
struct JsonDocuments {
    lines: Vec<String>,
    column: String,
}

impl JsonDocuments {
    /// Parse the batch's documents into one shredded variant column. A batch is
    /// only ever built from at least one document, so there is no empty case.
    fn to_record_batch(self) -> Result<RecordBatch, ArrowError> {
        let json: ArrayRef = Arc::new(StringArray::from(self.lines));
        let variants = json_to_variant(&json)?;
        let schema = Schema::new(vec![variants.field(&self.column)]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(variants.into_inner())])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let inputs = resolve_inputs(&args.input)?;
    if inputs.is_empty() {
        return Err("no .json or .json.gz files found in the given input paths".into());
    }
    std::fs::create_dir_all(&args.output)?;

    let workers = args.workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let dispatch = Dispatch::spin_up(workers, RING_BYTES / BUFFER_SIZE, None);

    let started = Instant::now();
    let (mut documents, mut files) = (0usize, 0usize);
    for (i, input) in inputs.iter().enumerate() {
        let read = Instant::now();
        let items = read_documents(input, &args.column)?;
        let rows: usize = items.iter().map(|item| item.lines.len()).sum();

        // Parsing JSON into a variant is the expensive part, so each batch of
        // documents converts on a worker rather than on the reader.
        let batches = values_input(dispatch.dispatcher(), items)
            .map_each(|documents: JsonDocuments| {
                documents
                    .to_record_batch()
                    .expect("a batch of documents converts to Arrow")
            })
            .record_batches();
        let encoded = encode_record_batches(
            batches,
            Arc::from([]),
            Arc::from([]),
            ROW_GROUP_ROWS,
            ROW_GROUPS_PER_FILE,
        );
        for file in encoded {
            let path = args.output.join(format!("bluesky-{files:05}.parquet"));
            std::fs::write(&path, &file?.bytes)?;
            files += 1;
        }
        documents += rows;
        println!(
            "[{}/{}] {} — {rows} documents in {:.1?}",
            i + 1,
            inputs.len(),
            input.display(),
            read.elapsed()
        );
    }

    println!(
        "{documents} documents into {files} parquet files in {:.1?}",
        started.elapsed()
    );
    dispatch.exit();
    Ok(())
}

/// Every `.json` / `.json.gz` under the given paths, each of which may be a file
/// or a directory of them, in a stable order so two loads of the same input lay
/// out the same.
fn resolve_inputs(paths: &[PathBuf]) -> std::io::Result<Vec<PathBuf>> {
    let mut inputs = Vec::new();
    for path in paths {
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?.path();
                if is_json(&entry) {
                    inputs.push(entry);
                }
            }
        } else if is_json(path) {
            inputs.push(path.clone());
        }
    }
    inputs.sort();
    Ok(inputs)
}

fn is_json(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    name.ends_with(".json") || name.ends_with(".json.gz")
}

/// Read one file's documents into batches, decompressing on the fly for a `.gz`
/// so the raw ndjson never lands on disk (the full dataset is several times the
/// size of its download).
fn read_documents(path: &Path, column: &str) -> std::io::Result<Vec<JsonDocuments>> {
    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut items = Vec::new();
    let mut lines = Vec::with_capacity(DOCUMENTS_PER_BATCH);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        lines.push(line);
        if lines.len() == DOCUMENTS_PER_BATCH {
            items.push(JsonDocuments {
                lines: std::mem::take(&mut lines),
                column: column.to_string(),
            });
            lines.reserve(DOCUMENTS_PER_BATCH);
        }
    }
    if !lines.is_empty() {
        items.push(JsonDocuments {
            lines,
            column: column.to_string(),
        });
    }
    Ok(items)
}
