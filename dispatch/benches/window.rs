//! Benchmarks for the window operator over large inputs.
//!
//! Every scenario runs one window over the lines of customer orders, and
//! feeds the operator exactly the columns that window reads, in memory. The
//! output batches are dropped on the workers that produced them and only
//! their row counts leave the dataflow, so the numbers are the window
//! operator's: no scan in front of it and no aggregate behind it.
//!
//! # The rows
//!
//! [`OrderLines`] generates them, the same rows for every scenario:
//!
//! * An order has 1 to 7 lines, numbered from 1. The lines of an order
//!   arrive next to each other and the orders arrive in key order, the way a
//!   table written in order-key order is scanned.
//! * Each line names one of `rows / 600` suppliers and one of `rows / 30`
//!   parts, drawn uniformly, so a supplier has about 600 lines and a part
//!   about 30.
//! * A line has a quantity of 1 to 50, a price that is the quantity times
//!   its part's unit price, a discount of 0 to 10 percent, and a ship date
//!   within a span of about seven years.
//! * A line has a ship mode (one of seven strings of 3 to 7 bytes), shipping
//!   instructions (one of four strings of 4 to 17 bytes) and a comment (10
//!   to 43 bytes cut from running text at an arbitrary place).
//!
//! # Scenarios
//!
//! * `sum_per_order` - two sums of the discounted price over all lines of
//!   the line's order. No order key, and partitions of a few rows that
//!   arrive together.
//! * `top_3_prices_per_part` - `row_number()` per part by descending price,
//!   keeping the rows numbered up to 3. The window drops the other rows
//!   itself.
//! * `running_sum_and_lag_per_supplier` - a running sum of the quantity and
//!   the previous line's price, per supplier by ship date, order key and
//!   line number: four fixed-width key columns.
//! * `rank_by_revenue_per_supplier` - `rank()` per supplier by descending
//!   discounted price.
//! * `row_number_by_ship_mode_then_date` - `row_number()` per supplier by
//!   ship mode, then ship date: a short string in the middle of the key.
//! * `row_number_by_nullable_line_then_date` - `row_number()` per supplier
//!   by the line number, which is NULL for the first line of an order, then
//!   ship date.
//! * `row_number_by_comment_then_date` - `row_number()` per supplier by
//!   comment, then ship date: a long string in the middle of the key.
//! * `row_number_by_instructions_then_date` - `row_number()` per supplier by
//!   shipping instructions, then ship date: a string that is short in half
//!   the rows and long in the other half, with many ties.
//! * `row_number_by_price_without_partitions` - `row_number()` over all the
//!   rows by price. Without a partition key the window has one bucket, so
//!   every row goes through one merge.
//!
//! # Running
//!
//! ```sh
//! # All scenarios:
//! cargo bench --bench window
//!
//! # A/B across two builds of the operator (run, rebuild, run):
//! cargo bench --bench window -- --save-baseline before
//! cargo bench --bench window -- --baseline before
//! ```
//!
//! Tunables (env): `PIVOT_BENCH_WINDOW_ROWS` (rows per scenario, default 60
//! million), `PIVOT_BENCH_WORKERS` (default: all cores), `PIVOT_BENCH_BUFFERS`
//! (ring slots of 2 MiB each; defaults to 200 bytes per row, 12 GB at the
//! default row count).
//!
//! The ring is one pre-faulted mapping, and it holds the rows the window has
//! bucketed, its sorted runs and its output at once. The generated rows are
//! held outside it, up to 55 bytes per row. Keep the two together under the
//! machine's memory.

use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{ArrayRef, Date32Array, Decimal64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, SamplingMode, Throughput};

use dispatch::{
    BUFFER_SIZE, DataFlowDispatcher, Dispatch, OrderBy, RECORD_BATCH_SIZE, WindowFunction,
    WindowRowLimit, WindowSpec, memory_ctx, values_input,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Ring bytes per row of a scenario. The widest scenario's rows take about 55
/// bytes, and the ring holds them bucketed, as sorted runs and as output.
const RING_BYTES_PER_ROW: usize = 200;

/// Lines per supplier and per part, which set how many suppliers and parts a
/// scenario of a given size has.
const LINES_PER_SUPPLIER: usize = 600;
const LINES_PER_PART: usize = 30;

/// Read an env override, falling back to `default` only when the variable is
/// unset. A set-but-unparseable value aborts rather than silently reverting,
/// so a typo cannot benchmark a different configuration than the one asked
/// for.
fn read_env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be an integer, got {value:?}")),
        Err(_) => default,
    }
}

/// Rows each scenario runs its window over.
fn read_row_count() -> usize {
    let row_count = read_env_usize("PIVOT_BENCH_WINDOW_ROWS", 60_000_000);
    assert!(
        row_count >= LINES_PER_SUPPLIER,
        "PIVOT_BENCH_WINDOW_ROWS must be at least {LINES_PER_SUPPLIER}"
    );
    row_count
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64), so both sides of an A/B see identical rows.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, bound)`.
    #[inline]
    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// The rows: lines of orders, one batch at a time
// ---------------------------------------------------------------------------

/// Generates the lines of orders in order-key order (see the module docs),
/// and holds the lines of the batch generated last as plain values. A
/// scenario turns the values it reads into the batch's columns.
struct OrderLines {
    rng: Rng,
    supplier_count: usize,
    part_count: usize,
    /// The order the next line belongs to, and how many more lines it has.
    order_key: i64,
    lines_left_in_order: usize,
    line_number: i32,

    order_keys: Vec<i64>,
    line_numbers: Vec<i32>,
    supplier_keys: Vec<i64>,
    part_keys: Vec<i64>,
    /// Whole units, 1 to 50.
    quantities: Vec<i64>,
    /// Whole percents, 0 to 10.
    discount_percents: Vec<i64>,
    /// Days since the epoch.
    ship_days: Vec<i32>,
}

impl OrderLines {
    fn new(total_rows: usize) -> Self {
        OrderLines {
            rng: Rng(1),
            supplier_count: total_rows / LINES_PER_SUPPLIER,
            part_count: (total_rows / LINES_PER_PART).max(1),
            order_key: 0,
            lines_left_in_order: 0,
            line_number: 0,
            order_keys: Vec::new(),
            line_numbers: Vec::new(),
            supplier_keys: Vec::new(),
            part_keys: Vec::new(),
            quantities: Vec::new(),
            discount_percents: Vec::new(),
            ship_days: Vec::new(),
        }
    }

    /// Replaces the held lines with the next `row_count` lines.
    fn generate_next_lines(&mut self, row_count: usize) {
        self.order_keys.clear();
        self.line_numbers.clear();
        self.supplier_keys.clear();
        self.part_keys.clear();
        self.quantities.clear();
        self.discount_percents.clear();
        self.ship_days.clear();
        for _ in 0..row_count {
            if self.lines_left_in_order == 0 {
                self.order_key += 1;
                self.lines_left_in_order = 1 + self.rng.below(7);
                self.line_number = 0;
            }
            self.lines_left_in_order -= 1;
            self.line_number += 1;
            self.order_keys.push(self.order_key);
            self.line_numbers.push(self.line_number);
            self.supplier_keys
                .push(1 + self.rng.below(self.supplier_count) as i64);
            self.part_keys
                .push(1 + self.rng.below(self.part_count) as i64);
            self.quantities.push(1 + self.rng.below(50) as i64);
            self.discount_percents.push(self.rng.below(11) as i64);
            self.ship_days.push(8_000 + self.rng.below(2_500) as i32);
        }
    }

    /// The price of each held line in hundredths: its quantity times the unit
    /// price of its part. The unit price is fixed per part and lies between
    /// 900.00 and about 2,100.00, so the lines of one part take at most 50
    /// different prices and some of them tie.
    fn compute_prices(&self) -> impl Iterator<Item = i64> + '_ {
        self.quantities
            .iter()
            .zip(&self.part_keys)
            .map(|(&quantity, &part_key)| {
                let unit_price = 90_000 + (part_key / 10) % 20_001 + 100 * (part_key % 1_000);
                quantity * unit_price
            })
    }

    fn build_order_key_col(&self) -> ArrayRef {
        Arc::new(Int64Array::from(self.order_keys.clone()))
    }

    fn build_supplier_key_col(&self) -> ArrayRef {
        Arc::new(Int64Array::from(self.supplier_keys.clone()))
    }

    fn build_part_key_col(&self) -> ArrayRef {
        Arc::new(Int64Array::from(self.part_keys.clone()))
    }

    fn build_line_number_col(&self) -> ArrayRef {
        Arc::new(Int32Array::from(self.line_numbers.clone()))
    }

    /// The line numbers, with NULL in place of every 1.
    fn build_line_number_or_null_col(&self) -> ArrayRef {
        let line_numbers: Int32Array = self
            .line_numbers
            .iter()
            .map(|&line_number| (line_number != 1).then_some(line_number))
            .collect();
        Arc::new(line_numbers)
    }

    fn build_ship_date_col(&self) -> ArrayRef {
        Arc::new(Date32Array::from(self.ship_days.clone()))
    }

    /// The quantities as a decimal with two fraction digits.
    fn build_quantity_col(&self) -> ArrayRef {
        let hundredths: Vec<i64> = self
            .quantities
            .iter()
            .map(|quantity| quantity * 100)
            .collect();
        Arc::new(
            Decimal64Array::from(hundredths)
                .with_precision_and_scale(15, 2)
                .unwrap(),
        )
    }

    /// The prices as a decimal with two fraction digits.
    fn build_price_col(&self) -> ArrayRef {
        Arc::new(
            Decimal64Array::from(self.compute_prices().collect::<Vec<i64>>())
                .with_precision_and_scale(15, 2)
                .unwrap(),
        )
    }

    /// The discounts as a decimal fraction with two fraction digits: 0.00 to
    /// 0.10.
    fn build_discount_col(&self) -> ArrayRef {
        Arc::new(
            Decimal64Array::from(self.discount_percents.clone())
                .with_precision_and_scale(15, 2)
                .unwrap(),
        )
    }

    /// `price * (1 - discount)` as a decimal with four fraction digits.
    fn build_discounted_price_col(&self) -> ArrayRef {
        let ten_thousandths: Vec<i64> = self
            .compute_prices()
            .zip(&self.discount_percents)
            .map(|(price, &discount_percent)| price * (100 - discount_percent))
            .collect();
        Arc::new(
            Decimal64Array::from(ten_thousandths)
                .with_precision_and_scale(18, 4)
                .unwrap(),
        )
    }
}

/// Draws the values of a string column from a block of bytes: every value is
/// a view of one stretch of the block, which is how a dictionary-encoded
/// column decodes. A value of at most 12 bytes is held inside its view.
struct StringSource {
    rng: Rng,
    block: arrow_buffer::Buffer,
    /// When set, a value is one of these `(offset, length)` stretches. When
    /// not, it is a stretch of 10 to 43 bytes starting anywhere in the block.
    stretches: Option<Vec<(u32, u32)>>,
}

impl StringSource {
    /// Values drawn uniformly from `values`.
    fn new_from_values(seed: u64, values: &[&str]) -> Self {
        let mut bytes = Vec::new();
        let mut stretches = Vec::new();
        for value in values {
            stretches.push((bytes.len() as u32, value.len() as u32));
            bytes.extend_from_slice(value.as_bytes());
        }
        StringSource {
            rng: Rng(seed),
            block: arrow_buffer::Buffer::from_vec(bytes),
            stretches: Some(stretches),
        }
    }

    /// Values of 10 to 43 bytes cut from 4 MiB of running text at arbitrary
    /// places. Two values rarely start with the same bytes.
    fn new_from_running_text(seed: u64) -> Self {
        const WORDS: [&str; 24] = [
            "carefully",
            "furiously",
            "quickly",
            "blithely",
            "slyly",
            "fluffily",
            "final",
            "ironic",
            "regular",
            "pending",
            "express",
            "special",
            "bold",
            "even",
            "deposits",
            "accounts",
            "packages",
            "requests",
            "instructions",
            "platelets",
            "sleep",
            "haggle",
            "above the",
            "across the",
        ];
        let mut rng = Rng(seed);
        let mut text = String::new();
        while text.len() < 4 << 20 {
            text.push_str(WORDS[rng.below(WORDS.len())]);
            text.push(' ');
        }
        StringSource {
            rng,
            block: arrow_buffer::Buffer::from_vec(text.into_bytes()),
            stretches: None,
        }
    }

    fn build_col(&mut self, row_count: usize) -> ArrayRef {
        let mut builder = StringViewBuilder::with_capacity(row_count);
        let block_id = builder.append_block(self.block.clone());
        for _ in 0..row_count {
            let (offset, length) = match &self.stretches {
                Some(stretches) => stretches[self.rng.below(stretches.len())],
                None => {
                    let length = 10 + self.rng.below(34);
                    let offset = self.rng.below(self.block.len() - length);
                    (offset as u32, length as u32)
                }
            };
            builder.try_append_view(block_id, offset, length).unwrap();
        }
        Arc::new(builder.finish())
    }
}

// ---------------------------------------------------------------------------
// Run helper: stream the prebuilt batches through the window and drain.
// ---------------------------------------------------------------------------

/// Runs `spec` over `batches` through one dataflow, dropping every output
/// batch on the worker that produced it. Only per-batch row counts leave the
/// dataflow, so the measurement covers the window alone. The row-count check
/// catches a window that drops or duplicates rows, which would otherwise
/// read as a throughput change.
fn run_window(dispatcher: &DataFlowDispatcher, batches: &[RecordBatch], spec: &WindowSpec) {
    let output_rows_by_batch = values_input(dispatcher, batches.to_vec())
        .record_batches()
        .window(spec.clone())
        .map(|| |batch: RecordBatch| Ok(batch.num_rows()))
        .collect()
        .unwrap();
    let input_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    let output_rows: usize = output_rows_by_batch.iter().sum();
    if spec.row_limit.is_some() {
        assert!(
            0 < output_rows && output_rows < input_rows,
            "the row limit kept {output_rows} of {input_rows} rows"
        );
    } else {
        assert_eq!(output_rows, input_rows, "the window did not emit every row");
    }
}

/// Registers one scenario: `spec` over batches with the columns `fields`
/// names, which `build_columns` makes from each batch of generated lines.
/// The rows are generated only when the scenario is selected, and once for
/// all its iterations.
fn register_scenario(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dispatcher: &DataFlowDispatcher,
    name: &str,
    fields: Vec<Field>,
    mut build_columns: impl FnMut(&OrderLines, usize) -> Vec<ArrayRef>,
    spec: WindowSpec,
) {
    let total_rows = read_row_count();
    let schema = Arc::new(Schema::new(fields));
    let mut generated: Option<Vec<RecordBatch>> = None;
    group.bench_function(name, |bencher| {
        let batches = generated.get_or_insert_with(|| {
            let mut lines = OrderLines::new(total_rows);
            let mut batches = Vec::with_capacity(total_rows.div_ceil(RECORD_BATCH_SIZE));
            let mut rows_left = total_rows;
            while rows_left > 0 {
                let batch_rows = rows_left.min(RECORD_BATCH_SIZE);
                lines.generate_next_lines(batch_rows);
                let columns = build_columns(&lines, batch_rows);
                batches.push(RecordBatch::try_new(schema.clone(), columns).unwrap());
                rows_left -= batch_rows;
            }
            batches
        });
        // Per-iteration setup, not timed: zero the buffers the previous
        // iteration dirtied, so that cost stays out of the measurement.
        bencher.iter_batched(
            || {
                dispatcher.run_on_workers(|| {
                    memory_ctx().zero_dirty_buffers();
                })
            },
            |_| run_window(dispatcher, batches, &spec),
            BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

fn bench_windows(criterion: &mut Criterion, dispatcher: &DataFlowDispatcher) {
    let mut group = criterion.benchmark_group("window");
    group.throughput(Throughput::Elements(read_row_count() as u64));
    // A whole window runs for about a second per iteration: flat sampling
    // keeps every sample at one iteration.
    group.sampling_mode(SamplingMode::Flat);

    let int64 = |name: &str| Field::new(name, DataType::Int64, false);
    let date = |name: &str| Field::new(name, DataType::Date32, false);
    let string = |name: &str| Field::new(name, DataType::Utf8View, false);
    let decimal = |name: &str, precision: u8, scale: i8| {
        Field::new(name, DataType::Decimal64(precision, scale), false)
    };

    register_scenario(
        &mut group,
        dispatcher,
        "sum_per_order",
        vec![
            int64("order_key"),
            decimal("price", 15, 2),
            decimal("discount", 15, 2),
            decimal("discounted_price", 18, 4),
        ],
        |lines, _| {
            vec![
                lines.build_order_key_col(),
                lines.build_price_col(),
                lines.build_discount_col(),
                lines.build_discounted_price_col(),
            ]
        },
        WindowSpec {
            partition_by: vec![0],
            order_by: Vec::new(),
            functions: vec![
                WindowFunction::RunningSum {
                    column: 3,
                    through_tie_group: true,
                },
                WindowFunction::RunningSum {
                    column: 3,
                    through_tie_group: true,
                },
            ],
            row_limit: None,
        },
    );

    register_scenario(
        &mut group,
        dispatcher,
        "top_3_prices_per_part",
        vec![int64("part_key"), decimal("price", 15, 2)],
        |lines, _| vec![lines.build_part_key_col(), lines.build_price_col()],
        WindowSpec {
            partition_by: vec![0],
            order_by: vec![OrderBy::new(1, true, false)],
            functions: vec![WindowFunction::RowNumber],
            row_limit: Some(WindowRowLimit {
                function_index: 0,
                limit: 3,
            }),
        },
    );

    register_scenario(
        &mut group,
        dispatcher,
        "running_sum_and_lag_per_supplier",
        vec![
            decimal("quantity", 15, 2),
            int64("supplier_key"),
            date("ship_date"),
            int64("order_key"),
            Field::new("line_number", DataType::Int32, false),
            decimal("price", 15, 2),
        ],
        |lines, _| {
            vec![
                lines.build_quantity_col(),
                lines.build_supplier_key_col(),
                lines.build_ship_date_col(),
                lines.build_order_key_col(),
                lines.build_line_number_col(),
                lines.build_price_col(),
            ]
        },
        WindowSpec {
            partition_by: vec![1],
            order_by: vec![
                OrderBy::new(2, false, false),
                OrderBy::new(3, false, false),
                OrderBy::new(4, false, false),
            ],
            functions: vec![
                WindowFunction::RunningSum {
                    column: 0,
                    through_tie_group: true,
                },
                WindowFunction::Lag {
                    column: 5,
                    offset: 1,
                },
            ],
            row_limit: None,
        },
    );

    register_scenario(
        &mut group,
        dispatcher,
        "rank_by_revenue_per_supplier",
        vec![
            int64("supplier_key"),
            decimal("price", 15, 2),
            decimal("discount", 15, 2),
            decimal("discounted_price", 18, 4),
        ],
        |lines, _| {
            vec![
                lines.build_supplier_key_col(),
                lines.build_price_col(),
                lines.build_discount_col(),
                lines.build_discounted_price_col(),
            ]
        },
        WindowSpec {
            partition_by: vec![0],
            order_by: vec![OrderBy::new(3, true, false)],
            functions: vec![WindowFunction::Rank],
            row_limit: None,
        },
    );

    register_scenario(
        &mut group,
        dispatcher,
        "row_number_by_nullable_line_then_date",
        vec![
            int64("supplier_key"),
            Field::new("line_number", DataType::Int32, false),
            date("ship_date"),
            Field::new("line_number_or_null", DataType::Int32, true),
        ],
        |lines, _| {
            vec![
                lines.build_supplier_key_col(),
                lines.build_line_number_col(),
                lines.build_ship_date_col(),
                lines.build_line_number_or_null_col(),
            ]
        },
        WindowSpec {
            partition_by: vec![0],
            order_by: vec![OrderBy::new(3, false, false), OrderBy::new(2, false, false)],
            functions: vec![WindowFunction::RowNumber],
            row_limit: None,
        },
    );

    register_scenario(
        &mut group,
        dispatcher,
        "row_number_by_price_without_partitions",
        vec![decimal("price", 15, 2)],
        |lines, _| vec![lines.build_price_col()],
        WindowSpec {
            partition_by: Vec::new(),
            order_by: vec![OrderBy::new(0, false, false)],
            functions: vec![WindowFunction::RowNumber],
            row_limit: None,
        },
    );

    // The three scenarios with a string between the supplier and the ship
    // date differ only in the strings.
    let string_scenarios = [
        (
            "row_number_by_ship_mode_then_date",
            StringSource::new_from_values(
                2,
                &["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"],
            ),
        ),
        (
            "row_number_by_comment_then_date",
            StringSource::new_from_running_text(3),
        ),
        (
            "row_number_by_instructions_then_date",
            StringSource::new_from_values(
                4,
                &[
                    "NONE",
                    "COLLECT COD",
                    "TAKE BACK RETURN",
                    "DELIVER IN PERSON",
                ],
            ),
        ),
    ];
    for (name, mut strings) in string_scenarios {
        register_scenario(
            &mut group,
            dispatcher,
            name,
            vec![int64("supplier_key"), string("text"), date("ship_date")],
            |lines, row_count| {
                vec![
                    lines.build_supplier_key_col(),
                    strings.build_col(row_count),
                    lines.build_ship_date_col(),
                ]
            },
            WindowSpec {
                partition_by: vec![0],
                order_by: vec![OrderBy::new(1, false, false), OrderBy::new(2, false, false)],
                functions: vec![WindowFunction::RowNumber],
                row_limit: None,
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Harness: one shared Dispatch (worker pool) for every scenario.
// ---------------------------------------------------------------------------

fn main() {
    let row_count = read_row_count();
    let workers = read_env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map_or(1, |cores| cores.len()),
    );
    let buffers = read_env_usize(
        "PIVOT_BENCH_BUFFERS",
        row_count * RING_BYTES_PER_ROW / BUFFER_SIZE,
    );
    eprintln!(
        "window benches: {workers} workers, {buffers} ring buffers, {row_count} rows per scenario"
    );

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let dispatcher = dispatch.dispatcher().clone();

    // Default to a handful of samples, each a whole window over every row. A
    // `--sample-size` on the command line still overrides this.
    let mut criterion = Criterion::default().sample_size(10).configure_from_args();
    bench_windows(&mut criterion, &dispatcher);
    criterion.final_summary();

    dispatch.exit();
}
