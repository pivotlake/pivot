use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{Date32Builder, Int32Builder, Int64Builder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Decimal64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Debug)]
struct Args {
    input_dir: PathBuf,
    output: PathBuf,
    batch_rows: usize,
    row_group_rows: usize,
}

#[derive(Debug)]
struct Order {
    custkey: i64,
    orderstatus: String,
    totalprice_cents: i64,
    orderdate: i32,
    orderpriority: String,
    clerk: String,
    shippriority: i32,
    comment: String,
}

#[derive(Debug)]
struct Customer {
    name: String,
    address: String,
    nationkey: i64,
    phone: String,
    acctbal_cents: i64,
    mktsegment: String,
    comment: String,
    nation_name: String,
    regionkey: i64,
    region_name: String,
}

#[derive(Debug)]
struct Part {
    name: String,
    mfgr: String,
    brand: String,
    part_type: String,
    size: i32,
    container: String,
    retailprice_cents: i64,
    comment: String,
}

#[derive(Debug)]
struct Supplier {
    name: String,
    address: String,
    nationkey: i64,
    phone: String,
    acctbal_cents: i64,
    comment: String,
    nation_name: String,
    regionkey: i64,
    region_name: String,
}

#[derive(Debug)]
struct PartSupp {
    availqty: i32,
    supplycost_cents: i64,
    comment: String,
}

#[derive(Debug)]
struct Nation {
    name: String,
    regionkey: i64,
    region_name: String,
}

#[derive(Debug)]
struct Region {
    name: String,
}

enum ColumnBuilder {
    I64(Int64Builder),
    I32(Int32Builder),
    Date(Date32Builder),
    Str(StringBuilder),
}

enum Value<'a> {
    I64(i64),
    I32(i32),
    Date(i32),
    Str(&'a str),
}

struct FlatBatchBuilder {
    schema: SchemaRef,
    columns: Vec<ColumnBuilder>,
    rows: usize,
}

impl FlatBatchBuilder {
    fn new(schema: SchemaRef, capacity: usize) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| match field.data_type() {
                DataType::Int64 => ColumnBuilder::I64(Int64Builder::with_capacity(capacity)),
                DataType::Int32 => ColumnBuilder::I32(Int32Builder::with_capacity(capacity)),
                DataType::Date32 => ColumnBuilder::Date(Date32Builder::with_capacity(capacity)),
                DataType::Utf8 => ColumnBuilder::Str(StringBuilder::with_capacity(
                    capacity,
                    capacity.saturating_mul(24),
                )),
                other => panic!("unsupported output type: {other:?}"),
            })
            .collect();
        Self {
            schema,
            columns,
            rows: 0,
        }
    }

    fn append(&mut self, values: &[Value<'_>]) {
        assert_eq!(self.columns.len(), values.len());
        for (column, value) in self.columns.iter_mut().zip(values.iter()) {
            match (column, value) {
                (ColumnBuilder::I64(builder), Value::I64(v)) => builder.append_value(*v),
                (ColumnBuilder::I32(builder), Value::I32(v)) => builder.append_value(*v),
                (ColumnBuilder::Date(builder), Value::Date(v)) => builder.append_value(*v),
                (ColumnBuilder::Str(builder), Value::Str(v)) => builder.append_value(v),
                _ => panic!("output value does not match output schema"),
            }
        }
        self.rows += 1;
    }

    fn is_empty(&self) -> bool {
        self.rows == 0
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn finish(&mut self) -> Result<RecordBatch> {
        let columns = self
            .columns
            .iter_mut()
            .map(|column| match column {
                ColumnBuilder::I64(builder) => Arc::new(builder.finish()) as ArrayRef,
                ColumnBuilder::I32(builder) => Arc::new(builder.finish()) as ArrayRef,
                ColumnBuilder::Date(builder) => Arc::new(builder.finish()) as ArrayRef,
                ColumnBuilder::Str(builder) => Arc::new(builder.finish()) as ArrayRef,
            })
            .collect();
        self.rows = 0;
        Ok(RecordBatch::try_new(self.schema.clone(), columns)?)
    }
}

fn parse_args() -> Result<Args> {
    let mut positional = Vec::new();
    let mut batch_rows = 65_536usize;
    let mut row_group_rows = 1_048_576usize;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--batch-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--batch-rows needs a value"))?;
                batch_rows = value.parse()?;
            }
            "--row-group-rows" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid("--row-group-rows needs a value"))?;
                row_group_rows = value.parse()?;
            }
            "-h" | "--help" => {
                println!(
                    "usage: tpch-flatten <tpchgen-dir> <output.parquet> [--batch-rows N] [--row-group-rows N]"
                );
                std::process::exit(0);
            }
            _ => positional.push(PathBuf::from(arg)),
        }
    }

    if positional.len() != 2 {
        return Err(invalid(
            "usage: tpch-flatten <tpchgen-dir> <output.parquet> [--batch-rows N] [--row-group-rows N]",
        )
        .into());
    }
    if batch_rows == 0 || row_group_rows == 0 {
        return Err(invalid("batch and row-group sizes must be positive").into());
    }

    Ok(Args {
        input_dir: positional.remove(0),
        output: positional.remove(0),
        batch_rows,
        row_group_rows,
    })
}

fn invalid(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}

fn table_path(dir: &Path, table: &str) -> PathBuf {
    dir.join(format!("{table}.parquet"))
}

fn read_table(
    dir: &Path,
    table: &str,
    batch_rows: usize,
) -> Result<impl Iterator<Item = Result<RecordBatch>>> {
    let file = File::open(table_path(dir, table))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(batch_rows)
        .build()?;
    Ok(reader.map(|batch| Ok(batch?)))
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef> {
    let idx = batch.schema().index_of(name)?;
    Ok(batch.column(idx))
}

fn i64_at(array: &ArrayRef, row: usize) -> Result<i64> {
    if array.is_null(row) {
        return Err(invalid("unexpected null BIGINT").into());
    }
    match array.data_type() {
        DataType::Int64 => Ok(array.as_primitive::<Int64Type>().value(row)),
        DataType::Int32 => Ok(i64::from(array.as_primitive::<Int32Type>().value(row))),
        other => Err(invalid(format!("expected integer column, got {other:?}")).into()),
    }
}

fn i32_at(array: &ArrayRef, row: usize) -> Result<i32> {
    if array.is_null(row) {
        return Err(invalid("unexpected null INTEGER").into());
    }
    match array.data_type() {
        DataType::Int32 => Ok(array.as_primitive::<Int32Type>().value(row)),
        DataType::Int64 => Ok(i32::try_from(array.as_primitive::<Int64Type>().value(row))?),
        other => Err(invalid(format!("expected integer column, got {other:?}")).into()),
    }
}

fn date_at(array: &ArrayRef, row: usize) -> Result<i32> {
    if array.is_null(row) {
        return Err(invalid("unexpected null DATE").into());
    }
    match array.data_type() {
        DataType::Date32 => Ok(array.as_primitive::<Date32Type>().value(row)),
        DataType::Int32 => Ok(array.as_primitive::<Int32Type>().value(row)),
        other => Err(invalid(format!("expected DATE column, got {other:?}")).into()),
    }
}

fn decimal_raw_at(array: &ArrayRef, row: usize) -> Result<i64> {
    if array.is_null(row) {
        return Err(invalid("unexpected null DECIMAL").into());
    }
    match array.data_type() {
        DataType::Decimal128(_, _) => Ok(i64::try_from(
            array.as_primitive::<Decimal128Type>().value(row),
        )?),
        DataType::Decimal64(_, _) => Ok(array.as_primitive::<Decimal64Type>().value(row)),
        DataType::Int64 => Ok(array.as_primitive::<Int64Type>().value(row)),
        DataType::Int32 => Ok(i64::from(array.as_primitive::<Int32Type>().value(row))),
        other => Err(invalid(format!("expected DECIMAL/integer column, got {other:?}")).into()),
    }
}

fn string_at<'a>(array: &'a ArrayRef, row: usize) -> Result<&'a str> {
    if array.is_null(row) {
        return Err(invalid("unexpected null VARCHAR").into());
    }
    match array.data_type() {
        DataType::Utf8 => Ok(array.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => Ok(array.as_string::<i64>().value(row)),
        DataType::Utf8View => Ok(array.as_string_view().value(row)),
        other => Err(invalid(format!("expected VARCHAR column, got {other:?}")).into()),
    }
}

fn owned_string(batch: &RecordBatch, name: &str, row: usize) -> Result<String> {
    Ok(string_at(column(batch, name)?, row)?.to_string())
}

fn load_regions(dir: &Path, batch_rows: usize) -> Result<HashMap<i64, Region>> {
    let mut regions = HashMap::new();
    for batch in read_table(dir, "region", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "r_regionkey")?, row)?;
            regions.insert(
                key,
                Region {
                    name: owned_string(&batch, "r_name", row)?,
                },
            );
        }
    }
    Ok(regions)
}

fn load_nations(
    dir: &Path,
    batch_rows: usize,
    regions: &HashMap<i64, Region>,
) -> Result<HashMap<i64, Nation>> {
    let mut nations = HashMap::new();
    for batch in read_table(dir, "nation", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "n_nationkey")?, row)?;
            let regionkey = i64_at(column(&batch, "n_regionkey")?, row)?;
            let region = regions
                .get(&regionkey)
                .ok_or_else(|| invalid(format!("missing region {regionkey} for nation {key}")))?;
            nations.insert(
                key,
                Nation {
                    name: owned_string(&batch, "n_name", row)?,
                    regionkey,
                    region_name: region.name.clone(),
                },
            );
        }
    }
    Ok(nations)
}

fn load_customers(
    dir: &Path,
    batch_rows: usize,
    nations: &HashMap<i64, Nation>,
) -> Result<HashMap<i64, Customer>> {
    let mut customers = HashMap::new();
    for batch in read_table(dir, "customer", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "c_custkey")?, row)?;
            let nationkey = i64_at(column(&batch, "c_nationkey")?, row)?;
            let nation = nations
                .get(&nationkey)
                .ok_or_else(|| invalid(format!("missing nation {nationkey} for customer {key}")))?;
            customers.insert(
                key,
                Customer {
                    name: owned_string(&batch, "c_name", row)?,
                    address: owned_string(&batch, "c_address", row)?,
                    nationkey,
                    phone: owned_string(&batch, "c_phone", row)?,
                    acctbal_cents: decimal_raw_at(column(&batch, "c_acctbal")?, row)?,
                    mktsegment: owned_string(&batch, "c_mktsegment", row)?,
                    comment: owned_string(&batch, "c_comment", row)?,
                    nation_name: nation.name.clone(),
                    regionkey: nation.regionkey,
                    region_name: nation.region_name.clone(),
                },
            );
        }
    }
    Ok(customers)
}

fn load_suppliers(
    dir: &Path,
    batch_rows: usize,
    nations: &HashMap<i64, Nation>,
) -> Result<HashMap<i64, Supplier>> {
    let mut suppliers = HashMap::new();
    for batch in read_table(dir, "supplier", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "s_suppkey")?, row)?;
            let nationkey = i64_at(column(&batch, "s_nationkey")?, row)?;
            let nation = nations
                .get(&nationkey)
                .ok_or_else(|| invalid(format!("missing nation {nationkey} for supplier {key}")))?;
            suppliers.insert(
                key,
                Supplier {
                    name: owned_string(&batch, "s_name", row)?,
                    address: owned_string(&batch, "s_address", row)?,
                    nationkey,
                    phone: owned_string(&batch, "s_phone", row)?,
                    acctbal_cents: decimal_raw_at(column(&batch, "s_acctbal")?, row)?,
                    comment: owned_string(&batch, "s_comment", row)?,
                    nation_name: nation.name.clone(),
                    regionkey: nation.regionkey,
                    region_name: nation.region_name.clone(),
                },
            );
        }
    }
    Ok(suppliers)
}

fn load_parts(dir: &Path, batch_rows: usize) -> Result<HashMap<i64, Part>> {
    let mut parts = HashMap::new();
    for batch in read_table(dir, "part", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "p_partkey")?, row)?;
            parts.insert(
                key,
                Part {
                    name: owned_string(&batch, "p_name", row)?,
                    mfgr: owned_string(&batch, "p_mfgr", row)?,
                    brand: owned_string(&batch, "p_brand", row)?,
                    part_type: owned_string(&batch, "p_type", row)?,
                    size: i32_at(column(&batch, "p_size")?, row)?,
                    container: owned_string(&batch, "p_container", row)?,
                    retailprice_cents: decimal_raw_at(column(&batch, "p_retailprice")?, row)?,
                    comment: owned_string(&batch, "p_comment", row)?,
                },
            );
        }
    }
    Ok(parts)
}

fn load_partsupp(dir: &Path, batch_rows: usize) -> Result<HashMap<(i64, i64), PartSupp>> {
    let mut partsupp = HashMap::new();
    for batch in read_table(dir, "partsupp", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let partkey = i64_at(column(&batch, "ps_partkey")?, row)?;
            let suppkey = i64_at(column(&batch, "ps_suppkey")?, row)?;
            partsupp.insert(
                (partkey, suppkey),
                PartSupp {
                    availqty: i32_at(column(&batch, "ps_availqty")?, row)?,
                    supplycost_cents: decimal_raw_at(column(&batch, "ps_supplycost")?, row)?,
                    comment: owned_string(&batch, "ps_comment", row)?,
                },
            );
        }
    }
    Ok(partsupp)
}

fn load_orders(dir: &Path, batch_rows: usize) -> Result<HashMap<i64, Order>> {
    let mut orders = HashMap::new();
    for batch in read_table(dir, "orders", batch_rows)? {
        let batch = batch?;
        for row in 0..batch.num_rows() {
            let key = i64_at(column(&batch, "o_orderkey")?, row)?;
            orders.insert(
                key,
                Order {
                    custkey: i64_at(column(&batch, "o_custkey")?, row)?,
                    orderstatus: owned_string(&batch, "o_orderstatus", row)?,
                    totalprice_cents: decimal_raw_at(column(&batch, "o_totalprice")?, row)?,
                    orderdate: date_at(column(&batch, "o_orderdate")?, row)?,
                    orderpriority: owned_string(&batch, "o_orderpriority", row)?,
                    clerk: owned_string(&batch, "o_clerk", row)?,
                    shippriority: i32_at(column(&batch, "o_shippriority")?, row)?,
                    comment: owned_string(&batch, "o_comment", row)?,
                },
            );
        }
    }
    Ok(orders)
}

fn flat_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("l_orderkey", DataType::Int64, false),
        Field::new("l_partkey", DataType::Int64, false),
        Field::new("l_suppkey", DataType::Int64, false),
        Field::new("l_linenumber", DataType::Int32, false),
        Field::new("l_quantity", DataType::Int64, false),
        Field::new("l_extendedprice_cents", DataType::Int64, false),
        Field::new("l_discount_pct", DataType::Int64, false),
        Field::new("l_tax_pct", DataType::Int64, false),
        Field::new("l_discount_amount_cents", DataType::Int64, false),
        Field::new("l_disc_price_cents", DataType::Int64, false),
        Field::new("l_charge_cents", DataType::Int64, false),
        Field::new("l_returnflag", DataType::Utf8, false),
        Field::new("l_returnflag_code", DataType::Int32, false),
        Field::new("l_linestatus", DataType::Utf8, false),
        Field::new("l_linestatus_code", DataType::Int32, false),
        Field::new("l_shipdate", DataType::Date32, false),
        Field::new("l_commitdate", DataType::Date32, false),
        Field::new("l_receiptdate", DataType::Date32, false),
        Field::new("l_shipinstruct", DataType::Utf8, false),
        Field::new("l_shipmode", DataType::Utf8, false),
        Field::new("l_comment", DataType::Utf8, false),
        Field::new("o_orderkey", DataType::Int64, false),
        Field::new("o_custkey", DataType::Int64, false),
        Field::new("o_orderstatus", DataType::Utf8, false),
        Field::new("o_totalprice_cents", DataType::Int64, false),
        Field::new("o_orderdate", DataType::Date32, false),
        Field::new("o_orderpriority", DataType::Utf8, false),
        Field::new("o_clerk", DataType::Utf8, false),
        Field::new("o_shippriority", DataType::Int32, false),
        Field::new("o_comment", DataType::Utf8, false),
        Field::new("c_custkey", DataType::Int64, false),
        Field::new("c_name", DataType::Utf8, false),
        Field::new("c_address", DataType::Utf8, false),
        Field::new("c_nationkey", DataType::Int64, false),
        Field::new("c_phone", DataType::Utf8, false),
        Field::new("c_acctbal_cents", DataType::Int64, false),
        Field::new("c_mktsegment", DataType::Utf8, false),
        Field::new("c_comment", DataType::Utf8, false),
        Field::new("c_nation_name", DataType::Utf8, false),
        Field::new("c_regionkey", DataType::Int64, false),
        Field::new("c_region_name", DataType::Utf8, false),
        Field::new("p_partkey", DataType::Int64, false),
        Field::new("p_name", DataType::Utf8, false),
        Field::new("p_mfgr", DataType::Utf8, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("p_container", DataType::Utf8, false),
        Field::new("p_retailprice_cents", DataType::Int64, false),
        Field::new("p_comment", DataType::Utf8, false),
        Field::new("s_suppkey", DataType::Int64, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("s_address", DataType::Utf8, false),
        Field::new("s_nationkey", DataType::Int64, false),
        Field::new("s_phone", DataType::Utf8, false),
        Field::new("s_acctbal_cents", DataType::Int64, false),
        Field::new("s_comment", DataType::Utf8, false),
        Field::new("s_nation_name", DataType::Utf8, false),
        Field::new("s_regionkey", DataType::Int64, false),
        Field::new("s_region_name", DataType::Utf8, false),
        Field::new("ps_availqty", DataType::Int32, false),
        Field::new("ps_supplycost_cents", DataType::Int64, false),
        Field::new("ps_comment", DataType::Utf8, false),
    ]))
}

fn status_code(value: &str) -> i32 {
    value.as_bytes().first().copied().unwrap_or_default() as i32
}

fn flatten_lineitem(
    args: &Args,
    orders: &HashMap<i64, Order>,
    customers: &HashMap<i64, Customer>,
    parts: &HashMap<i64, Part>,
    suppliers: &HashMap<i64, Supplier>,
    partsupp: &HashMap<(i64, i64), PartSupp>,
) -> Result<u64> {
    let schema = flat_schema();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(args.row_group_rows))
        .build();
    let file = File::create(&args.output)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
    let mut builder = FlatBatchBuilder::new(schema, args.batch_rows);
    let mut rows = 0u64;

    for batch in read_table(&args.input_dir, "lineitem", args.batch_rows)? {
        let batch = batch?;
        let l_orderkey = column(&batch, "l_orderkey")?;
        let l_partkey = column(&batch, "l_partkey")?;
        let l_suppkey = column(&batch, "l_suppkey")?;
        let l_linenumber = column(&batch, "l_linenumber")?;
        let l_quantity = column(&batch, "l_quantity")?;
        let l_extendedprice = column(&batch, "l_extendedprice")?;
        let l_discount = column(&batch, "l_discount")?;
        let l_tax = column(&batch, "l_tax")?;
        let l_returnflag = column(&batch, "l_returnflag")?;
        let l_linestatus = column(&batch, "l_linestatus")?;
        let l_shipdate = column(&batch, "l_shipdate")?;
        let l_commitdate = column(&batch, "l_commitdate")?;
        let l_receiptdate = column(&batch, "l_receiptdate")?;
        let l_shipinstruct = column(&batch, "l_shipinstruct")?;
        let l_shipmode = column(&batch, "l_shipmode")?;
        let l_comment = column(&batch, "l_comment")?;

        for row in 0..batch.num_rows() {
            let orderkey = i64_at(l_orderkey, row)?;
            let partkey = i64_at(l_partkey, row)?;
            let suppkey = i64_at(l_suppkey, row)?;
            let order = orders
                .get(&orderkey)
                .ok_or_else(|| invalid(format!("missing order {orderkey}")))?;
            let customer = customers
                .get(&order.custkey)
                .ok_or_else(|| invalid(format!("missing customer {}", order.custkey)))?;
            let part = parts
                .get(&partkey)
                .ok_or_else(|| invalid(format!("missing part {partkey}")))?;
            let supplier = suppliers
                .get(&suppkey)
                .ok_or_else(|| invalid(format!("missing supplier {suppkey}")))?;
            let ps = partsupp
                .get(&(partkey, suppkey))
                .ok_or_else(|| invalid(format!("missing partsupp ({partkey}, {suppkey})")))?;

            let quantity = decimal_raw_at(l_quantity, row)? / 100;
            let extendedprice_cents = decimal_raw_at(l_extendedprice, row)?;
            let discount_pct = decimal_raw_at(l_discount, row)?;
            let tax_pct = decimal_raw_at(l_tax, row)?;
            let discount_amount_cents = extendedprice_cents * discount_pct / 100;
            let disc_price_cents = extendedprice_cents - discount_amount_cents;
            let charge_cents = disc_price_cents * (100 + tax_pct) / 100;
            let returnflag = string_at(l_returnflag, row)?;
            let linestatus = string_at(l_linestatus, row)?;

            builder.append(&[
                Value::I64(orderkey),
                Value::I64(partkey),
                Value::I64(suppkey),
                Value::I32(i32_at(l_linenumber, row)?),
                Value::I64(quantity),
                Value::I64(extendedprice_cents),
                Value::I64(discount_pct),
                Value::I64(tax_pct),
                Value::I64(discount_amount_cents),
                Value::I64(disc_price_cents),
                Value::I64(charge_cents),
                Value::Str(returnflag),
                Value::I32(status_code(returnflag)),
                Value::Str(linestatus),
                Value::I32(status_code(linestatus)),
                Value::Date(date_at(l_shipdate, row)?),
                Value::Date(date_at(l_commitdate, row)?),
                Value::Date(date_at(l_receiptdate, row)?),
                Value::Str(string_at(l_shipinstruct, row)?),
                Value::Str(string_at(l_shipmode, row)?),
                Value::Str(string_at(l_comment, row)?),
                Value::I64(orderkey),
                Value::I64(order.custkey),
                Value::Str(order.orderstatus.as_str()),
                Value::I64(order.totalprice_cents),
                Value::Date(order.orderdate),
                Value::Str(order.orderpriority.as_str()),
                Value::Str(order.clerk.as_str()),
                Value::I32(order.shippriority),
                Value::Str(order.comment.as_str()),
                Value::I64(order.custkey),
                Value::Str(customer.name.as_str()),
                Value::Str(customer.address.as_str()),
                Value::I64(customer.nationkey),
                Value::Str(customer.phone.as_str()),
                Value::I64(customer.acctbal_cents),
                Value::Str(customer.mktsegment.as_str()),
                Value::Str(customer.comment.as_str()),
                Value::Str(customer.nation_name.as_str()),
                Value::I64(customer.regionkey),
                Value::Str(customer.region_name.as_str()),
                Value::I64(partkey),
                Value::Str(part.name.as_str()),
                Value::Str(part.mfgr.as_str()),
                Value::Str(part.brand.as_str()),
                Value::Str(part.part_type.as_str()),
                Value::I32(part.size),
                Value::Str(part.container.as_str()),
                Value::I64(part.retailprice_cents),
                Value::Str(part.comment.as_str()),
                Value::I64(suppkey),
                Value::Str(supplier.name.as_str()),
                Value::Str(supplier.address.as_str()),
                Value::I64(supplier.nationkey),
                Value::Str(supplier.phone.as_str()),
                Value::I64(supplier.acctbal_cents),
                Value::Str(supplier.comment.as_str()),
                Value::Str(supplier.nation_name.as_str()),
                Value::I64(supplier.regionkey),
                Value::Str(supplier.region_name.as_str()),
                Value::I32(ps.availqty),
                Value::I64(ps.supplycost_cents),
                Value::Str(ps.comment.as_str()),
            ]);

            rows += 1;
            if builder.len() >= args.batch_rows {
                writer.write(&builder.finish()?)?;
            }
        }
    }

    if !builder.is_empty() {
        writer.write(&builder.finish()?)?;
    }
    writer.close()?;
    Ok(rows)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    eprintln!(
        "flattening TPCH tables from {} -> {} (batch_rows={}, row_group_rows={}, compression=snappy)",
        args.input_dir.display(),
        args.output.display(),
        args.batch_rows,
        args.row_group_rows
    );

    let regions = load_regions(&args.input_dir, args.batch_rows)?;
    let nations = load_nations(&args.input_dir, args.batch_rows, &regions)?;
    let customers = load_customers(&args.input_dir, args.batch_rows, &nations)?;
    let suppliers = load_suppliers(&args.input_dir, args.batch_rows, &nations)?;
    let parts = load_parts(&args.input_dir, args.batch_rows)?;
    let partsupp = load_partsupp(&args.input_dir, args.batch_rows)?;
    let orders = load_orders(&args.input_dir, args.batch_rows)?;
    let rows = flatten_lineitem(&args, &orders, &customers, &parts, &suppliers, &partsupp)?;

    eprintln!("wrote {rows} wide TPCH rows");
    Ok(())
}
