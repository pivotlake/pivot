# Examples

Working clients for Pivot, kept runnable rather than illustrative: each one is
a program that has been pointed at a real deployment, not a snippet.

## `kafka_arrow_copy_consumer.py`

Streams a Kafka topic into a table with
`COPY <table> FROM STDIN WITH (FORMAT arrow)`, batching rows into Arrow record
batches instead of issuing INSERTs.

```sh
pip install kafka-python-ng psycopg2-binary pyarrow

KAFKA_BOOTSTRAP=localhost:9092 TOPIC=otel-logs \
PG_HOST=localhost TABLE=otel_logs \
BATCH_ROWS=100000 \
python kafka_arrow_copy_consumer.py
```

Every setting is an environment variable, listed at the top of the file. The
table it expects is in the file's header comment.

What the example is really showing:

- **Build the batch column-wise.** Rows go into one python list per column and
  reach Arrow whole. There is no per-row tuple and no statement text, which is
  what made the INSERT version slow.
- **Send the client's own types.** Columns match the table by position (an
  explicit column list also works) and the server casts them, so a client sends
  `pa.string()` without knowing the table's physical layout.
- **Commit after the copy, not before.** Kafka offsets advance once the copy
  returns, so a crash replays the in-flight batch instead of losing it.
- **Batch size has a ceiling.** A single column's buffer must fit the server's
  2MB slab, which for a variable-width column works out at 131,072 rows.
  `BATCH_ROWS=100000` sits under it.

### Where it starts reading

A new consumer group starts at the live edge and ignores the existing backlog
(`OFFSET_RESET=latest`); set `OFFSET_RESET=earliest` to load the topic's whole
history instead.

That setting only decides where a group with **no committed offset** begins. A
group that has run before resumes where it left off, so a restart works through
whatever accumulated while it was down. To begin every boot at the live edge
regardless, set `SKIP_BACKLOG_ON_START=true` — it seeks past the backlog after
joining, which means messages produced while the consumer was stopped are never
loaded.

Run several processes in one consumer group for concurrent copies; Kafka
divides the partitions between them.
