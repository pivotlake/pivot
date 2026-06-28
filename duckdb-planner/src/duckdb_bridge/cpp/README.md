# C++ Bridge

We use DuckDB as a SQL planner only (no execution). This bridge embeds a
DuckDB instance, feeds it SQL, and hands Rust an opaque `PlanHandle` that owns
DuckDB's resolved `LogicalOperator` tree. The Rust plan builder walks that live
tree directly through accessor functions (`lo_*` for operators, `expr_*` for
expressions) to build its `PlanNode` tree; there is no intermediate
representation. The trick is wiring a Rust catalog into DuckDB so that table
lookups during planning call back into Rust, and the resulting
`Arc<dyn GetDuckDBTypedColumns>` objects survive the round-trip through C++
and come back to Rust attached to the plan.

## How we hack the DuckDB planner

DuckDB has a storage extension system meant for plugging in external storage
backends. We abuse it to inject our own catalog that doesn't store anything —
it just forwards table lookups to Rust.

On context creation (`new_context`):

1. Register a `PivotExtension` storage extension (in `extension.cpp`)
2. Create a `PivotStorageInfo` that holds a pointer to the Rust `CatalogContext`
3. `ATTACH '' AS pv (TYPE pivotdb)` — DuckDB calls our `pivot_catalog_attach`,
   which creates a `PivotCatalog` backed by the Rust catalog
4. `USE pv` — makes it the default catalog so all unqualified table references
   go through our catalog

When DuckDB plans a query and encounters a table name, it walks:

```
DuckDB binder
  -> PivotCatalog::LookupSchema("main")    -- we only have one schema
  -> PivotSchemaCatalogEntry::LookupEntry("users")
       calls Rust FFI: catalog_get_table(ctx, "users")
       Rust calls DuckDBBind::try_bind("users")
       returns CatalogGetTableResult {
           found: true,
           columns: [...],                    // from duckdb_typed_columns()
           table: Box<OptionalTableWrapper>   // wraps Option<Arc<dyn GetDuckDBTypedColumns>>
       }
```

The columns tell DuckDB the schema so it can type-check and plan the query.
The `Box<OptionalTableWrapper>` is the opaque Rust table object riding along.

## How the Rust catalog object travels through C++

The `Arc<dyn GetDuckDBTypedColumns>` can't cross FFI directly (CXX doesn't support trait
objects), so it's wrapped in `OptionalTableWrapper` and boxed. Here's the
full path:

```
Rust DuckDBBind::try_bind()
  returns Arc<dyn GetDuckDBTypedColumns>
    |
    v
catalog_get_table() wraps it:
  Box<OptionalTableWrapper> (contains Option<Arc<dyn GetDuckDBTypedColumns>>)
    |
    v  (crosses FFI as opaque Box in CatalogGetTableResult)
    |
PivotSchemaCatalogEntry::LookupEntry()
  creates PivotTableCatalogEntry, which owns the Box
    |
    v  (stored in PivotStorageInfo::table_entries for the duration of planning)
    |
lo_get_take_table() during the Rust plan walk
  moves Box out of PivotTableCatalogEntry into the Rust-side tables vector
    |
    v
Rust PlannerContext::plan()
  unwraps OptionalTableWrapper -> Arc<dyn GetDuckDBTypedColumns>
  attaches to Input nodes while building the plan
```

The `PivotStorageInfo::table_entries` vector keeps the catalog entries alive
during planning. Because the Rust walk moves the table handles out of the
entries (it runs after `extract_plan` returns), the entries are only cleared by
`ClearTableEntries()` when the `PlanHandle` is dropped, not at the end of
`extract_plan`. By that point the `Box<OptionalTableWrapper>`s have already been
moved out into the Rust tables vector.

## File map

```
bridge.h / bridge.cpp             DuckPlannerContext, extract_plan, plan normalization + accessors
catalog/
  catalog.h / .cpp                PivotCatalog (single-schema catalog)
  schema_entry.h / .cpp           PivotSchemaCatalogEntry (calls Rust on table lookup)
  table_entry.h / .cpp            PivotTableCatalogEntry (holds Box<OptionalTableWrapper>)
storage_info.h / .cpp             PivotStorageInfo (keeps table entries alive during planning)
pivot_config.h / .cpp             Carries Rust CatalogContext pointer into DuckDB config
extension.h / extension.cpp       Registers "pivotdb" storage extension
extension_loader.cpp              Static extension loading at startup
transaction_manager.h / .cpp      No-op transaction manager (planning only, no writes)
common.h                          RUST_NOT_IMPLEMENTED macro
```
