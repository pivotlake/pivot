//! End-to-end ClickBench harness over the pivot SQL stack
//! (`planner` + `catalog` + `dispatch`).
//!
//! Bootstraps a [`ParquetCatalog`], registers `hits` against the parquet
//! directory in `SOURCE_DIRECTORY` via `CREATE TABLE`, then runs a fixed
//! matrix of queries 5 times each, printing the total, planning, compilation,
//! and execution time per iteration.
//!
//! # Environment variables
//!
//! - `SOURCE_DIRECTORY` (required) — directory containing the ClickBench
//!   `hits_*.parquet` files.
//! - `WORKER_COUNT` — worker threads (default: number of CPU cores).
//! - `ITERATIONS` — how many times to run each query (default: 5).

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::util::pretty::pretty_format_batches;
use arrow_array::RecordBatch;
use catalog::ParquetCatalog;
use planner::Planner;
use planner::catalog::Catalog;

const QUERIES: &[(&str, &str)] = &[
    // ("count_star", "SELECT COUNT(*) FROM hits"),
    // (
    //     "w",
    //     "SELECT UserID FROM hits WHERE UserID = 435090932899640449;",
    // ),
    (
        "7",
        "SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';",
    ),
    // (
    //     "string",
    //     "SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;;"
    //     )
    // // (
    //     "#another one",
    //     "SELECT AdvEngineID, COUNT() FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT() DESC"
    //     )
    // (
    //     "url_contains_google",
    //     "SELECT COUNT(*) FROM hits WHERE contains(URL, 'google')",
    // ),
    // (
    //     "top_urls",
    //     "SELECT URL, COUNT(*) AS c FROM hits \
    //      GROUP BY URL \
    //      ORDER BY c DESC \
    //      LIMIT 10",
    // ),
];

fn main() {
    let source_directory = std::env::var("SOURCE_DIRECTORY")
        .expect("SOURCE_DIRECTORY must point at the ClickBench parquet directory");
    let workers: usize = std::env::var("WORKER_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| core_affinity::get_core_ids().unwrap().len());
    let iterations: usize = std::env::var("ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    dispatch::init(workers);

    let catalog: Arc<dyn Catalog> = Arc::new(ParquetCatalog::new());
    let mut planner = Planner::new(catalog);

    let create_sql = create_table_sql(&source_directory);
    run_once(&mut planner, &create_sql).expect("CREATE TABLE failed");

    println!("workers: {workers}, iterations: {iterations}, source: {source_directory}\n");

    for (name, sql) in QUERIES {
        println!("=== {name} ===");
        println!("    {sql}");
        for i in 1..=iterations {
            match run_once(&mut planner, sql) {
                Ok((timings, batches)) => {
                    println!("    [{i}/{iterations}] {timings}");
                    if i == 1 {
                        //print_result(&batches);
                    }
                }
                Err(err) => {
                    println!("    [{i}/{iterations}] ERROR: {err}");
                    break;
                }
            }
            // Cooldown between iterations so file caches / allocator state
            // settle before the next timing.
            if i != iterations {
                std::thread::sleep(Duration::from_millis(12));
            }
        }
        println!();
    }
}

/// Pretty-print the query result. Indented to stay grouped under the query
/// header and timing lines.
fn print_result(batches: &[RecordBatch]) {
    if batches.is_empty() {
        println!("    (no rows)");
        return;
    }
    match pretty_format_batches(batches) {
        Ok(formatted) => {
            for line in formatted.to_string().lines() {
                println!("    {line}");
            }
        }
        Err(err) => println!("    (failed to format result: {err})"),
    }
}

#[derive(Default)]
struct Timings {
    plan: Duration,
    compile: Duration,
    execute: Duration,
}

impl Timings {
    fn total(&self) -> Duration {
        self.plan + self.compile + self.execute
    }
}

impl std::fmt::Display for Timings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total {:>7.2}ms (plan {:>6.2}ms, compile {:>6.2}ms, execute {:>7.2}ms)",
            self.total().as_secs_f64() * 1000.0,
            self.plan.as_secs_f64() * 1000.0,
            self.compile.as_secs_f64() * 1000.0,
            self.execute.as_secs_f64() * 1000.0,
        )
    }
}

fn run_once(planner: &mut Planner, sql: &str) -> Result<(Timings, Vec<RecordBatch>), String> {
    let mut t = Timings::default();

    let start = Instant::now();
    // println!("start  time: {:?}", SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_millis());
    let plan = planner.plan(sql).map_err(|e| format!("plan: {e}"))?;
    t.plan = start.elapsed();

    let start = Instant::now();
    let spec = plan.compile().map_err(|e| format!("compile: {e}"))?;
    t.compile = start.elapsed();

    let start = Instant::now();
    let batches = spec.collect();
    t.execute = start.elapsed();

    Ok((t, batches.unwrap()))
}

/// `CREATE TABLE hits` with all 105 ClickBench columns, mapped onto the type
/// set the planner currently understands. Date/timestamp columns are declared
/// as `BIGINT` so the DDL parses; queries against them won't work until the
/// planner gets `Date`/`Timestamp` variants.
fn create_table_sql(path: &str) -> String {
    format!(
        r#"CREATE TABLE hits (
    WatchID                BIGINT,
    JavaEnable             SMALLINT,
    Title                  VARCHAR,
    GoodEvent              SMALLINT,
    EventTime              BIGINT,
    EventDate              BIGINT,
    CounterID              INTEGER,
    ClientIP               INTEGER,
    RegionID               INTEGER,
    UserID                 BIGINT,
    CounterClass           SMALLINT,
    OS                     SMALLINT,
    UserAgent              SMALLINT,
    URL                    VARCHAR,
    Referer                VARCHAR,
    IsRefresh              SMALLINT,
    RefererCategoryID      SMALLINT,
    RefererRegionID        INTEGER,
    URLCategoryID          SMALLINT,
    URLRegionID            INTEGER,
    ResolutionWidth        SMALLINT,
    ResolutionHeight       SMALLINT,
    ResolutionDepth        SMALLINT,
    FlashMajor             SMALLINT,
    FlashMinor             SMALLINT,
    FlashMinor2            VARCHAR,
    NetMajor               SMALLINT,
    NetMinor               SMALLINT,
    UserAgentMajor         SMALLINT,
    UserAgentMinor         VARCHAR,
    CookieEnable           SMALLINT,
    JavascriptEnable       SMALLINT,
    IsMobile               SMALLINT,
    MobilePhone            SMALLINT,
    MobilePhoneModel       VARCHAR,
    Params                 VARCHAR,
    IPNetworkID            INTEGER,
    TraficSourceID         SMALLINT,
    SearchEngineID         SMALLINT,
    SearchPhrase           VARCHAR,
    AdvEngineID            SMALLINT,
    IsArtifical            SMALLINT,
    WindowClientWidth      SMALLINT,
    WindowClientHeight     SMALLINT,
    ClientTimeZone         SMALLINT,
    ClientEventTime        BIGINT,
    SilverlightVersion1    SMALLINT,
    SilverlightVersion2    SMALLINT,
    SilverlightVersion3    INTEGER,
    SilverlightVersion4    SMALLINT,
    PageCharset            VARCHAR,
    CodeVersion            INTEGER,
    IsLink                 SMALLINT,
    IsDownload             SMALLINT,
    IsNotBounce            SMALLINT,
    FUniqID                BIGINT,
    OriginalURL            VARCHAR,
    HID                    INTEGER,
    IsOldCounter           SMALLINT,
    IsEvent                SMALLINT,
    IsParameter            SMALLINT,
    DontCountHits          SMALLINT,
    WithHash               SMALLINT,
    HitColor               VARCHAR,
    LocalEventTime         BIGINT,
    Age                    SMALLINT,
    Sex                    SMALLINT,
    Income                 SMALLINT,
    Interests              SMALLINT,
    Robotness              SMALLINT,
    RemoteIP               INTEGER,
    WindowName             INTEGER,
    OpenerName             INTEGER,
    HistoryLength          SMALLINT,
    BrowserLanguage        VARCHAR,
    BrowserCountry         VARCHAR,
    SocialNetwork          VARCHAR,
    SocialAction           VARCHAR,
    HTTPError              SMALLINT,
    SendTiming             INTEGER,
    DNSTiming              INTEGER,
    ConnectTiming          INTEGER,
    ResponseStartTiming    INTEGER,
    ResponseEndTiming      INTEGER,
    FetchTiming            INTEGER,
    SocialSourceNetworkID  SMALLINT,
    SocialSourcePage       VARCHAR,
    ParamPrice             BIGINT,
    ParamOrderID           VARCHAR,
    ParamCurrency          VARCHAR,
    ParamCurrencyID        SMALLINT,
    OpenstatServiceName    VARCHAR,
    OpenstatCampaignID     VARCHAR,
    OpenstatAdID           VARCHAR,
    OpenstatSourceID       VARCHAR,
    UTMSource              VARCHAR,
    UTMMedium              VARCHAR,
    UTMCampaign            VARCHAR,
    UTMContent             VARCHAR,
    UTMTerm                VARCHAR,
    FromTag                VARCHAR,
    HasGCLID               SMALLINT,
    RefererHash            BIGINT,
    URLHash                BIGINT,
    CLID                   INTEGER
) WITH (path = '{path}')"#
    )
}