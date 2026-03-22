//! ClickBench performance harness.
//!
//! Runs queries from the ClickBench suite against a local parquet dataset,
//! printing per-iteration wall-clock times.
//!
//! # Environment variables
//!
//! - `SOURCE_DIRECTORY` (required) — path to directory containing hits.parquet file(s)
//! - `WORKER_COUNT` — number of worker threads (default: number of CPU cores)
//! - `QUERY` — which query/queries to run, comma-separated (default: all)
//! - `QUERY_TEST_COUNT` — number of iterations (default: 20)
//! - `SLEEP` — seconds to sleep between iterations (optional)
//!
//! # Usage
//!
//! ```sh
//! SOURCE_DIRECTORY=/path/to/hits QUERY=7,20,33 QUERY_TEST_COUNT=5 cargo bench --bench clickbench
//! ```

use std::path::PathBuf;
use std::sync::{Arc, LazyLock};
use std::thread::sleep;
use std::time::{Duration, Instant};

use arrow::compute::concat_batches as arrow_concat_batches;
use arrow::util::display::ArrayFormatter;
use arrow_array::types::Int16Type;
use arrow_array::{BooleanArray, Int16Array, RecordBatch, StringViewArray};
use arrow_buffer::BooleanBuffer;
use tracing_subscriber::{EnvFilter, fmt};

use dispatch::table_input;
use dispatch::{Contains, IntKeyExtractor, OrderBy, ParquetTable, Projection, StringKeyExtractor};

static SOURCE_DIRECTORY: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(std::env::var("SOURCE_DIRECTORY").unwrap()));

fn get_env_var_with_default<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn concat_batches(batches: &[RecordBatch]) -> RecordBatch {
    assert!(!batches.is_empty(), "Cannot concatenate empty batch list");
    let schema = batches[0].schema();
    arrow_concat_batches(&schema, batches).expect("Failed to concatenate batches")
}

fn batch_to_tsv(batch: &RecordBatch) -> String {
    let mut result = String::new();
    let formatters: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| ArrayFormatter::try_new(col.as_ref(), &Default::default()).unwrap())
        .collect();

    for row in 0..batch.num_rows() {
        for (col_idx, formatter) in formatters.iter().enumerate() {
            if col_idx > 0 {
                result.push('\t');
            }
            result.push_str(&formatter.value(row).to_string());
        }
        result.push('\n');
    }
    result
}

fn assert_result(expected: &str, batches: &[RecordBatch]) {
    let batch = concat_batches(batches);
    let actual = batch_to_tsv(&batch);
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Result mismatch.\n\nActual:\n{}\n\nExpected:\n{}",
        actual,
        expected
    );
}

// Query 7: SELECT AdvEngineID, COUNT(*) FROM hits
//   WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC
fn run_query_7(table: &Arc<ParquetTable>) {
    const EXPECTED: &str = r#"2	404602
27	113167
13	45631
45	38960
44	9730
3	6896
62	5266
52	3554
50	938
28	836
53	350
25	343
61	158
21	38
42	20
16	7
7	3
22	1"#;

    let results = table_input(
        table,
        Projection::from_field_names(table.schema(), ["AdvEngineID"]),
        false,
    )
    .project(|| {
        let indices = vec![0];
        move |batch: &RecordBatch| batch.project(&indices).unwrap()
    })
    .filter(|| {
        move |batch: &RecordBatch| {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int16Array>()
                .unwrap();
            let values = col.values();
            BooleanArray::from(BooleanBuffer::collect_bool(values.len(), |i| {
                values[i] != 0
            }))
        }
    })
    .group_by_count::<IntKeyExtractor<Int16Type>>(0)
    .order_by_limit(vec![OrderBy::new(1, true, false)], 10000000)
    .collect();

    assert_result(EXPECTED, &results);
}

// Query 20: SELECT count(*) FROM hits WHERE URL LIKE '%google%'
fn run_query_20(table: &Arc<ParquetTable>) {
    const EXPECTED: &str = "15911\n";

    let results = table_input(
        table,
        Projection::from_field_names(table.schema(), ["URL"]),
        false,
    )
    .filter(|| {
        let mut contains = Contains::new("google");
        move |batch: &RecordBatch| {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            contains.run(col)
        }
    })
    .count()
    .collect();

    assert_result(EXPECTED, &results);
}

// Query 23: SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10
fn run_query_23(table: &Arc<ParquetTable>) {
    const EXPECTED: &str = "7675678523794456216\t1\tGlavnaya gorand. Цветные объявлений районе, вером\t1\t1372708869\t15888\t64469\t1840073959\t2\t3714843517822510735\t0\t44\t5\thttp://e96.ru/search/page.googleTBR%26ad%3D0%26rnd%3D158197%26anbietersburg\thttp://bdsmpeople.ru/obrazom_position/?page\t0\t13593\t158\t13606\t216\t1638\t1658\t22\t15\t7\t700\t0\t0\t22\tnA\t1\t1\t0\t0\t\t\t4005373\t-1\t0\t\t0\t0\t1052\t775\t135\t1372760422\t0\t0\t0\t0\twindows\t1601\t0\t0\t0\t0\t\t213893614\t0\t0\t0\t0\t0\t6\t1372766291\t0\t0\t0\t0\t0\t1412515749\t63522\t-1\t12\tS0\t\u{fffd}\u{c}\t\t\t0\t0\t0\t0\t445\t1234\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t5972490271588207794\t1369713899219085694\t0\n6147260061318473746\t1\tGlavnaya gorand. Цветные объявлений районе, вером\t1\t1372708889\t15888\t64469\t1840073959\t2\t3714843517822510735\t0\t44\t5\thttp://e96.ru/search/page.googleTBR%26ad%3D0%26rnd%3D158197%26anbietersburg\thttp://bdsmpeople.ru/obrazom_position/?page\t0\t13593\t158\t13606\t216\t1638\t1658\t22\t15\t7\t700\t0\t0\t22\tD\u{fffd}\t1\t1\t0\t0\t\t\t4005373\t-1\t0\t\t0\t0\t1052\t775\t135\t1372760442\t0\t0\t0\t0\twindows\t1601\t0\t0\t0\t0\t\t810892658\t0\t0\t0\t0\t0\t6\t1372766305\t0\t0\t0\t0\t0\t1412515749\t63522\t-1\t13\tS0\t\u{fffd}\u{c}\t\t\t0\t0\t0\t0\t0\t16\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t5972490271588207794\t1369713899219085694\t0\n5972689683963797854\t1\tGlavnaya gorand. Цветные объявлений районе, вером\t1\t1372708931\t15888\t64469\t1840073959\t2\t3714843517822510735\t0\t44\t5\thttp://e96.ru/search/page.googleTBR%26ad%3D0%26rnd%3D158197%26anbietersburg\thttp://bdsmpeople.ru/obrazom_position/?page\t0\t13593\t158\t13606\t216\t1638\t1658\t22\t15\t7\t700\t0\t0\t22\tD\u{fffd}\t1\t1\t0\t0\t\t\t4005373\t-1\t0\t\t0\t0\t1052\t775\t135\t1372760476\t0\t0\t0\t0\twindows\t1601\t0\t0\t0\t0\t\t47015096\t0\t0\t0\t0\t0\t6\t1372766330\t0\t0\t0\t0\t0\t1412515749\t63522\t-1\t13\tS0\th1\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t5972490271588207794\t1369713899219085694\t0\n8008688361303225116\t1\tСкачать онлайн играй! - Туризма - Крымский тренчкоты в интернет магазин Wildberries.ru (Работа - IRR.ru - модных словариумных\t1\t1372709521\t15888\t63217\t1975817788\t229\t804133623150786791\t1\t44\t7\thttp://bjdswaps.google-photo\thttp://loveche.html?ctid\t0\t12409\t20\t10093\t22\t1749\t867\t23\t15\t3\t700.224\t0\t0\t15\tD\u{fffd}\t1\t1\t0\t0\t\t\t3056753\t-1\t0\t\t0\t0\t1608\t662\t135\t1372745995\t4\t1\t16561\t0\twindows\t1\t0\t0\t0\t5347008031302181363\t\t766531830\t0\t0\t0\t0\t0\t5\t1372766425\t31\t1\t3\t11237\t31\t1870660671\t-1\t-1\t-1\tE3\t_i\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t-2736470446903004689\t7116991598850737408\t0\n7436461208655480623\t1\tКомпании Вино в хорошие\t1\t1372710744\t15888\t35534\t1741555497\t39\t7082047337377160280\t0\t44\t5\thttp://rsdn.ru/catalog/cifrovye-advertisement=little&category=22&input_bdsmpeople.ru/index,google.ru/news/39826\thttp://kalina?block/?inst_the_book.php?cPath=40_57470493958402/\t0\t10634\t20\t0\t0\t1996\t1781\t37\t15\t7\t700\t0\t0\t22\tD\u{fffd}\t1\t1\t0\t0\t\t\t808950\t5\t0\t\t0\t0\t1261\t1017\t433\t1372791792\t4\t1\t16561\t0\twindows-1251;charset\t1601\t1\t0\t0\t6804199628189316872\t\t626511463\t0\t0\t0\t1\t0\t5\t1372732542\t31\t2\t3\t694\t57\t1448806868\t-1\t-1\t-1\tE3\t_i\t\t\t0\t0\t0\t3\t345\t147\t239\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t8931346360692564721\t1971436513446935197\t0\n5564518777317455184\t0\t\u{ab}set\u{bb} в пробег аппах и обслуживатизиров\t1\t1372711469\t15888\t5822\t1920787234\t32\t3712346975274085073\t1\t2\t3\thttp://auto_gruppy/christikha/hotel=-1&trafkey=605&from=&power_name=Платья&produkty%2Furl.google.ru/index\thttp://rmnt.ru/cars/passenger/hellardous/42/~37/?suggest&id=3869551753&custom=0&undefined/undefined/under=28036,5;362;108;16762643539\t0\t8563\t21482\t9822\t18528\t1638\t1658\t23\t15\t7\t700\t0\t0\t16\tD\u{fffd}\t1\t1\t0\t0\t\t\t207348\t-1\t0\t\t0\t0\t1509\t733\t135\t1372750392\t4\t1\t15738\t0\twindows-1251;charset\t1\t0\t0\t0\t8007561756096276896\t\t1034507462\t0\t0\t0\t0\t0\t5\t1372759436\t0\t0\t0\t0\t0\t2016848722\t-1\t-1\t-1\tS0\th1\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t7820066807630413322\t-3258566084785303139\t0\n7381524648140977766\t1\tPloshchad' stolitsi zwy 110911923, Г официальная Прессы и Огонек\t1\t1372712506\t15888\t63217\t1638850281\t59\t1564939829982760596\t1\t44\t5\thttp://bjdleaksbrand=bpc bonprix%2F12.02&he=1024&location=pm;f=inbox;pmsg_1733/page.google\thttp://loveplanet.ru/url?sa=t&rct\t0\t12409\t20\t10093\t22\t1996\t1666\t37\t15\t7\t700\t0\t0\t22\tD\u{fffd}\t1\t1\t0\t0\t\t\t2059788\t-1\t0\t\t0\t0\t1261\t1206\t433\t1372759478\t0\t0\t0\t0\twindows\t1\t0\t0\t0\t0\t\t827020970\t0\t0\t0\t0\t0\t5\t1372791236\t0\t0\t0\t0\t0\t2001459352\t-1\t-1\t-1\tS0\t\u{fffd}\u{c}\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t1384301141639030267\t9047048983006699504\t0\n8614832219462424183\t1\tМодель для сумки - регеш (Россия)\t1\t1372713563\t15888\t3035\t2022088895\t38\t2590751384199385434\t1\t2\t88\thttp://smeshariki.ru/googleTBR%26ar_ntype=citykurortmag\thttp://holodilnik.ru/GameMain.aspx?color=0&choos&source=web&cd\t0\t10271\t158\t13384\t216\t1917\t879\t37\t15\t7\t700\t0\t0\t1\tD\u{fffd}\t1\t1\t0\t0\t\t\t3991944\t-1\t0\t\t0\t0\t746\t464\t322\t1372754670\t0\t0\t0\t0\twindows-1251;charset\t1\t0\t0\t0\t7607749204513951316\t\t427646581\t0\t0\t0\t0\t0\t5\t1372720757\t50\t2\t3\t0\t30\t1146019198\t-1\t-1\t-1\tS0\t\u{fffd}\u{c}\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t1157471009075867478\t1000799766482180932\t0\n7168314068394418899\t1\tпо полиция +опытовой рецензии,\t1\t1372713760\t15888\t35534\t-1420082055\t11579\t4822773326251181180\t0\t2\t7\thttp:%2F%2Fsapozhki-advertime-2/#page.google/dodge\thttp://saint-peters-total=меньше 100007&text=b.akhua_deckaya-look/time-2/#page=3&oprnd=6817922197946&ei=JtHTUYWRCqXA&bvm=bv.49784469,d.ZWU&cad=rjt&fu=0&type_id=172&msid=1&marka=88&text=krasnaia-moda\t0\t14550\t952\t8565\t375\t1304\t978\t37\t15\t4\t700.224\t2\t7\t13\tD\u{fffd}\t1\t1\t0\t0\t\t\t2675432\t3\t3\tdave kino 2013 года в ростопримеча\t0\t0\t1972\t778\t135\t1372712707\t4\t1\t16561\t0\twindows\t1601\t0\t0\t0\t6494516778257365839\t\t393719418\t0\t0\t0\t0\t0\t5\t1372782490\t31\t2\t2\t14851\t1\t-1016483843\t61823\t-1\t1\tS0\t\u{fffd}\u{c}\t\t\t0\t0\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t-4470345086215748575\t-5322637665780806659\t0\n8235569889353442646\t1\t\t1\t1372713773\t15888\t35534\t-1420082055\t11579\t4822773326251181180\t0\t2\t7\thttp:%2F%2Fsapozhki-advertime-2/#page.google/dodge\t\t0\t0\t0\t8565\t375\t1304\t978\t37\t15\t4\t700.224\t2\t7\t13\tD\u{fffd}\t1\t1\t0\t0\t\t\t2675432\t0\t0\t\t0\t1\t1972\t778\t135\t1372712722\t4\t1\t16561\t0\twindows\t1601\t0\t0\t1\t6494516778257365839\t\t393719418\t0\t0\t0\t1\t0\t5\t1372782504\t31\t2\t2\t14851\t1\t-1016483843\t61823\t-1\t2\tS0\t\u{fffd}\u{c}\t\t\t0\t318\t0\t0\t0\t0\t0\t0\t\t0\t\tNH\u{1c}\t0\t\t\t\t\t\t\t\t\t\t\t0\t-296158784638538920\t-5322637665780806659\t0";

    let results = table_input(
        table,
        Projection::from_field_names(table.schema(), ["EventTime", "URL"]),
        true,
    )
    .filter(|| {
        let mut contains = Contains::new("google");
        move |batch: &RecordBatch| {
            let col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            contains.run(col)
        }
    })
    .order_by_limit(vec![OrderBy::new(0, false, false)], 10)
    .materialize(table.clone(), Projection::all(105))
    .order_by_limit(vec![OrderBy::new(4, false, false)], 10)
    .project(|| {
        let indices = (0..105).collect::<Vec<_>>();
        move |batch: &RecordBatch| batch.project(&indices).unwrap()
    })
    .collect();

    assert_result(EXPECTED, &results);
}

// Query 33: SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10
fn run_query_33(table: &Arc<ParquetTable>) {
    const EXPECTED: &str = r#"http://liver.ru/belgorod/page/1006.jки/доп_приборы	3288173
http://kinopoisk.ru	1625250
http://bdsm_po_yers=0&with_video	791465
http://video.yandex	582400
http://smeshariki.ru/region	514984
http://auto_fiat_dlya-bluzki%2F8536.30.18&he=900&with	507995
http://liver.ru/place_rukodel=365115eb7bbb90	359893
http://kinopoisk.ru/vladimir.irr.ru	354690
http://video.yandex.ru/search/?jenre=50&s_yers	318979
http://tienskaia-moda	289355
"#;

    let results = table_input(table, Projection::columns([13]), false)
        .group_by_count::<StringKeyExtractor>(0)
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect();

    assert_result(EXPECTED, &results);
}

// ── Query registry ────────────────────────────────────────────────────────

const QUERIES: &[(u32, fn(&Arc<ParquetTable>))] = &[
    (7, run_query_7),
    (20, run_query_20),
    (23, run_query_23),
    (33, run_query_33),
];

fn get_query_fn(id: u32) -> fn(&Arc<ParquetTable>) {
    QUERIES
        .iter()
        .find(|(qid, _)| *qid == id)
        .unwrap_or_else(|| panic!("Unknown query: {id}"))
        .1
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stdout)
        .init();

    let num_workers: usize = std::env::var("WORKER_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| core_affinity::get_core_ids().unwrap().len());
    dispatch::init(num_workers);

    let table =
        Arc::new(ParquetTable::from_directory(&SOURCE_DIRECTORY).expect("Could not create source"));

    let queries: Vec<(u32, fn(&Arc<ParquetTable>))> = match std::env::var("QUERY") {
        Ok(val) => val
            .split(',')
            .map(|s| {
                let id: u32 = s
                    .trim()
                    .parse()
                    .expect("QUERY must be comma-separated numbers");
                (id, get_query_fn(id))
            })
            .collect(),
        Err(_) => QUERIES.to_vec(),
    };
    let iterations = get_env_var_with_default("QUERY_TEST_COUNT", 1);

    for (id, run) in &queries {
        println!("=== Query {} ===", id);
        for i in 0..iterations {
            let start = Instant::now();
            run(&table);
            println!(
                "[{}/{}] Query {} — {}ms",
                i + 1,
                iterations,
                id,
                start.elapsed().as_millis()
            );
            if let Ok(a) = std::env::var("SLEEP") {
                sleep(Duration::from_secs(a.parse().unwrap()));
            }
        }
    }
}
