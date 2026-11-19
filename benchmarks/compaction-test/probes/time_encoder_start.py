"""Box-only: time the parts of the encoder's first step for a leaf."""
import os

p = os.path.expanduser('~/pivot-merge/parquet-engine/src/writing/encoder/mod.rs')
s = open(p).read()


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)


once('''        let values = concat_chunks(allocator, &job.value_chunks).map_err(WriteError::from)?;
        let leaf = Leaf {''',
     '''        let concat_started = std::time::Instant::now();
        let values = concat_chunks(allocator, &job.value_chunks).map_err(WriteError::from)?;
        let concat_ms = concat_started.elapsed().as_secs_f64() * 1000.0;
        if concat_ms >= 5.0 {
            eprintln!(
                "PHASE concat {:.1} {} {}",
                concat_ms,
                job.path.join("."),
                values.get_array_memory_size() >> 20
            );
        }
        let leaf = Leaf {''')

once('''        let statistics = leaf_statistics(&leaf);
        let (dictionary_page, data_page_encoding, data_pages, remaining_pages) =
            match dictionary::try_encode(&leaf, codec, allocator)? {''',
     '''        let stats_started = std::time::Instant::now();
        let statistics = leaf_statistics(&leaf);
        let stats_ms = stats_started.elapsed().as_secs_f64() * 1000.0;
        if stats_ms >= 5.0 {
            eprintln!("PHASE stats {:.1} {} 0", stats_ms, leaf.path.join("."));
        }
        let dictionary_started = std::time::Instant::now();
        let dictionary_attempt = dictionary::try_encode(&leaf, codec, allocator)?;
        let dictionary_ms = dictionary_started.elapsed().as_secs_f64() * 1000.0;
        if dictionary_ms >= 5.0 {
            eprintln!(
                "PHASE dictionary_{} {:.1} {} 0",
                if dictionary_attempt.is_some() { "used" } else { "rejected" },
                dictionary_ms,
                leaf.path.join(".")
            );
        }
        let (dictionary_page, data_page_encoding, data_pages, remaining_pages) =
            match dictionary_attempt {''')
open(p, 'w').write(s)
print('encoder phases timed')
