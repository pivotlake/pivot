"""Machine-only probe: how long a fetched row group waits in the fetcher before it is sent on,
and how long a request waits before its read is asked for."""
import sys

p = sys.argv[1] + '/parquet-engine/src/reading/fetching/fetcher.rs'
s = open(p).read()
for a, b in [
    ('''        self.fetched_row_groups
            .push_back(request.into_row_group_buffer(Some(response), Some(read_ahead_charge)));
        Ok(())''', '''        self.fetched_row_groups
            .push_back(request.into_row_group_buffer(Some(response), Some(read_ahead_charge)));
        self.fetched_at.push_back(std::time::Instant::now());
        Ok(())'''),
    ('''        match self.fetched_row_groups.pop_front() {
            Some(row_group_buffer) => {''', '''        match self.fetched_row_groups.pop_front() {
            Some(row_group_buffer) => {
                if let Some(fetched_at) = self.fetched_at.pop_front() {
                    eprintln!(
                        "FETCHWAIT waited_ms={} still_waiting={}",
                        fetched_at.elapsed().as_millis(),
                        self.fetched_row_groups.len()
                    );
                }'''),
    ('''    fetched_row_groups: VecDeque<RowGroupBuffer<M>>,
}''', '''    fetched_row_groups: VecDeque<RowGroupBuffer<M>>,
    fetched_at: VecDeque<std::time::Instant>,
}'''),
    ('''            fetched_row_groups: VecDeque::new(),
        }''', '''            fetched_row_groups: VecDeque::new(),
            fetched_at: VecDeque::new(),
        }'''),
    ('''        self.uncached_row_groups_sender
            .send(request.into_metadata())?;
        Ok(())''', '''        self.uncached_row_groups_sender
            .send(request.into_metadata())?;
        eprintln!("FETCHREQ t_ms={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
        Ok(())'''),
    ('''            self.in_flight.insert(id, (request, read_ahead_charge));
        }
        Ok(WorkStatus::Ran)''', '''            self.in_flight.insert(id, (request, read_ahead_charge));
            eprintln!("FETCHISSUE t_ms={}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
        }
        Ok(WorkStatus::Ran)'''),
]:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('fetch wait probe applied')
