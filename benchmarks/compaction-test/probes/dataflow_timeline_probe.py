"""Machine-only probe: when each worker receives and finishes each dataflow, and every
dataflow step of 3 ms or more. Lines (t in microseconds since the epoch):
  DFRECV w=<worker> df=<id> t=<t>
  DFDONE w=<worker> df=<id> t=<t>
  STEP df=<id> ms=<duration> t=<end> kind=<closure type>"""
import sys

root = sys.argv[1] + '/dispatch/src/'
NOW = 'std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros()'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


edit('worker.rs', [
    ('''                Ok(data_flow) => {
                    LIVE_DATAFLOWS.fetch_add(1, Ordering::Relaxed);''', '''                Ok(data_flow) => {
                    eprintln!("DFRECV w={} df={} t={}", self.id, data_flow.id(), %s);
                    LIVE_DATAFLOWS.fetch_add(1, Ordering::Relaxed);''' % NOW),
    ('''            debug!("Finished data flow {:?}", id);''', '''            debug!("Finished data flow {:?}", id);
            eprintln!("DFDONE w={} df={} t={}", self.id, id, %s);''' % NOW),
])
edit('data_flow.rs', [
    ('''    fn try_run_or<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {''',
     '''    fn try_run_or<R, F: FnOnce(&mut Self) -> Result<R>>(&mut self, default: R, f: F) -> R {
        let probe_started = Instant::now();
        let probe_id = self.id;
        let probe_result = self.try_run_or_unprobed(default, f);
        let probe_elapsed = probe_started.elapsed();
        if probe_elapsed.as_micros() >= 3000 {
            eprintln!(
                "STEP df={} ms={:.1} t={} kind={}",
                probe_id,
                probe_elapsed.as_secs_f64() * 1000.0,
                %s,
                std::any::type_name::<F>()
            );
        }
        probe_result
    }

    fn try_run_or_unprobed<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {''' % NOW),
])
print('dataflow timeline probe applied')
