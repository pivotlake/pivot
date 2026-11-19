"""Machine-only probe: every dataflow step of 20 ms or more, and every operator call of
20 ms or more with the operator's name.
  STEP ms=<duration> kind=<closure type>
  SLOWOP <consume|run> <operator type> <ms>"""
import sys

root = sys.argv[1] + '/dispatch/src/'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


edit('data_flow.rs', [
    ('''    fn try_run_or<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {''',
     '''    fn try_run_or<R, F: FnOnce(&mut Self) -> Result<R>>(&mut self, default: R, f: F) -> R {
        let probe_started = Instant::now();
        let probe_result = self.try_run_or_unprobed(default, f);
        let probe_elapsed = probe_started.elapsed();
        if probe_elapsed.as_millis() >= 20 {
            eprintln!(
                "STEP ms={:.1} kind={}",
                probe_elapsed.as_secs_f64() * 1000.0,
                std::any::type_name::<F>()
            );
        }
        probe_result
    }

    fn try_run_or_unprobed<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {'''),
])
print('long step probe applied')
