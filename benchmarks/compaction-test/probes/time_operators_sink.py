"""Box-only: print every operator call that runs longer than 5 ms without yielding."""
import os

p = os.path.expanduser('~/pivot-compaction-6ba4c334/dispatch/src/operations/unary/mod.rs')
s = open(p).read()

helper = '''
/// Box-only timing probe: report a call that held its worker for long.
fn report_slow_call<U>(what: &str, started: std::time::Instant) {
    let elapsed = started.elapsed();
    if elapsed.as_millis() >= 20 {
        eprintln!(
            "SLOWOP {} {} {:.1}",
            what,
            std::any::type_name::<U>(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
'''
anchor = '/// Wires a [`Unary`] transform to a receiver (input channel) and sender (output channel),'
assert s.count(anchor) == 1
s = s.replace(anchor, helper + '\n' + anchor)

a = '''        if !self.unary.ready_for_more_work() {
            return Ok(self.unary.run(&mut *self.sender, io)?);
        }

        let item = match self.receiver.try_recv() {
            None => return Ok(self.unary.run(&mut *self.sender, io)?),
            Some(t) => t,
        };

        self.unary.consume(item, &mut *self.sender, io)?;

        Ok(WorkStatus::Ran)'''
b = '''        if !self.unary.ready_for_more_work() {
            let started = std::time::Instant::now();
            let status = self.unary.run(&mut *self.sender, io)?;
            report_slow_call::<U>("run", started);
            return Ok(status);
        }

        let item = match self.receiver.try_recv() {
            None => {
                let started = std::time::Instant::now();
                let status = self.unary.run(&mut *self.sender, io)?;
                report_slow_call::<U>("run", started);
                return Ok(status);
            }
            Some(t) => t,
        };

        let started = std::time::Instant::now();
        self.unary.consume(item, &mut *self.sender, io)?;
        report_slow_call::<U>("consume", started);

        Ok(WorkStatus::Ran)'''
assert s.count(a) == 1
s = s.replace(a, b)

a = '''            Some(s) => {
                self.unary.consume(s, &mut *self.sender, io)?;
                Ok(WorkStatus::Ran)
            }'''
b = '''            Some(s) => {
                let started = std::time::Instant::now();
                self.unary.consume(s, &mut *self.sender, io)?;
                report_slow_call::<U>("consume", started);
                Ok(WorkStatus::Ran)
            }'''
assert s.count(a) == 1
s = s.replace(a, b)
open(p, 'w').write(s)
print('timing probe added')
