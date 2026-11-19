"""Machine-only probe: what the compactor's selection sees in each partition a round looks at.
Lines: LAYOUTPROBE partition=<bytes as text> small_batch=<n or none> candidates=<n> measured=<n>
       files_with_partners=<n> pairs_clearing_bar=<n> group=<n or none>
and    ROUNDPROBE pass=<..> lists_every_partition=<bool> partitions=<n>"""
import sys

root = sys.argv[1] + '/datastore-pivotlake/src/compact/'


def edit(name, pairs):
    s = open(root + name).read()
    for a, b in pairs:
        assert s.count(a) == 1, (name, s.count(a), a[:80])
        s = s.replace(a, b)
    open(root + name, 'w').write(s)


edit('mod.rs', [
    ('''        let cache = overlaps.entry(partition.to_vec()).or_default();
        cache.observe(&candidates, table.sort_by(), pass)?;
        let Some(group) = cache.select_group(reserved, pass, self.layout_clique_size) else {
            return Ok(None);
        };''', '''        let cache = overlaps.entry(partition.to_vec()).or_default();
        cache.observe(&candidates, table.sort_by(), pass)?;
        let selected = cache.select_group(reserved, pass, self.layout_clique_size);
        {
            use overlap::LayoutSource as _;
            let mut candidate_count = 0usize;
            candidates.walk_candidates(&mut |_, _| candidate_count += 1)?;
            let (measured, files_with_partners, pairs_clearing_bar) = cache.probe_counts();
            eprintln!(
                "LAYOUTPROBE partition={} candidates={} measured={} files_with_partners={} pairs_clearing_bar={} group={}",
                String::from_utf8_lossy(partition),
                candidate_count,
                measured,
                files_with_partners,
                pairs_clearing_bar,
                selected.as_ref().map_or("none".to_string(), |group| group.len().to_string()),
            );
        }
        let Some(group) = selected else {
            return Ok(None);
        };'''),
    ('''        for partitions in unfinished.chunks(PARTITIONS_PER_INDEXING) {
            table.index_partitions(partitions)?;''', '''        eprintln!(
            "ROUNDPROBE unfinished_partitions={} listed={}",
            unfinished.len(),
            progress.partitions.len()
        );
        for partitions in unfinished.chunks(PARTITIONS_PER_INDEXING) {
            table.index_partitions(partitions)?;'''),
])
edit('overlap.rs', [
    ('''    /// Let go of the pairs below the contested bar, which only the final''', '''    /// Probe: files measured, files that have a partner, pairs that clear the bar.
    pub(super) fn probe_counts(&self) -> (usize, usize, usize) {
        let pairs: usize = self
            .partners
            .values()
            .map(|partners| partners.values().filter(|overlap| overlap.clears_bar).count())
            .sum();
        (self.measured.len(), self.partners.len(), pairs / 2)
    }

    /// Let go of the pairs below the contested bar, which only the final'''),
])
print('layout probe applied')
