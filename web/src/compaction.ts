// Which of a table's small files a routine compaction round would merge,
// worked out on the page from the sizes `system.table_files` gives.
//
// The rule is the server's (`datastore-pivotlake/src/compact`), which treats
// a partition's files in two ways by their size:
//
// - Small files, those below half the target size, are merged for their size
//   alone: once their total reaches the merge target, or once there are
//   enough of them and the pile is balanced. Sizes are all this needs, so
//   what the page works out is what the server does.
// - Files at least half the target size are merged for their layout: a group
//   of files of a sorted table whose keys all interleave is rewritten in key
//   order. The server decides that from the bounds of each file's row
//   groups, read from the files' footers, which `system.table_files` does
//   not carry, so the page does not predict these merges. A file with one
//   row group interleaves with nothing unless its whole range lies inside
//   another file's, however much the two files' ranges overlap.

import type { CompactionSettings, TableFile } from "./tables";

// Each file weighs this much more than its size when a pile of small files
// is tested for balance, so a pile of many tiny files qualifies.
const FIXED_FILE_COST_BYTES = 5 * 1024 * 1024;
// A balanced pile totals at least this many times its largest file.
const MIN_BALANCE_RATIO = 5;

export function isSmallFile(file: TableFile, settings: CompactionSettings): boolean {
  return file.bytes < Math.ceil(settings.targetBytes / 2);
}

// The next group of `smallFiles` a routine round would merge: the first of
// them that together reach the merge target, or, for a pile of at least the
// minimum count that never reaches it, what is left of the pile after its
// largest files are dropped until the rest is balanced.
function takeSmallFileBatch(
  smallFiles: TableFile[],
  settings: CompactionSettings,
): TableFile[] | undefined {
  let totalBytes = 0;
  const batch: TableFile[] = [];
  for (const file of smallFiles) {
    totalBytes += file.bytes;
    batch.push(file);
    if (batch.length >= 2 && totalBytes >= settings.mergeTargetBytes) {
      return batch;
    }
  }
  if (batch.length < Math.max(settings.minFiles, 2)) {
    return undefined;
  }
  while (batch.length >= 2) {
    const largest = batch.reduce((largest, file) => (file.bytes > largest.bytes ? file : largest));
    const ratio =
      (totalBytes + FIXED_FILE_COST_BYTES * batch.length) / (largest.bytes + FIXED_FILE_COST_BYTES);
    if (ratio >= MIN_BALANCE_RATIO) {
      return batch;
    }
    totalBytes -= largest.bytes;
    batch.splice(batch.indexOf(largest), 1);
  }
  return undefined;
}

// The merges of small files a routine round would start in one partition, as
// it finds them: each group of small files that qualifies, taken in turn
// from the ones no earlier group took.
export function planSmallFileMerges(files: TableFile[], settings: CompactionSettings): TableFile[][] {
  const merges: TableFile[][] = [];
  const merging = new Set<TableFile>();
  for (;;) {
    const smallFiles = files.filter((file) => isSmallFile(file, settings) && !merging.has(file));
    const batch = takeSmallFileBatch(smallFiles, settings);
    if (!batch) {
      return merges;
    }
    for (const file of batch) merging.add(file);
    merges.push(batch);
  }
}
