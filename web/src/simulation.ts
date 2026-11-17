// Made-up compaction progress for the page to draw until the server reports
// its own. The exported type is the shape a server would report; everything
// that decides what happens next stays inside `CompactionSimulation`.

export type Compaction =
  | { kind: "running"; table: string; filesToMerge: number; filesMerged: number }
  | { kind: "idle"; finishedTable: string; finishedFiles: number };

const TABLES = ["events", "page_views", "orders", "sessions", "line_items", "trips"];

// Average rates per second, so the pace does not depend on how often the
// simulation is advanced.
const COMPACTION_STARTS_PER_SECOND = 0.12;
const COMPACTION_FILES_PER_SECOND = 5;

export class CompactionSimulation {
  private compaction: Compaction;
  private advancedAtMs: number;

  // Opens part way through a round, so the first frame already shows progress.
  constructor(nowMs: number) {
    this.advancedAtMs = nowMs;
    this.compaction = { ...this.startCompaction(), filesMerged: 37 };
  }

  advance(nowMs: number): Compaction {
    const elapsedSeconds = (nowMs - this.advancedAtMs) / 1000;
    this.advancedAtMs = nowMs;

    const compaction = this.compaction;
    if (compaction.kind === "idle") {
      const starts = Math.random() < 1 - Math.exp(-COMPACTION_STARTS_PER_SECOND * elapsedSeconds);
      this.compaction = starts ? this.startCompaction() : compaction;
      return this.compaction;
    }

    const filesThisStep = COMPACTION_FILES_PER_SECOND * elapsedSeconds * (0.4 + 1.2 * Math.random());
    // The fraction of a file left over decides whether one more file lands in
    // this step, so short steps still add up to the average rate.
    const wholeFilesThisStep = Math.floor(filesThisStep) + (Math.random() < filesThisStep % 1 ? 1 : 0);
    const filesMerged = compaction.filesMerged + wholeFilesThisStep;
    this.compaction =
      filesMerged >= compaction.filesToMerge
        ? { kind: "idle", finishedTable: compaction.table, finishedFiles: compaction.filesToMerge }
        : { ...compaction, filesMerged };
    return this.compaction;
  }

  private startCompaction(): Extract<Compaction, { kind: "running" }> {
    return {
      kind: "running",
      table: TABLES[Math.floor(Math.random() * TABLES.length)],
      filesToMerge: 60 + Math.floor(Math.random() * 360),
      filesMerged: 0,
    };
  }
}
