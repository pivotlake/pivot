import { formatMemory } from "../format";
import type { CompactionSettings, TableFile } from "../tables";

// One group of small files as blocks along a track that is full at the merge
// target: each block a file, as wide as its share of the target. A group
// longer than the target fills the track.
function PileRow({
  label,
  files,
  settings,
}: {
  label: string;
  files: TableFile[];
  settings: CompactionSettings;
}) {
  const bytes = files.reduce((total, file) => total + file.bytes, 0);
  const trackBytes = Math.max(bytes, settings.mergeTargetBytes);
  return (
    <div className="pile-row">
      <div className="pile-row-heading">
        <span>{label}</span>
        <span>
          {files.length} files · {formatMemory(bytes)}
        </span>
      </div>
      <div
        className="pile-track"
        role="img"
        aria-label={`${label}: ${files.length} files, ${formatMemory(bytes)} of the ${formatMemory(settings.mergeTargetBytes)} a merge starts at`}
      >
        {files.map((file) => (
          <div
            key={file.path}
            className="pile-file"
            style={{ width: `${(file.bytes / trackBytes) * 100}%` }}
            title={`${file.path.slice(file.path.lastIndexOf("/") + 1)}: ${formatMemory(file.bytes)}`}
          />
        ))}
      </div>
    </div>
  );
}

// A partition's small files as the server's compacter sees them: a pile it
// merges by size alone, whatever keys the files hold. Files are taken in
// order until they total the merge target, and that group is a merge; what
// is left waits for more, or for the pile to reach the file count at which
// it may merge short of the target.
export function SmallFilePile({
  files,
  merges,
  settings,
}: {
  files: TableFile[];
  // The groups of `files` a routine round would merge now.
  merges: TableFile[][];
  settings: CompactionSettings;
}) {
  const merging = new Set(merges.flat());
  const waiting = files.filter((file) => !merging.has(file));
  const waitingBytes = waiting.reduce((total, file) => total + file.bytes, 0);

  return (
    <div className="pile">
      {merges.map((merge, mergeIndex) => (
        <div key={merge[0].path} className="pile-group is-ready">
          <PileRow label={`Merge ${mergeIndex + 1}, ready`} files={merge} settings={settings} />
        </div>
      ))}
      {waiting.length > 0 && (
        <div className="pile-group">
          <PileRow label="Waiting" files={waiting} settings={settings} />
          <p className="pile-note">
            Merges at {formatMemory(settings.mergeTargetBytes)} ({formatMemory(waitingBytes)} so far) or at{" "}
            {settings.minFiles} files ({waiting.length} so far).
          </p>
        </div>
      )}
    </div>
  );
}
