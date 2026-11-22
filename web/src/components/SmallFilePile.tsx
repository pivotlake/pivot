import { chooseBlockLabel, readFileKeys } from "../fileKeys";
import { formatMemory } from "../format";
import type { CompactionSettings, TableFile } from "../tables";
import { placeInView, useZoom, WIDTH_ONLY, ZoomControls, ZoomSurface, type Zoom } from "../zoom";

// A track is one row of blocks, so zooming to a file leaves its height as
// it is.
const CENTRE_DOWN = 0.5;

// One group of small files as blocks along a track that is full at the merge
// target: each block a file, as wide as its share of the target. A group
// longer than the target fills the track. A block wide enough for it has the
// file's keys written on it, and a double click on a block zooms to its
// file.
function PileRow({
  label,
  files,
  settings,
  sortBy,
  zoom,
}: {
  label: string;
  files: TableFile[];
  settings: CompactionSettings;
  sortBy: string[];
  zoom: Zoom;
}) {
  const bytes = files.reduce((total, file) => total + file.bytes, 0);
  const trackBytes = Math.max(bytes, settings.mergeTargetBytes);
  let bytesBefore = 0;
  const blocks = files.map((file, index) => {
    const start = bytesBefore / trackBytes;
    bytesBefore += file.bytes;
    return { file, start, end: bytesBefore / trackBytes, isAlternate: index % 2 === 1 };
  });
  return (
    <div className="pile-row">
      <div className="pile-row-heading">
        <span>{label}</span>
        <span>
          {files.length} files · {formatMemory(bytes)}
        </span>
      </div>
      <ZoomSurface
        zoom={zoom}
        className="pile-track"
        label={`${label}: ${files.length} files, ${formatMemory(bytes)} of the ${formatMemory(settings.mergeTargetBytes)} a merge starts at`}
      >
        {(widthPx) =>
          blocks.map((block) => {
            const place = placeInView(zoom.shown, block);
            if (place.width <= 0) return null;
            const keys = chooseBlockLabel(readFileKeys(block.file, sortBy), place.width * widthPx);
            return (
              <div
                key={block.file.path}
                className={block.isAlternate ? "pile-file is-alternate" : "pile-file"}
                style={{ left: `${place.left * 100}%`, width: `${place.width * 100}%` }}
                title={`${block.file.path.slice(block.file.path.lastIndexOf("/") + 1)}: ${formatMemory(block.file.bytes)}`}
                onDoubleClick={(event) => {
                  event.stopPropagation();
                  zoom.showStretch(block, CENTRE_DOWN);
                }}
              >
                {keys && <span className="file-label">{keys}</span>}
              </div>
            );
          })
        }
      </ZoomSurface>
    </div>
  );
}

// A partition's small files as the server's compacter sees them: a pile it
// merges by size alone, whatever keys the files hold. Files are taken in
// order until they total the merge target, and that group is a merge; what
// is left waits for more, or for the pile to reach the file count at which
// it may merge short of the target.
//
// The pile's tracks zoom together, into the same part of each.
export function SmallFilePile({
  files,
  merges,
  settings,
  sortBy,
}: {
  files: TableFile[];
  // The groups of `files` a routine round would merge now.
  merges: TableFile[][];
  settings: CompactionSettings;
  // The table's sort columns, in order.
  sortBy: string[];
}) {
  const zoom = useZoom(WIDTH_ONLY);
  const merging = new Set(merges.flat());
  const waiting = files.filter((file) => !merging.has(file));
  const waitingBytes = waiting.reduce((total, file) => total + file.bytes, 0);

  return (
    <div className="pile">
      <ZoomControls zoom={zoom} />
      {merges.map((merge, mergeIndex) => (
        <div key={merge[0].path} className="pile-group is-ready">
          <PileRow
            label={`Merge ${mergeIndex + 1}, ready`}
            files={merge}
            settings={settings}
            sortBy={sortBy}
            zoom={zoom}
          />
        </div>
      ))}
      {waiting.length > 0 && (
        <div className="pile-group">
          <PileRow label="Waiting" files={waiting} settings={settings} sortBy={sortBy} zoom={zoom} />
          <p className="pile-note">
            Merges at {formatMemory(settings.mergeTargetBytes)} ({formatMemory(waitingBytes)} so far) or at{" "}
            {settings.minFiles} files ({waiting.length} so far).
          </p>
        </div>
      )}
    </div>
  );
}
