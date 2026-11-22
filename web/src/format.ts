const DECIMAL_BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"];
const BINARY_BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB"];

// Scales a byte count to the largest unit that keeps it at or above one, and
// writes it with three significant digits.
function formatScaledBytes(bytes: number, unitSize: number, units: string[]): string {
  let scaled = bytes;
  let unitIndex = 0;
  while (scaled >= unitSize && unitIndex < units.length - 1) {
    scaled /= unitSize;
    unitIndex += 1;
  }
  const digits = scaled >= 100 || unitIndex === 0 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits)} ${units[unitIndex]}`;
}

// Data on the move, in the decimal units throughput is quoted in.
export function formatBytes(bytes: number): string {
  return formatScaledBytes(bytes, 1000, DECIMAL_BYTE_UNITS);
}

// Memory, in the binary units it is allocated in.
export function formatMemory(bytes: number): string {
  return formatScaledBytes(bytes, 1024, BINARY_BYTE_UNITS);
}

export function formatDuration(milliseconds: number): string {
  const seconds = milliseconds / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.floor(seconds % 60)}s`;
}
