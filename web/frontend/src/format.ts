// Small formatting helpers shared across the dashboard.

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB", "PiB"];
  let value = n / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[i]}`;
}

export function count(n: number | null): string {
  if (n === null) return "-";
  return n.toLocaleString("en-US");
}

export function rate(n: number | null): string {
  if (n === null || n <= 0) return "idle";
  if (n < 1) return `${n.toFixed(2)} rows/s`;
  if (n < 1000) return `${n.toFixed(0)} rows/s`;
  return `${(n / 1000).toFixed(1)}k rows/s`;
}

export function percent(n: number): string {
  return `${n.toFixed(n >= 100 ? 0 : 1)}%`;
}

export function ago(unixMs: number): string {
  if (!unixMs) return "never";
  const secs = Math.max(0, (Date.now() - unixMs) / 1000);
  if (secs < 60) return `${Math.floor(secs)}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  return `${Math.floor(secs / 3600)}h ago`;
}
