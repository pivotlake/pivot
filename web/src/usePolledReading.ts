import { useEffect, useState } from "react";

const POLL_MS = 5000;

// What the page knows of something it reads from the server: nothing yet,
// the latest reading, or why the latest reading failed.
export type Loaded<Reading> =
  | { status: "loading" }
  | { status: "live"; reading: Reading }
  | { status: "failed"; reason: string };

// Reads with `read` now and again every few seconds, starting over whenever
// `key` changes, and a moment after each reading comes back, so a slow read
// never has another queue up behind it.
export function usePolledReading<Reading>(key: string, read: () => Promise<Reading>): Loaded<Reading> {
  const [loaded, setLoaded] = useState<Loaded<Reading>>({ status: "loading" });
  useEffect(() => {
    let isStopped = false;
    let nextPoll: number | undefined;
    setLoaded({ status: "loading" });
    const poll = async () => {
      try {
        const reading = await read();
        if (isStopped) return;
        setLoaded({ status: "live", reading });
      } catch (error) {
        if (isStopped) return;
        setLoaded({ status: "failed", reason: error instanceof Error ? error.message : String(error) });
      }
      nextPoll = window.setTimeout(poll, POLL_MS);
    };
    poll();
    return () => {
      isStopped = true;
      window.clearTimeout(nextPoll);
    };
    // `read` is a new function on every render; `key` says when what it reads changes.
  }, [key]);
  return loaded;
}
