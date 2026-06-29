import { useEffect, useState } from "react";
import { fetchOverview, type Overview } from "./api";
import OverviewTab from "./components/Overview";
import Console from "./components/Console";

type Tab = "overview" | "console";

const POLL_MS = 2000;

function initialTab(): Tab {
  return window.location.hash === "#console" ? "console" : "overview";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(initialTab);

  const selectTab = (next: Tab) => {
    setTab(next);
    window.location.hash = next === "console" ? "console" : "";
  };

  const [overview, setOverview] = useState<Overview | null>(null);
  const [reachable, setReachable] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const data = await fetchOverview();
        if (!alive) return;
        setOverview(data);
        setReachable(data.connected);
      } catch {
        if (alive) setReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <img src="/pivot-mark.png" alt="pivotdb" />
        </div>
        <nav className="tabs">
          <button
            className={`tab ${tab === "overview" ? "active" : ""}`}
            onClick={() => selectTab("overview")}
          >
            Overview
          </button>
          <button
            className={`tab ${tab === "console" ? "active" : ""}`}
            onClick={() => selectTab("console")}
          >
            SQL Console
          </button>
        </nav>
        <div className="spacer" />
        <div className="conn">
          <span className={`dot ${reachable ? "live" : "down"}`} />
          {reachable ? (
            <span className="store">{overview?.store ?? "connected"}</span>
          ) : (
            "engine unreachable"
          )}
        </div>
      </header>

      <main className="main">
        {tab === "overview" ? (
          <OverviewTab overview={overview} reachable={reachable} />
        ) : (
          <Console tables={overview?.tables ?? []} />
        )}
      </main>
    </div>
  );
}
