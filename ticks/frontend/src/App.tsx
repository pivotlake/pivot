import React, { useEffect, useState } from "react";
import { Summary } from "./types";
import FunctionPicker from "./FunctionPicker";
import Annotated from "./Annotated";
import "./App.css";

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/summary")
      .then((r) => r.json())
      .then(setSummary)
      .catch((e) => console.error("summary fetch failed", e));
  }, []);

  if (!summary) return <div className="container">Loading…</div>;
  if (picked) {
    return (
      <Annotated
        funcName={picked}
        summary={summary}
        onBack={() => setPicked(null)}
      />
    );
  }
  return <FunctionPicker summary={summary} onPick={setPicked} />;
}
