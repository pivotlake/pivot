// Drag-resizable horizontal splitter. Lets the user grab the divider between
// two stacked panes and drag it up/down to change the height of the pane
// above. Heights are tracked by the parent.

import { useEffect, useRef } from "react";

export function Splitter({
  onResize,
}: {
  /// Called continuously during a drag with the delta-y in pixels (positive
  /// means dragged downward).
  onResize: (deltaY: number) => void;
}) {
  const startY = useRef<number | null>(null);

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (startY.current == null) return;
      const dy = e.clientY - startY.current;
      startY.current = e.clientY;
      onResize(dy);
    }
    function onUp() {
      startY.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [onResize]);

  return (
    <div
      onMouseDown={(e) => {
        startY.current = e.clientY;
        document.body.style.cursor = "ns-resize";
        document.body.style.userSelect = "none";
      }}
      style={{
        height: 8,
        cursor: "ns-resize",
        background: "oklch(0.93 0.005 250)",
        borderTop: "1px solid oklch(0.88 0.005 250)",
        borderBottom: "1px solid oklch(0.88 0.005 250)",
        flex: "0 0 auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "background 100ms",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background = "oklch(0.82 0.06 250)";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = "oklch(0.93 0.005 250)";
      }}
    >
      {/* Grip dots — visual hint that this is draggable. */}
      <div
        style={{
          width: 28,
          height: 2,
          borderRadius: 1,
          background: "oklch(0.65 0.01 250)",
          opacity: 0.6,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
