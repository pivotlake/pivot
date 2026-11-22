// Zooming into a drawing: showing a part of it over the whole of the room it
// has. Every drawing zooms along its width. One that is squeezed to fit its
// height zooms down it as well, until it is as tall as it is drawn in full.

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type ReactNode,
} from "react";

// A stretch of a drawing's width or height, the whole of it running from 0
// to 1.
export type Span = { start: number; end: number };

// A place in what is shown of a drawing, or a distance over it: `across` in
// shares of the shown width, `down` in shares of the shown height.
export type Place = { across: number; down: number };

const WHOLE: Span = { start: 0, end: 1 };
const CENTRE: Place = { across: 0.5, down: 0.5 };
// The most a drawing is magnified along its width.
const DEEPEST_ZOOM = 10_000;
// The deepest zoom down a drawing whose whole height is always shown.
export const WIDTH_ONLY = 1;
// How much one pixel of wheel travel magnifies, as an exponent.
const WHEEL_ZOOM_RATE = 0.002;
// A wheel that reports lines rather than pixels moves this many pixels a line.
const WHEEL_LINE_PX = 16;
const BUTTON_ZOOM_STEP = 2;
// How much of the shown width a stretch takes when the drawing is zoomed to
// it, the rest being what lies either side of it.
const FITTED_STRETCH_SHARE = 0.8;
// How far the pointer moves with its button down before it is dragging the
// drawing, so a click that slips a little is still a click.
const DRAG_START_PX = 3;
// The primary button among the buttons a pointer event reports as down.
const PRIMARY_BUTTON = 1;

export type Zoom = {
  // The part of the drawing's width that is shown.
  shown: Span;
  // The part of the drawing's height that is shown.
  shownDown: Span;
  // How many times wider than whole the drawing is shown.
  factor: number;
  // Whether all of the drawing is shown.
  isWhole: boolean;
  // Magnifies what is shown `factor` times (below 1, shrinks it), keeping in
  // place what is at `anchor`.
  zoomBy(factor: number, anchor: Place): void;
  // Moves what is shown right and down by `distance`.
  panBy(distance: Place): void;
  showWhole(): void;
  // Zooms to `stretch` of the drawing's width, with what is at `downCentre`
  // of its height in the middle. Shows the whole again when that is what is
  // shown already.
  showStretch(stretch: Span, downCentre: number): void;
};

// The span `length` long that starts at `start`, moved to lie inside the
// drawing.
function fitSpan(start: number, length: number): Span {
  const fittedStart = Math.min(Math.max(start, 0), 1 - length);
  return { start: fittedStart, end: fittedStart + length };
}

// `span` made `length` long, keeping in place what is at `anchor` of it, 0
// at its start and 1 at its end.
function resizeSpan(span: Span, length: number, anchor: number): Span {
  if (length === 1) return WHOLE;
  return fitSpan(span.start + anchor * (span.end - span.start - length), length);
}

// What is shown of a drawing, along its width and down its height.
type View = { across: Span; down: Span };

const WHOLE_VIEW: View = { across: WHOLE, down: WHOLE };

// The zoom of a drawing that is magnified at most `deepestDown` times down
// its height, `WIDTH_ONLY` for one whose whole height is always shown.
//
// The two directions zoom together, as a map does: the drawing is as many
// times taller as it is wider, until `deepestDown`, past which only its
// width grows.
export function useZoom(deepestDown: number): Zoom {
  const [view, setView] = useState(WHOLE_VIEW);

  // The part of the height that goes with `across`. `down` is where the view
  // was left, which is the wrong length once `deepestDown` has changed under
  // it.
  const readShownDown = useCallback(
    (across: Span, down: Span, anchor: number) =>
      resizeSpan(down, 1 / Math.min(1 / (across.end - across.start), deepestDown), anchor),
    [deepestDown],
  );

  const zoomBy = useCallback(
    (factor: number, anchor: Place) => {
      setView((current) => {
        const length = current.across.end - current.across.start;
        const zoomedLength = Math.min(Math.max(length / factor, 1 / DEEPEST_ZOOM), 1);
        if (zoomedLength === 1) return WHOLE_VIEW;
        const across = resizeSpan(current.across, zoomedLength, anchor.across);
        return { across, down: readShownDown(across, current.down, anchor.down) };
      });
    },
    [readShownDown],
  );
  const panBy = useCallback(
    (distance: Place) => {
      setView((current) => {
        if (current === WHOLE_VIEW) return WHOLE_VIEW;
        const length = current.across.end - current.across.start;
        const down = readShownDown(current.across, current.down, CENTRE.down);
        const downLength = down.end - down.start;
        return {
          across: fitSpan(current.across.start + distance.across * length, length),
          down: fitSpan(down.start + distance.down * downLength, downLength),
        };
      });
    },
    [readShownDown],
  );
  const showWhole = useCallback(() => setView(WHOLE_VIEW), []);
  const showStretch = useCallback(
    (stretch: Span, downCentre: number) => {
      const length = Math.min(
        Math.max((stretch.end - stretch.start) / FITTED_STRETCH_SHARE, 1 / DEEPEST_ZOOM),
        1,
      );
      if (length === 1) {
        setView(WHOLE_VIEW);
        return;
      }
      const across = fitSpan((stretch.start + stretch.end - length) / 2, length);
      const downLength = 1 / Math.min(1 / length, deepestDown);
      const fitted = { across, down: fitSpan(downCentre - downLength / 2, downLength) };
      setView((current) => {
        const isShownAlready =
          Math.abs(current.across.start - across.start) < length / 1000 &&
          Math.abs(current.across.end - across.end) < length / 1000 &&
          Math.abs(current.down.start - fitted.down.start) < downLength / 1000;
        return isShownAlready ? WHOLE_VIEW : fitted;
      });
    },
    [deepestDown],
  );

  return {
    shown: view.across,
    shownDown: readShownDown(view.across, view.down, CENTRE.down),
    factor: 1 / (view.across.end - view.across.start),
    isWhole: view === WHOLE_VIEW,
    zoomBy,
    panBy,
    showWhole,
    showStretch,
  };
}

// Where a stretch of the drawing falls in what is shown, in shares of the
// shown width: the part of it in view, and whether it goes on past either
// edge. One wholly out of view has a `width` of 0 or less.
export function placeInView(
  shown: Span,
  stretch: Span,
): { left: number; width: number; isCutLeft: boolean; isCutRight: boolean } {
  const shownLength = shown.end - shown.start;
  const start = (stretch.start - shown.start) / shownLength;
  const end = (stretch.end - shown.start) / shownLength;
  const left = Math.max(start, 0);
  return { left, width: Math.min(end, 1) - left, isCutLeft: start < 0, isCutRight: end > 1 };
}

// The room a zoomed drawing has: turning the wheel over it zooms about the
// pointer, dragging it pans, as scrolling it sideways does along its width,
// and a double click on it shows the whole again. Its children are drawn
// from its width in pixels, and one of them that takes a double click for
// itself keeps the click from reaching the surface.
export function ZoomSurface({
  zoom,
  className,
  style,
  label,
  children,
}: {
  zoom: Zoom;
  className: string;
  style?: CSSProperties;
  // What the drawing shows, for a reader who cannot see it.
  label: string;
  children: (widthPx: number) => ReactNode;
}) {
  const surface = useRef<HTMLDivElement>(null);
  const [widthPx, setWidthPx] = useState(0);
  const { isWhole, zoomBy, panBy, showWhole } = zoom;

  useEffect(() => {
    const element = surface.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidthPx(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // React listens to the wheel passively, and a passive listener cannot keep
  // the page from scrolling, so this one is attached to the element itself.
  useEffect(() => {
    const element = surface.current;
    if (!element) return;
    const followWheel = (event: WheelEvent) => {
      const box = element.getBoundingClientRect();
      const pixelsPerUnit = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? WHEEL_LINE_PX : 1;
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        if (isWhole) return;
        event.preventDefault();
        panBy({ across: (event.deltaX * pixelsPerUnit) / box.width, down: 0 });
        return;
      }
      // Zooming out of the whole drawing has nothing more to show, so that
      // turn of the wheel scrolls the page.
      if (isWhole && event.deltaY > 0) return;
      event.preventDefault();
      zoomBy(Math.exp(-event.deltaY * pixelsPerUnit * WHEEL_ZOOM_RATE), {
        across: (event.clientX - box.left) / box.width,
        down: (event.clientY - box.top) / box.height,
      });
    };
    element.addEventListener("wheel", followWheel, { passive: false });
    return () => element.removeEventListener("wheel", followWheel);
  }, [isWhole, panBy, zoomBy]);

  // Where the pointer was last seen with its button down over the drawing.
  // It holds the drawing once it has moved far enough to be dragging it:
  // holding it from the press would take the clicks meant for what is drawn.
  const draggedFrom = useRef<{ x: number; y: number }>();
  const startDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (isWhole || event.button !== 0) return;
    draggedFrom.current = { x: event.clientX, y: event.clientY };
  };
  const followDrag = (event: PointerEvent<HTMLDivElement>) => {
    if (!draggedFrom.current) return;
    // A button let go outside the drawing, before the drawing was held, is
    // not reported to it: the pointer coming back with no button down is
    // not dragging.
    if ((event.buttons & PRIMARY_BUTTON) === 0) {
      draggedFrom.current = undefined;
      return;
    }
    if (!event.currentTarget.hasPointerCapture(event.pointerId)) {
      const moved = Math.hypot(
        event.clientX - draggedFrom.current.x,
        event.clientY - draggedFrom.current.y,
      );
      if (moved < DRAG_START_PX) return;
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    const box = event.currentTarget.getBoundingClientRect();
    panBy({
      across: (draggedFrom.current.x - event.clientX) / box.width,
      down: (draggedFrom.current.y - event.clientY) / box.height,
    });
    draggedFrom.current = { x: event.clientX, y: event.clientY };
  };
  const endDrag = () => {
    draggedFrom.current = undefined;
  };

  return (
    <div
      ref={surface}
      className={`${className} zoom-surface${isWhole ? "" : " is-zoomed"}`}
      style={style}
      role="img"
      aria-label={label}
      onPointerDown={startDrag}
      onPointerMove={followDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={showWhole}
    >
      {children(widthPx)}
    </div>
  );
}

function formatFactor(factor: number): string {
  return factor < 10 ? `${Number(factor.toFixed(1))}×` : `${Math.round(factor).toLocaleString()}×`;
}

// The buttons that zoom a drawing, with how far in it is.
export function ZoomControls({ zoom }: { zoom: Zoom }) {
  const { isWhole } = zoom;
  return (
    <div className="zoom-controls">
      <button
        type="button"
        aria-label="Zoom out"
        disabled={isWhole}
        onClick={() => zoom.zoomBy(1 / BUTTON_ZOOM_STEP, CENTRE)}
      >
        −
      </button>
      <span className="zoom-factor">{formatFactor(zoom.factor)}</span>
      <button
        type="button"
        aria-label="Zoom in"
        disabled={zoom.factor >= DEEPEST_ZOOM}
        onClick={() => zoom.zoomBy(BUTTON_ZOOM_STEP, CENTRE)}
      >
        +
      </button>
      <button type="button" disabled={isWhole} onClick={zoom.showWhole}>
        Reset
      </button>
      <span className="zoom-hint">Scroll, drag, or double-click a file</span>
    </div>
  );
}
