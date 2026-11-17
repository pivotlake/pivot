import { Position, type Rect, type XYPosition } from "@xyflow/react";

// Where clients sit around the server, which is at the origin. A wide window
// puts them in a column on either side of it; a tall one puts them in two
// columns above and below it, with a lane between the columns for their
// connections to run down.
export type GraphLayout = "wide" | "tall";

export type SlotPlacement = {
  position: XYPosition;
  // The side of the client card its connection leaves from.
  clientHandle: Position;
  // The side of the server card its connection arrives at.
  serverHandle: Position;
};

const WIDE_COLUMN_OFFSET = 470;
const WIDE_ROW_SPACING = 150;

const TALL_COLUMN_OFFSET = 172;
const TALL_FIRST_ROW_OFFSET = 315;
const TALL_ROW_SPACING = 142;

// The area the server and the first eight client slots cover. The view is
// fitted to this rather than to the nodes present, so it stays put as clients
// come and go.
export const GRAPH_BOUNDS: Record<GraphLayout, Rect> = {
  wide: { x: -620, y: -300, width: 1240, height: 600 },
  tall: { x: -310, y: -530, width: 620, height: 1060 },
};

// Picks the layout that can be shown larger in a viewport of this size.
export function chooseLayout(viewportWidth: number, viewportHeight: number): GraphLayout {
  const zoomToFit = (bounds: Rect) =>
    Math.min(viewportWidth / bounds.width, viewportHeight / bounds.height);
  return zoomToFit(GRAPH_BOUNDS.wide) >= zoomToFit(GRAPH_BOUNDS.tall) ? "wide" : "tall";
}

// Slots alternate between the two sides of the server and fill outwards from
// it, so a few clients sit close to the server and many spread evenly around.
export function placeSlot(layout: GraphLayout, slot: number): SlotPlacement {
  const isFirstSide = slot % 2 === 0;
  const rank = Math.floor(slot / 2);

  if (layout === "wide") {
    const rowsFromCenter = (Math.floor(rank / 2) + 0.5) * (rank % 2 === 0 ? -1 : 1);
    return {
      position: {
        x: isFirstSide ? -WIDE_COLUMN_OFFSET : WIDE_COLUMN_OFFSET,
        y: rowsFromCenter * WIDE_ROW_SPACING,
      },
      clientHandle: isFirstSide ? Position.Right : Position.Left,
      serverHandle: isFirstSide ? Position.Left : Position.Right,
    };
  }

  const isLeftColumn = rank % 2 === 0;
  const distanceFromServer = TALL_FIRST_ROW_OFFSET + Math.floor(rank / 2) * TALL_ROW_SPACING;
  return {
    position: {
      x: isLeftColumn ? -TALL_COLUMN_OFFSET : TALL_COLUMN_OFFSET,
      y: isFirstSide ? -distanceFromServer : distanceFromServer,
    },
    clientHandle: isLeftColumn ? Position.Right : Position.Left,
    serverHandle: isFirstSide ? Position.Top : Position.Bottom,
  };
}
