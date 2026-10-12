import { describe, expect, it } from "vitest";

import {
  edgeInsertIndex,
  hitVertex,
  isSimple,
  pointInPolygon,
  round4,
  suggestName,
  zoneProblems,
  type Pt,
} from "../src/zones-model";

const SQUARE: Pt[] = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]];
const W = 1280;
const H = 720;

describe("hit testing", () => {
  it("finds the vertex within the pixel radius, preferring the selected zone", () => {
    const other: Pt[] = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]];
    const zones = [{ name: "a", polygon: SQUARE }, { name: "b", polygon: other }];
    expect(hitVertex(zones, [0.5 + 5 / W, 0.5], W, H, 10, null)).toEqual({ zone: 0, index: 2 });
    expect(hitVertex(zones, [0.1, 0.1], W, H, 10, 1)).toEqual({ zone: 1, index: 0 });
    expect(hitVertex(zones, [0.8, 0.8], W, H, 10, null)).toBeNull();
  });

  it("inserts on the nearest edge only within tolerance", () => {
    expect(edgeInsertIndex(SQUARE, [0.3, 0.1 + 3 / H], W, H, 8)).toBe(1);
    expect(edgeInsertIndex(SQUARE, [0.1, 0.3], W, H, 8)).toBe(4); // closing edge 3 -> 0
    expect(edgeInsertIndex(SQUARE, [0.3, 0.3], W, H, 8)).toBeNull();
  });

  it("tests points against polygons", () => {
    expect(pointInPolygon([0.3, 0.3], SQUARE)).toBe(true);
    expect(pointInPolygon([0.6, 0.3], SQUARE)).toBe(false);
  });
});

describe("validation mirrors the server", () => {
  it("rejects self-intersecting and repeated points", () => {
    expect(isSimple(SQUARE)).toBe(true);
    expect(isSimple([[0, 0], [1, 1], [1, 0], [0, 1]])).toBe(false); // bow tie
    expect(isSimple([[0, 0], [1, 0], [1, 0], [0, 1]])).toBe(false);
  });

  it("lists every problem", () => {
    expect(
      zoneProblems([
        { name: "door", polygon: SQUARE },
        { name: "door", polygon: [[0, 0], [1, 0]] },
        { name: "Bad Name", polygon: [[0, 0], [1, 1], [1, 0], [0, 1]] },
        { name: "flat", polygon: [[0, 0], [0.5, 0], [1, 0]] },
      ]),
    ).toEqual([
      "door: duplicate name",
      "door: needs at least 3 points",
      "Bad Name: names are lowercase letters, digits and _ and start with a letter",
      "Bad Name: edges cross each other",
      "flat: zero area", // collinear: every edge is adjacent, so only the area check fails
    ]);
    expect(zoneProblems([{ name: "ok_1", polygon: SQUARE }])).toEqual([]);
  });
});

describe("helpers", () => {
  it("rounds like the server", () => {
    expect(round4([0.123456, 0.99995])).toEqual([0.1235, 1]);
  });

  it("suggests free names", () => {
    expect(suggestName([])).toBe("zone_1");
    expect(suggestName(["zone_1", "zone_3", "front_door"])).toBe("zone_2");
  });
});
