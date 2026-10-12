// Pure geometry and editing logic for the zone editor, unit-tested in tests/zones-model.test.ts.
// Points are normalized frame coordinates (0..1, origin top-left), as stored in cameras.yaml.

export type Pt = [number, number];

export interface Zone {
  name: string;
  polygon: Pt[];
}

export const NAME_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

/** Round to 4 decimals, as the server writes them (~0.1 px at 1280 wide). */
export function round4(p: Pt): Pt {
  return [Math.round(p[0] * 1e4) / 1e4, Math.round(p[1] * 1e4) / 1e4];
}

/** Distance in pixels between two normalized points on a w x h frame. */
export function pixelDistance(a: Pt, b: Pt, w: number, h: number): number {
  return Math.hypot((a[0] - b[0]) * w, (a[1] - b[1]) * h);
}

/** The vertex nearest to p within `radiusPx`, searching the selected zone first. */
export function hitVertex(
  zones: Zone[],
  p: Pt,
  w: number,
  h: number,
  radiusPx: number,
  selected: number | null,
): { zone: number; index: number } | null {
  const order = zones.map((_, i) => i);
  if (selected !== null) order.sort((a, b) => (a === selected ? -1 : b === selected ? 1 : 0));
  for (const zi of order) {
    const poly = zones[zi]!.polygon;
    let best = -1;
    let bestD = radiusPx;
    poly.forEach((v, i) => {
      const d = pixelDistance(v, p, w, h);
      if (d <= bestD) {
        best = i;
        bestD = d;
      }
    });
    if (best >= 0) return { zone: zi, index: best };
  }
  return null;
}

function distanceToSegmentPx(p: Pt, a: Pt, b: Pt, w: number, h: number): number {
  const ax = a[0] * w, ay = a[1] * h, bx = b[0] * w, by = b[1] * h, px = p[0] * w, py = p[1] * h;
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** Index at which to insert p if it lies within `tolPx` of an edge of `polygon`, else null. */
export function edgeInsertIndex(polygon: Pt[], p: Pt, w: number, h: number, tolPx: number): number | null {
  let best: number | null = null;
  let bestD = tolPx;
  for (let i = 0; i < polygon.length; i++) {
    const d = distanceToSegmentPx(p, polygon[i]!, polygon[(i + 1) % polygon.length]!, w, h);
    if (d <= bestD) {
      best = i + 1;
      bestD = d;
    }
  }
  return best;
}

/** Ray casting; points on the boundary may land either way (only used for selection). */
export function pointInPolygon(p: Pt, polygon: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]!;
    const [xj, yj] = polygon[j]!;
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function orient(a: Pt, b: Pt, c: Pt): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function onSegment(a: Pt, b: Pt, p: Pt): boolean {
  return (
    Math.min(a[0], b[0]) <= p[0] && p[0] <= Math.max(a[0], b[0]) &&
    Math.min(a[1], b[1]) <= p[1] && p[1] <= Math.max(a[1], b[1])
  );
}

function segmentsTouch(a: Pt, b: Pt, c: Pt, d: Pt): boolean {
  const o1 = orient(a, b, c), o2 = orient(a, b, d), o3 = orient(c, d, a), o4 = orient(c, d, b);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
    return true;
  }
  return (
    (o1 === 0 && onSegment(a, b, c)) || (o2 === 0 && onSegment(a, b, d)) ||
    (o3 === 0 && onSegment(c, d, a)) || (o4 === 0 && onSegment(c, d, b))
  );
}

/** Same rule as the server (vms.storage.catalog.is_simple): no repeated vertex and no two
 * non-adjacent edges touching. Lets the editor warn before saving. */
export function isSimple(polygon: Pt[]): boolean {
  const keys = new Set(polygon.map((p) => `${p[0]},${p[1]}`));
  if (keys.size !== polygon.length) return false;
  const n = polygon.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const adjacent = j === i + 1 || (i === 0 && j === n - 1);
      if (adjacent) continue;
      if (segmentsTouch(polygon[i]!, polygon[(i + 1) % n]!, polygon[j]!, polygon[(j + 1) % n]!)) {
        return false;
      }
    }
  }
  return true;
}

export function area(polygon: Pt[]): number {
  let s = 0;
  polygon.forEach((p, i) => {
    const q = polygon[(i + 1) % polygon.length]!;
    s += p[0] * q[1] - q[0] * p[1];
  });
  return Math.abs(s) / 2;
}

/** Problems that would make the server refuse to save, one message per problem. */
export function zoneProblems(zones: Zone[]): string[] {
  const problems: string[] = [];
  const seen = new Set<string>();
  for (const z of zones) {
    const label = z.name || "(unnamed)";
    if (!NAME_PATTERN.test(z.name)) {
      problems.push(`${label}: names are lowercase letters, digits and _ and start with a letter`);
    }
    if (seen.has(z.name)) problems.push(`${label}: duplicate name`);
    seen.add(z.name);
    if (z.polygon.length < 3) problems.push(`${label}: needs at least 3 points`);
    else if (!isSimple(z.polygon)) problems.push(`${label}: edges cross each other`);
    else if (area(z.polygon) === 0) problems.push(`${label}: zero area`);
  }
  return problems;
}

/** First free name like zone_1, zone_2, ... (unique across `taken`). */
export function suggestName(taken: Iterable<string>): string {
  const names = new Set(taken);
  for (let i = 1; ; i++) {
    if (!names.has(`zone_${i}`)) return `zone_${i}`;
  }
}
