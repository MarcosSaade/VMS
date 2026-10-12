// Pure presentation logic, unit-tested in tests/view.test.ts: which view a response gets,
// how a result becomes chart series, labels, colors and formatting.

import type { Cell, ErrorType, QueryResponse } from "./api";

export type View = "scalar" | "table" | "chart" | "frames" | "abstention" | "error";

export interface Series {
  name: string;
  points: { x: string; y: number }[];
}

/** The main view for a response. Grouped counts with one key and a handful of groups get a bar
 * chart; time series always do. Everything else follows the result kind. */
export function chooseView(r: QueryResponse): View {
  if (r.status === "error") return "error";
  if (r.status === "abstained") return "abstention";
  switch (r.result_kind) {
    case "scalar":
      return "scalar";
    case "time_series":
      return "chart";
    case "frames":
      return "frames";
    case "table": {
      const metric = metricColumn(r.columns);
      const keys = r.columns.filter((c) => c !== metric);
      return metric !== null && keys.length === 1 && r.rows.length >= 2 && r.rows.length <= 24
        ? "chart"
        : "table";
    }
    default:
      return "table";
  }
}

export function metricColumn(columns: string[]): string | null {
  if (columns.includes("n")) return "n";
  if (columns.includes("value")) return "value";
  return null;
}

/** Chart series from rows: one series per combination of the non-metric, non-bucket columns,
 * x = bucket (time series) or the single group key (bar chart). */
export function toSeries(columns: string[], rows: Cell[][]): Series[] {
  const metric = metricColumn(columns);
  if (metric === null) return [];
  const yAt = columns.indexOf(metric);
  const xCol = columns.includes("bucket") ? "bucket" : columns.find((c) => c !== metric);
  if (xCol === undefined) return [];
  const xAt = columns.indexOf(xCol);
  const groupAts = columns
    .map((c, i) => [c, i] as const)
    .filter(([c]) => c !== metric && c !== xCol)
    .map(([, i]) => i);
  const byName = new Map<string, Series>();
  for (const row of rows) {
    const name = groupAts.length ? groupAts.map((i) => String(row[i])).join(" / ") : metric;
    let series = byName.get(name);
    if (series === undefined) {
      series = { name, points: [] };
      byName.set(name, series);
    }
    series.points.push({ x: formatX(row[xAt] ?? null, xCol), y: Number(row[yAt] ?? 0) });
  }
  return [...byName.values()];
}

function formatX(value: Cell, column: string): string {
  if (column === "bucket" && typeof value === "string") {
    // Camera-local wall-clock timestamps without offset: "2026-10-12T14:00:00".
    return value.slice(0, 16).replace("T", " ");
  }
  return String(value);
}

/** "Nice" axis maximum: 1, 2 or 5 times a power of ten, at least the data maximum. */
export function niceMax(max: number): number {
  if (!(max > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(max));
  for (const step of [1, 2, 5, 10]) {
    if (step * power >= max) return step * power;
  }
  return 10 * power;
}

const ERROR_TITLES: Record<ErrorType, string> = {
  planner_unavailable: "Planner unavailable",
  invalid_plan: "Could not plan this question",
  compile_error: "Plan could not be compiled",
  sql_rejected: "Blocked by the SQL validator",
  timeout: "Query timed out",
  execution_error: "Database error",
};

export function errorTitle(type: ErrorType): string {
  return ERROR_TITLES[type];
}

const ABSTAIN_TITLES: Record<string, string> = {
  ambiguous: "Ambiguous question",
  unanswerable: "Not in the data",
  unsupported: "Not supported yet",
  disallowed: "Not allowed",
};

export function abstainTitle(reason: string | null): string {
  return (reason !== null && ABSTAIN_TITLES[reason]) || "Not answered";
}

// Colors per tracked class; anything else is grey.
const CLASS_COLORS: Record<string, string> = {
  person: "#4ade80",
  car: "#60a5fa",
  truck: "#f59e0b",
  bus: "#f472b6",
  bicycle: "#a78bfa",
  motorcycle: "#22d3ee",
};

export function classColor(cls: string): string {
  return CLASS_COLORS[cls] ?? "#94a3b8";
}

/** Seconds since `iso`, as a short label: "now", "4 s", "3 min", "2 h". */
export function formatAge(iso: string, nowMs: number): string {
  const seconds = Math.max(0, Math.round((nowMs - Date.parse(iso)) / 1000));
  if (seconds < 2) return "now";
  if (seconds < 120) return `${seconds} s ago`;
  if (seconds < 7200) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)} h ago`;
}

export function formatCell(value: Cell, column: string): string {
  if (value === null) return "—";
  if (typeof value === "number") {
    if (column === "confidence") return value.toFixed(2);
    return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toFixed(2);
  }
  if (typeof value === "string" && /^\d{4}-\d\d-\d\dT\d\d:\d\d/.test(value)) {
    return value.slice(0, 19).replace("T", " ");
  }
  return String(value);
}

/** Counts per class for a tile caption, most common first: "3 person · 1 car". */
export function classSummary(classes: string[]): string {
  const counts = new Map<string, number>();
  for (const c of classes) counts.set(c, (counts.get(c) ?? 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([c, n]) => `${n} ${c}`)
    .join(" · ");
}
