import { describe, expect, it } from "vitest";

import type { QueryResponse } from "../src/api";
import {
  abstainTitle,
  chooseView,
  classSummary,
  errorTitle,
  formatAge,
  formatCell,
  niceMax,
  toSeries,
} from "../src/view";

function response(overrides: Partial<QueryResponse>): QueryResponse {
  return {
    query_id: "q",
    question: "?",
    status: "ok",
    answer: "",
    result_kind: "table",
    columns: [],
    rows: [],
    row_count: 0,
    truncated: false,
    plan: null,
    compiled_sql: null,
    params: {},
    abstain_reason: null,
    error: null,
    llm: "canned",
    repaired: false,
    tokens: 0,
    timings_ms: {},
    ...overrides,
  };
}

describe("chooseView", () => {
  it("shows errors and abstentions before any result kind", () => {
    const error = { type: "sql_rejected" as const, detail: "x", errors: [] };
    expect(chooseView(response({ status: "error", error, result_kind: "table" }))).toBe("error");
    expect(chooseView(response({ status: "abstained", result_kind: "abstention" }))).toBe("abstention");
  });

  it("follows the result kind", () => {
    expect(chooseView(response({ result_kind: "scalar" }))).toBe("scalar");
    expect(chooseView(response({ result_kind: "time_series" }))).toBe("chart");
    expect(chooseView(response({ result_kind: "frames" }))).toBe("frames");
  });

  it("charts small one-key groupings and tabulates the rest", () => {
    const grouped = { columns: ["camera", "n"], rows: [["a", 1], ["b", 2]] };
    expect(chooseView(response(grouped))).toBe("chart");
    expect(chooseView(response({ columns: ["camera", "n"], rows: [["a", 1]] }))).toBe("table");
    const twoKeys = { columns: ["camera", "zone", "n"], rows: [["a", "z", 1], ["b", "z", 2]] };
    expect(chooseView(response(twoKeys))).toBe("table");
    const list = { columns: ["event_id", "ts"], rows: [["e", "t"], ["f", "u"]] };
    expect(chooseView(response(list))).toBe("table");
  });
});

describe("toSeries", () => {
  it("makes one series per group with buckets on x", () => {
    const series = toSeries(
      ["zone", "bucket", "n"],
      [
        ["register_1", "2026-10-12T09:00:00", 3],
        ["register_2", "2026-10-12T09:00:00", 1],
        ["register_1", "2026-10-12T10:00:00", 5],
      ],
    );
    expect(series).toEqual([
      { name: "register_1", points: [{ x: "2026-10-12 09:00", y: 3 }, { x: "2026-10-12 10:00", y: 5 }] },
      { name: "register_2", points: [{ x: "2026-10-12 09:00", y: 1 }] },
    ]);
  });

  it("uses the group key as x when there is no bucket", () => {
    expect(toSeries(["camera", "value"], [["entrance", 1.5]])).toEqual([
      { name: "value", points: [{ x: "entrance", y: 1.5 }] },
    ]);
  });

  it("returns nothing without a metric column", () => {
    expect(toSeries(["event_id"], [["e"]])).toEqual([]);
  });
});

describe("formatting", () => {
  it("rounds axis maxima up to 1, 2 or 5 times a power of ten", () => {
    expect([0, 0.3, 1, 3, 7, 11, 260].map(niceMax)).toEqual([1, 0.5, 1, 5, 10, 20, 500]);
  });

  it("formats ages relative to now", () => {
    const now = Date.parse("2026-10-12T12:00:00Z");
    expect(formatAge("2026-10-12T12:00:00Z", now)).toBe("now");
    expect(formatAge("2026-10-12T11:59:30Z", now)).toBe("30 s ago");
    expect(formatAge("2026-10-12T11:50:00Z", now)).toBe("10 min ago");
    expect(formatAge("2026-10-12T09:00:00Z", now)).toBe("3 h ago");
  });

  it("formats cells", () => {
    expect(formatCell(null, "n")).toBe("—");
    expect(formatCell(12345, "n")).toBe("12,345");
    expect(formatCell(0.876, "confidence")).toBe("0.88");
    expect(formatCell(61.25, "value")).toBe("61.25");
    expect(formatCell("2026-10-12T13:30:05.123+00:00", "ts")).toBe("2026-10-12 13:30:05");
    expect(formatCell("entrance", "camera")).toBe("entrance");
  });

  it("summarizes classes, most common first", () => {
    expect(classSummary(["car", "person", "person", "bus"])).toBe("2 person · 1 bus · 1 car");
    expect(classSummary([])).toBe("");
  });

  it("titles every error type and abstain reason", () => {
    expect(errorTitle("sql_rejected")).toBe("Blocked by the SQL validator");
    expect(errorTitle("invalid_plan")).toBe("Could not plan this question");
    expect(abstainTitle("unsupported")).toBe("Not supported yet");
    expect(abstainTitle(null)).toBe("Not answered");
  });
});
