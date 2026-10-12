// Query box: ask, then show the answer as a number, table, chart or frame grid, with clear
// states for abstention and errors, and a collapsible panel with the plan and compiled SQL.

import { ask, type QueryResponse } from "./api";
import { renderChart } from "./chart";
import { h } from "./dom";
import { abstainTitle, chooseView, errorTitle, formatCell, toSeries } from "./view";

export function mountQuery(form: HTMLFormElement, input: HTMLInputElement, out: HTMLElement): (q: string) => void {
  const button = form.querySelector("button")!;

  const run = async (question: string): Promise<void> => {
    const q = question.trim();
    if (!q) return;
    input.value = q;
    button.disabled = true;
    out.replaceChildren(h("div", { class: "pending" }, "Planning and running…"));
    try {
      out.replaceChildren(renderResponse(await ask(q)));
    } catch (err) {
      out.replaceChildren(
        callout("error", "Request failed", (err as Error).message),
      );
    } finally {
      button.disabled = false;
    }
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    void run(input.value);
  });
  return (q: string) => void run(q);
}

function callout(kind: "error" | "abstain", title: string, text: string, items: string[] = []): HTMLElement {
  return h(
    "div",
    { class: `callout ${kind}`, role: kind === "error" ? "alert" : "status" },
    h("div", { class: "callout-title" }, title),
    h("p", {}, text),
    items.length ? h("ul", {}, ...items.map((i) => h("li", {}, i))) : null,
  );
}

export function renderResponse(r: QueryResponse): HTMLElement {
  const view = chooseView(r);
  const body = h("div", { class: "result" });
  if (view === "error" && r.error) {
    const detail = r.error.type === "sql_rejected" || r.error.type === "execution_error" || r.error.type === "timeout"
      ? `${r.answer} (${r.error.detail})`
      : r.answer;
    body.append(callout("error", errorTitle(r.error.type), detail, r.error.errors));
  } else if (view === "abstention") {
    body.append(callout("abstain", abstainTitle(r.abstain_reason), r.answer));
  } else {
    body.append(h("p", { class: "answer" }, r.answer));
    if (view === "scalar") body.append(scalar(r));
    if (view === "chart") body.append(renderChart(toSeries(r.columns, r.rows)));
    if (view === "frames") body.append(frames(r));
    if (view === "table" || view === "chart") body.append(table(r));
  }
  body.append(debugPanel(r));
  return body;
}

function scalar(r: QueryResponse): HTMLElement {
  const value = r.rows[0]?.[0] ?? null;
  return h("div", { class: "scalar" }, formatCell(value, r.columns[0] ?? ""));
}

function table(r: QueryResponse): HTMLElement {
  const visible = r.columns.filter((c) => c !== "image_url" && c !== "object_key");
  const at = visible.map((c) => r.columns.indexOf(c));
  return h(
    "div",
    { class: "table-wrap" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, ...visible.map((c) => h("th", {}, c)))),
      h(
        "tbody",
        {},
        ...r.rows.map((row) =>
          h("tr", {}, ...at.map((i, k) => h("td", {}, formatCell(row[i] ?? null, visible[k]!)))),
        ),
      ),
    ),
    r.truncated ? h("p", { class: "muted" }, `Showing the first ${r.row_count} rows.`) : null,
  );
}

function frames(r: QueryResponse): HTMLElement {
  const url = r.columns.indexOf("image_url");
  const ts = r.columns.indexOf("ts");
  const cam = r.columns.indexOf("camera");
  return h(
    "div",
    { class: "frames" },
    ...r.rows.map((row) =>
      h(
        "figure",
        {},
        h("img", { src: String(row[url]), loading: "lazy", alt: "matching frame" }),
        h("figcaption", {}, `${row[cam]} · ${formatCell(row[ts] ?? null, "ts")}`),
      ),
    ),
  );
}

function debugPanel(r: QueryResponse): HTMLElement {
  const timings = Object.entries(r.timings_ms)
    .map(([k, v]) => `${k} ${v.toFixed(1)} ms`)
    .join(" · ");
  const meta = `${r.llm}${r.repaired ? " · repaired once" : ""} · ${r.tokens} tokens · ${timings}`;
  return h(
    "details",
    { class: "debug" },
    h("summary", {}, "Plan and SQL"),
    h("p", { class: "muted" }, meta),
    h("h4", {}, "QueryPlan"),
    h("pre", {}, r.plan ? JSON.stringify(r.plan, null, 2) : "(no valid plan)"),
    h("h4", {}, "Compiled SQL (validated; values are bound parameters)"),
    h("pre", {}, r.compiled_sql ?? "(no SQL: nothing was executed)"),
    Object.keys(r.params).length
      ? h("pre", { class: "params" }, JSON.stringify(r.params, null, 2))
      : null,
  );
}
