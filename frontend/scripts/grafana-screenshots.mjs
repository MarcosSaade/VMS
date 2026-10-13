// Screenshots of the Grafana dashboards for the README and write-up (task 2.2).
//
//   CHROMIUM_PATH=/path/to/chrome npm run screenshots:grafana -- http://localhost:3000 ../docs/images [from] [to]
//
// Needs the stack with live data (make demo, or loadgen against the app containers) and a
// Chromium binary. `from` and `to` are Grafana time range values (default now-15m, now), for
// example Unix milliseconds of a run. Dashboards are opened in kiosk mode (no menus) as the
// anonymous viewer, in the light theme, at a size that fits each dashboard without scrolling.

import { mkdirSync, readFileSync } from "node:fs";

import { chromium } from "playwright-core";

const [base = "http://localhost:3000", outDir = "screenshots", from = "now-15m", to = "now"] =
  process.argv.slice(2);
const dashboards = [
  ["vms-pipeline", "grafana-pipeline.png"],
  ["vms-inference", "grafana-inference.png"],
  ["vms-query", "grafana-query.png"],
  ["vms-resources", "grafana-resources.png"],
];
const HEADER_PX = 90; // kiosk mode's dashboard controls and bottom margin
const executablePath = process.env.CHROMIUM_PATH;
if (!executablePath) throw new Error("set CHROMIUM_PATH to a Chromium or Chrome binary");

mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ executablePath });
const errors = [];
for (const [uid, file] of dashboards) {
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  page.on("console", (m) => m.type() === "error" && errors.push(`${uid}: ${m.text()}`));
  const query = new URLSearchParams({ kiosk: "true", theme: "light", from, to });
  await page.goto(`${base}/d/${uid}?${query}`, { waitUntil: "networkidle" });
  await page.waitForSelector(".react-grid-layout", { timeout: 30_000 });
  // Grafana scrolls inside its own container and renders panels lazily, so the window is
  // sized to the whole grid up front, from the dashboard JSON: 30 px rows plus 8 px margins.
  const json = new URL(`../../deploy/grafana/dashboards/${uid.slice(4)}.json`, import.meta.url);
  const grid = JSON.parse(readFileSync(json, "utf8"));
  const units = Math.max(...grid.panels.map((p) => p.gridPos.y + p.gridPos.h));
  await page.setViewportSize({ width: 1600, height: HEADER_PX + units * 38 });
  await page.waitForTimeout(5000); // let every panel finish its queries and draw
  const noData = await page.locator("text=No data").count();
  await page.screenshot({ path: `${outDir}/${file}` });
  console.log(`${uid} -> ${outDir}/${file}${noData ? ` (${noData} panels show "No data")` : ""}`);
  await page.close();
}
await browser.close();
if (errors.length) console.error("browser console errors:", errors);
