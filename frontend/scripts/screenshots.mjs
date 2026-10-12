// Screenshots of the UI for the README (task 1.8): the camera wall, then one per question.
//
//   CHROMIUM_PATH=/path/to/chrome npm run screenshots -- http://localhost:8070 ../docs/images
//
// Needs a running query service with data (make demo) and a Chromium binary; playwright-core
// ships no browsers. Questions default to the demo's canned ones, so no LLM is needed.

import { mkdirSync } from "node:fs";

import { chromium } from "playwright-core";

const [url = "http://localhost:8070", outDir = "screenshots", ...asked] = process.argv.slice(2);
const questions = asked.length
  ? asked
  : [
      "How many people did each camera see in the last hour?",
      "Give me the hourly count of people at the entrance today.",
      "Show me the latest frames with a car in them.",
      "What was the license plate of the last car in the parking lot?",
    ];
const executablePath = process.env.CHROMIUM_PATH;
if (!executablePath) throw new Error("set CHROMIUM_PATH to a Chromium or Chrome binary");

mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ executablePath });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
await page.goto(url);
await page.waitForSelector(".tile figcaption strong");
await page.waitForTimeout(3000); // let images load and one refresh pass
await page.screenshot({ path: `${outDir}/ui-cameras.png` });
let i = 0;
for (const q of questions) {
  await page.fill("#question", q);
  await page.click("#ask button");
  await page.waitForSelector("#result .result", { timeout: 60_000 });
  await page.evaluate(() => document.querySelector("#result details")?.setAttribute("open", ""));
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${outDir}/ui-query-${++i}.png`, fullPage: true });
  console.log(`${q} -> ${await page.textContent("#result .answer, #result .callout-title")}`);
}
await browser.close();
if (errors.length) {
  console.error("browser console errors:", errors);
  process.exit(1);
}
