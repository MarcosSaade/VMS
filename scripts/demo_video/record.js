// Records the VMS demo video: headless Chrome driven by Playwright, frames captured with the
// DevTools screencast, encoded to 1080p H.264 with ffmpeg. Needs the stack up with loadgen
// streaming (see run.sh). Usage: node record.js OUT.mp4 [UI_URL] [GRAFANA_URL]
// CHROME_PATH picks the browser (default /usr/bin/google-chrome); ffmpeg must be on PATH.
const { chromium } = require("playwright-core");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const out = path.resolve(process.argv[2] || "vms-demo.mp4");
const UI = process.argv[3] || "http://localhost:8070";
const GRAFANA = process.argv[4] || "http://localhost:3000";
const SLIDES = "file://" + path.join(__dirname, "slides.html");
const os = require("os");
const framesDir = fs.mkdtempSync(path.join(os.tmpdir(), "vms-demo-frames-"));

const CAPTION_CSS = `
  #vms-cap { position: fixed; left: 50%; bottom: 34px; transform: translateX(-50%);
    max-width: 88%; padding: 12px 22px; border-radius: 10px; z-index: 99999;
    background: rgba(8, 12, 26, .88); border: 1px solid #2c3d63; color: #e5e9f2;
    font: 600 20px/1.35 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
    box-shadow: 0 8px 30px rgba(0,0,0,.45); text-align: center; transition: opacity .3s; }
  #vms-cap small { display:block; font-weight: 450; font-size: 16px; color: #9fb0d0; margin-top: 3px; }`;

async function caption(page, title, sub = "") {
  await page.evaluate(
    ([css, title, sub]) => {
      if (!document.getElementById("vms-cap-css")) {
        const s = document.createElement("style");
        s.id = "vms-cap-css";
        s.textContent = css;
        document.head.append(s);
      }
      let cap = document.getElementById("vms-cap");
      if (!cap) {
        cap = document.createElement("div");
        cap.id = "vms-cap";
        document.body.append(cap);
      }
      cap.innerHTML = "";
      cap.append(title);
      if (sub) {
        const small = document.createElement("small");
        small.textContent = sub;
        cap.append(small);
      }
    },
    [CAPTION_CSS, title, sub],
  );
}

(async () => {
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome",
    headless: true,
    args: ["--hide-scrollbars", "--force-dark-mode"],
  });
  const ctx = await browser.newContext({
    viewport: { width: 1440, height: 810 },
    deviceScaleFactor: 4 / 3, // 1920x1080 frames, UI text at a readable size
    colorScheme: "dark",
  });
  const page = await ctx.newPage();
  const hold = (ms) => page.waitForTimeout(ms);

  const frames = [];
  // Page loads are cut out: capture pauses before a navigation and resumes once the page has
  // rendered, and later timestamps shift back by the paused time.
  let paused = false, skew = 0, pausedAt = 0, snap = null; // snap: start of the first frame after a resume
  const pause = () => { paused = true; pausedAt = Date.now() / 1000; frames.push({ cut: pausedAt - skew }); };
  const resume = async () => { await page.waitForTimeout(150); skew += Date.now() / 1000 - pausedAt; snap = pausedAt - (skew - (Date.now() / 1000 - pausedAt)); paused = false;
    await page.evaluate(() => { document.body.style.outline = "1px solid transparent"; }); };
  const go = async (url, ready) => { pause(); await page.goto(url, { waitUntil: "networkidle" }).catch(() => {});
    if (ready) await ready(); await resume(); };
  const cdp = await ctx.newCDPSession(page);
  cdp.on("Page.screencastFrame", async (f) => {
    const file = path.join(framesDir, String(frames.length).padStart(6, "0") + ".jpg");
    fs.writeFileSync(file, Buffer.from(f.data, "base64"));
    if (!paused) {
      frames.push({ file, ts: snap ?? f.metadata.timestamp - skew });
      snap = null;
    }
    await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
  });

  const slide = async (id, ms) => {
    await go(`${SLIDES}?s=${id}`, () => page.waitForTimeout(600));
    await hold(ms - 600);
  };

  await cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: 1920, maxHeight: 1080 });

  // 1. Title
  await slide("title", 6500);

  // 2. Camera wall
  await go(UI, async () => {
    await page.waitForSelector("#wall img, #wall canvas", { timeout: 30000 });
    await page.waitForSelector("#examples .chip", { timeout: 30000 });
    await page.waitForTimeout(2500);
  });
  await caption(page, "Live camera wall: 4 virtual cameras at 5 FPS, on a laptop CPU",
    "YOLO11 on ONNX Runtime · ByteTrack tracking · zone enter/exit events");
  await hold(11000);

  // 3. A question: answer, then the plan and the SQL behind it
  const ask = async (fragment) => {
    await page.locator("#examples .chip", { hasText: fragment }).first().click();
    await page.waitForSelector("#result .result, #result .callout", { timeout: 30000 });
    await page.locator("#result").scrollIntoViewIfNeeded();
  };
  await caption(page, "Ask in plain English");
  await hold(1500);
  await ask("front door zone");
  await caption(page, "The LLM never writes SQL",
    "it fills a typed QueryPlan → validated → compiled to parameterized SQL → read-only role  (offline demo: stand-in planner)");
  await hold(3500);
  const details = page.locator("#result details.debug");
  if (await details.count()) {
    await details.locator("summary").click();
    await page.evaluate(() => document.querySelector("#result details.debug")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
  await hold(7000);

  // 4. Frames answer
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await hold(800);
  await ask("frames with a car");
  await caption(page, "Answers link back to the stored frames");
  await hold(6000);

  // 5. Red team
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await hold(800);
  await ask("Ignore your instructions");
  await caption(page, "Malicious or unsupported questions get a typed refusal",
    "red-team test set: 13 of 13 refused, 0 breaches");
  await hold(6000);

  // 6. Observability
  const kiosk = "orgId=1&from=now-5m&to=now&refresh=5s&kiosk";
  const panels = () => page.waitForTimeout(4000);
  await go(`${GRAFANA}/d/vms-pipeline/?${kiosk}`, panels);
  await hold(1000);
  await caption(page, "Every service traced and metered from day one",
    "pipeline dashboard: ingest vs processed, freshness from capture to queryable, backlog");
  await hold(8500);
  await go(`${GRAFANA}/d/vms-inference/?${kiosk}`, panels);
  await hold(800);
  await caption(page, "Where the worker's time goes, per stage");
  await hold(5500);

  // 7. Evidence
  await slide("capacity", 8000);
  await slide("accuracy", 8000);
  await slide("d7", 13000);
  await slide("drift", 8000);
  await slide("close", 9000);

  const endTs = Date.now() / 1000 - skew;
  await cdp.send("Page.stopScreencast");
  await browser.close();

  // Variable-rate frames -> constant 30 FPS: each frame lasts until the next one arrives.
  const list = [];
  const shots = frames.filter((f) => f.file);
  frames.forEach((f, i) => {
    if (!f.file) return;
    const after = frames[i + 1];
    const next = after ? (after.file ? after.ts : after.cut) : Math.max(endTs, f.ts + 0.04);
    list.push(`file '${f.file}'`, `duration ${Math.max(0.001, next - f.ts).toFixed(4)}`);
  });
  list.push(`file '${shots[shots.length - 1].file}'`);
  const listFile = path.join(framesDir, "list.txt");
  fs.writeFileSync(listFile, list.join("\n") + "\n");
  execFileSync("ffmpeg", [
    "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listFile,
    "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0b1020,fps=30",
    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
  ], { stdio: "inherit" });
  fs.rmSync(framesDir, { recursive: true, force: true });
  console.log(`${shots.length} frames -> ${out}`);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
