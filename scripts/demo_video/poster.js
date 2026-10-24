// Poster for the demo video: a camera-wall frame with a play button (docs/images/demo-poster.jpg).
// Usage: node poster.js FRAME.png OUT.jpg   (CHROME_PATH picks the browser)
const { chromium } = require("playwright-core"); const fs = require("fs");
(async () => {
  const [frame, out] = process.argv.slice(2);
  const b64 = fs.readFileSync(frame).toString("base64");
  const b = await chromium.launch({ executablePath: process.env.CHROME_PATH || "/usr/bin/google-chrome" });
  const p = await b.newPage({ viewport: { width: 1280, height: 720 } });
  await p.setContent(`<body style="margin:0"><div style="position:relative;width:1280px;height:720px;
    background:url(data:image/png;base64,${b64}) center/cover">
    <div style="position:absolute;inset:0;background:rgba(5,8,18,.35)"></div>
    <div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:132px;height:132px;border-radius:50%;
      background:rgba(11,16,32,.85);border:3px solid #60a5fa;display:flex;align-items:center;justify-content:center">
      <div style="width:0;height:0;border-left:44px solid #e5e9f2;border-top:28px solid transparent;border-bottom:28px solid transparent;margin-left:10px"></div></div>
    <div style="position:absolute;left:50%;top:calc(50% + 100px);transform:translateX(-50%);padding:8px 18px;border-radius:8px;
      background:rgba(11,16,32,.85);color:#e5e9f2;font:600 26px system-ui,sans-serif;white-space:nowrap">Watch the 2-minute demo</div>
  </div></body>`);
  await p.screenshot({ path: out, type: out.endsWith(".png") ? "png" : "jpeg", quality: out.endsWith(".png") ? undefined : 82 });
  await b.close();
})();
