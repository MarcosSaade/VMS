// Inline SVG charts: bars for one series, lines for several. No chart library.

import { s } from "./dom";
import { niceMax, type Series } from "./view";

const W = 640;
const H = 260;
const PAD = { top: 16, right: 16, bottom: 56, left: 48 };
const PALETTE = ["#60a5fa", "#4ade80", "#f59e0b", "#f472b6", "#a78bfa", "#22d3ee"];

export function renderChart(series: Series[]): SVGSVGElement {
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img" });
  const xs = [...new Set(series.flatMap((se) => se.points.map((p) => p.x)))];
  const max = niceMax(Math.max(0, ...series.flatMap((se) => se.points.map((p) => p.y))));
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;
  const y = (v: number): number => PAD.top + plotH - (v / max) * plotH;
  const band = plotW / Math.max(1, xs.length);
  const xMid = (i: number): number => PAD.left + band * (i + 0.5);

  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const value = max * t;
    svg.append(
      s("line", { x1: PAD.left, x2: W - PAD.right, y1: y(value), y2: y(value), class: "grid" }),
      s("text", { x: PAD.left - 8, y: y(value) + 4, class: "tick", "text-anchor": "end" },
        Number.isInteger(value) ? String(value) : value.toFixed(1)),
    );
  }
  const every = Math.ceil(xs.length / 12); // at most ~12 x labels
  xs.forEach((x, i) => {
    if (i % every !== 0) return;
    const label = s("text", {
      x: xMid(i), y: H - PAD.bottom + 16, class: "tick", "text-anchor": "end",
      transform: `rotate(-30 ${xMid(i)} ${H - PAD.bottom + 16})`,
    }, x.length > 16 ? x.slice(5) : x);
    svg.append(label);
  });

  if (series.length === 1) {
    const only = series[0]!;
    only.points.forEach((p) => {
      const i = xs.indexOf(p.x);
      const bw = Math.min(48, band * 0.7);
      const bar = s("rect", {
        x: xMid(i) - bw / 2, y: y(p.y), width: bw, height: Math.max(0, y(0) - y(p.y)),
        rx: 3, fill: PALETTE[0]!,
      });
      bar.append(s("title", {}, `${p.x}: ${p.y}`));
      svg.append(bar);
    });
  } else {
    series.forEach((se, k) => {
      const color = PALETTE[k % PALETTE.length]!;
      const d = se.points
        .map((p, j) => `${j ? "L" : "M"}${xMid(xs.indexOf(p.x))},${y(p.y)}`)
        .join(" ");
      svg.append(s("path", { d, fill: "none", stroke: color, "stroke-width": 2 }));
      svg.append(
        s("text", { x: W - PAD.right, y: PAD.top + 14 * (k + 1), class: "legend",
          "text-anchor": "end", fill: color }, se.name),
      );
    });
  }
  return svg;
}
