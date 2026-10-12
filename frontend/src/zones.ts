import "./style.css";
import "./zones.css";

import { h, s } from "./dom";
import {
  clamp01,
  edgeInsertIndex,
  hitVertex,
  pixelDistance,
  pointInPolygon,
  round4,
  suggestName,
  zoneProblems,
  type Pt,
  type Zone,
} from "./zones-model";

interface Camera {
  camera_id: string;
  name: string;
  clip_id: string | null;
  frames: number;
  zones: Zone[];
}

interface State {
  cameras: Camera[];
  current: number;
  working: Map<string, Zone[]>; // unsaved edits per camera
  dirty: Set<string>;
  selected: number | null;
  drawing: Pt[] | null;
  frame: number;
  drag: { zone: number; index: number } | null;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const tabs = $("tabs");
const stage = $("stage");
const img = $<HTMLImageElement>("frame");
const svg = document.getElementById("overlay") as unknown as SVGSVGElement;
const list = $("zones");
const slider = $<HTMLInputElement>("frame-index");
const frameLabel = $("frame-label");
const messages = $("messages");
const help = $("help");
const VERTEX_PX = 12; // hit radius in displayed pixels
const EDGE_PX = 8;

const state: State = {
  cameras: [],
  current: 0,
  working: new Map(),
  dirty: new Set(),
  selected: null,
  drawing: null,
  frame: 0,
  drag: null,
};

const camera = (): Camera => state.cameras[state.current]!;
const zones = (): Zone[] => state.working.get(camera().camera_id)!;
const clone = (zs: Zone[]): Zone[] => zs.map((z) => ({ name: z.name, polygon: z.polygon.map((p) => [...p] as Pt) }));

function markDirty(): void {
  state.dirty.add(camera().camera_id);
  render();
}

function displayScale(): { w: number; h: number } {
  const r = img.getBoundingClientRect();
  return { w: r.width || 1, h: r.height || 1 };
}

function toPoint(e: MouseEvent): Pt {
  const r = img.getBoundingClientRect();
  return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)];
}

function say(text: string, kind: "ok" | "error" | "info" = "info", items: string[] = []): void {
  messages.replaceChildren(
    h("div", { class: `note ${kind}` }, text, items.length ? h("ul", {}, ...items.map((i) => h("li", {}, i))) : null),
  );
}

// --- rendering ------------------------------------------------------------------------------

function render(): void {
  renderTabs();
  renderOverlay();
  renderList();
  const cam = camera();
  help.textContent = state.drawing
    ? "Drawing: click to add points · click the first point or press Enter to finish · Backspace undoes a point · Esc cancels"
    : "Drag a point to move it · double-click an edge of the selected zone to add a point · right-click a point to delete it";
  $<HTMLButtonElement>("save").disabled = !state.dirty.has(cam.camera_id);
  $<HTMLButtonElement>("revert").disabled = !state.dirty.has(cam.camera_id);
}

function renderTabs(): void {
  tabs.replaceChildren(
    ...state.cameras.map((c, i) => {
      const tab = h(
        "button",
        { type: "button", class: `tab${i === state.current ? " active" : ""}` },
        c.name,
        state.dirty.has(c.camera_id) ? h("span", { class: "dot", title: "unsaved changes" }, " ●") : null,
      );
      tab.addEventListener("click", () => switchCamera(i));
      return tab;
    }),
  );
}

function renderOverlay(): void {
  const W = 1000;
  const H = img.naturalWidth ? (1000 * img.naturalHeight) / img.naturalWidth : 562.5;
  const scale = W / displayScale().w; // overlay units per displayed pixel
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const pts = (poly: Pt[]): string => poly.map(([x, y]) => `${x * W},${y * H}`).join(" ");
  zones().forEach((z, zi) => {
    const selected = zi === state.selected;
    svg.append(s("polygon", { points: pts(z.polygon), class: `zone${selected ? " selected" : ""}` }));
    const [lx, ly] = z.polygon[0] ?? [0, 0];
    svg.append(s("text", { x: lx * W + 8 * scale, y: ly * H + 18 * scale, class: "zone-name", "font-size": 14 * scale }, z.name));
    if (selected) {
      z.polygon.forEach(([x, y]) => svg.append(s("circle", { cx: x * W, cy: y * H, r: 6 * scale, class: "vertex" })));
    }
  });
  if (state.drawing) {
    const d = state.drawing;
    svg.append(s("polyline", { points: pts(d), class: "drawing" }));
    d.forEach(([x, y], i) =>
      svg.append(s("circle", { cx: x * W, cy: y * H, r: (i === 0 ? 8 : 5) * scale, class: i === 0 ? "vertex first" : "vertex" })),
    );
  }
}

function renderList(): void {
  const items = zones().map((z, zi) => {
    const name = h("input", { type: "text", value: z.name, spellcheck: "false", "aria-label": "zone name" });
    name.addEventListener("input", () => {
      z.name = name.value.trim();
      state.dirty.add(camera().camera_id);
      renderTabs();
      renderOverlay();
      ($<HTMLButtonElement>("save")).disabled = false;
      ($<HTMLButtonElement>("revert")).disabled = false;
    });
    name.addEventListener("focus", () => select(zi));
    const del = h("button", { type: "button", class: "ghost", title: "delete zone" }, "Delete");
    del.addEventListener("click", () => {
      zones().splice(zi, 1);
      state.selected = null;
      markDirty();
    });
    const row = h("li", { class: zi === state.selected ? "selected" : "" }, name, h("span", { class: "muted" }, `${z.polygon.length} pts`), del);
    row.addEventListener("click", (e) => {
      if (e.target === row) select(zi);
    });
    return row;
  });
  list.replaceChildren(...items);
  if (!items.length) list.append(h("li", { class: "muted empty" }, "No zones on this camera yet."));
}

function select(zi: number | null): void {
  if (state.selected === zi) return;
  state.selected = zi;
  renderOverlay();
  [...list.children].forEach((li, i) => li.classList.toggle("selected", i === zi));
}

// --- frames and cameras ---------------------------------------------------------------------

function loadFrame(): void {
  const cam = camera();
  if (!cam.clip_id || !cam.frames) {
    img.removeAttribute("src");
    stage.classList.add("no-frame");
    frameLabel.textContent = "no clip for this camera (make data-mini)";
    return;
  }
  stage.classList.remove("no-frame");
  img.src = `/api/clips/${encodeURIComponent(cam.clip_id)}/frames/${state.frame}`;
  frameLabel.textContent = `${cam.clip_id} · frame ${state.frame + 1} of ${cam.frames}`;
}

function switchCamera(i: number): void {
  state.current = i;
  state.selected = null;
  state.drawing = null;
  const cam = camera();
  slider.max = String(Math.max(0, cam.frames - 1));
  state.frame = Math.floor(cam.frames / 2);
  slider.value = String(state.frame);
  loadFrame();
  messages.replaceChildren();
  render();
}

// --- pointer interaction --------------------------------------------------------------------

svg.addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  const p = toPoint(e);
  const { w, h: ht } = displayScale();
  if (state.drawing) {
    const d = state.drawing;
    if (d.length >= 3 && pixelDistance(d[0]!, p, w, ht) <= VERTEX_PX) finishDrawing();
    else d.push(p);
    renderOverlay();
    return;
  }
  const hit = hitVertex(zones(), p, w, ht, VERTEX_PX, state.selected);
  if (hit && hit.zone === state.selected) {
    state.drag = hit;
    svg.setPointerCapture(e.pointerId);
    return;
  }
  const inside = zones().findIndex((z) => pointInPolygon(p, z.polygon));
  select(hit ? hit.zone : inside >= 0 ? inside : null);
});

svg.addEventListener("pointermove", (e) => {
  if (!state.drag) return;
  const zone = zones()[state.drag.zone]!;
  zone.polygon[state.drag.index] = toPoint(e);
  state.dirty.add(camera().camera_id);
  renderOverlay();
});

svg.addEventListener("pointerup", () => {
  if (state.drag) {
    state.drag = null;
    markDirty();
  }
});

svg.addEventListener("dblclick", (e) => {
  if (state.drawing || state.selected === null) return;
  const zone = zones()[state.selected]!;
  const p = toPoint(e);
  const { w, h: ht } = displayScale();
  const at = edgeInsertIndex(zone.polygon, p, w, ht, EDGE_PX);
  if (at !== null) {
    zone.polygon.splice(at, 0, p);
    markDirty();
  }
});

svg.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (state.drawing || state.selected === null) return;
  const { w, h: ht } = displayScale();
  const hit = hitVertex(zones(), toPoint(e), w, ht, VERTEX_PX, state.selected);
  if (hit && hit.zone === state.selected) {
    const poly = zones()[hit.zone]!.polygon;
    if (poly.length <= 3) {
      say("A zone needs at least 3 points; delete the zone instead.", "error");
      return;
    }
    poly.splice(hit.index, 1);
    markDirty();
  }
});

function finishDrawing(): void {
  const d = state.drawing;
  if (!d || d.length < 3) return;
  zones().push({ name: suggestName(state.cameras.flatMap((c) => state.working.get(c.camera_id)!.map((z) => z.name))), polygon: d });
  state.drawing = null;
  state.selected = zones().length - 1;
  markDirty();
  (list.querySelectorAll("input")[state.selected] as HTMLInputElement | undefined)?.select();
}

window.addEventListener("keydown", (e) => {
  if (!state.drawing || (e.target instanceof HTMLInputElement && e.target.type === "text")) return;
  if (e.key === "Escape") state.drawing = null;
  else if (e.key === "Enter") finishDrawing();
  else if (e.key === "Backspace") state.drawing.pop();
  else return;
  e.preventDefault();
  render();
});

// --- buttons --------------------------------------------------------------------------------

$("new-zone").addEventListener("click", () => {
  state.drawing = [];
  state.selected = null;
  render();
});

$("revert").addEventListener("click", () => {
  const cam = camera();
  state.working.set(cam.camera_id, clone(cam.zones));
  state.dirty.delete(cam.camera_id);
  state.selected = null;
  messages.replaceChildren();
  render();
});

$("save").addEventListener("click", async () => {
  const cam = camera();
  const zs = zones().map((z) => ({ name: z.name, polygon: z.polygon.map(round4) }));
  const problems = zoneProblems(zs);
  if (problems.length) {
    say("Fix these before saving:", "error", problems);
    return;
  }
  const response = await fetch(`/api/cameras/${encodeURIComponent(cam.camera_id)}/zones`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ zones: zs }),
  });
  const body = (await response.json()) as { saved: boolean; catalog?: string; errors?: string[] };
  if (!body.saved) {
    say("The server refused to save:", "error", body.errors ?? []);
    return;
  }
  cam.zones = clone(zs);
  state.working.set(cam.camera_id, clone(zs));
  state.dirty.delete(cam.camera_id);
  say(`Saved ${cam.name} to ${body.catalog}.`, "ok", [
    "Load into the database: make migrate (workers pick zones up within a minute).",
    `Check on video: uv run python -m vms.worker render ${cam.clip_id ?? "CLIP"}:${cam.name} --out ~/Desktop/${cam.name}-zones.mp4`,
  ]);
  render();
});

slider.addEventListener("input", () => {
  state.frame = Number(slider.value);
  loadFrame();
});

img.addEventListener("load", () => renderOverlay());
window.addEventListener("resize", () => renderOverlay());
window.addEventListener("beforeunload", (e) => {
  if (state.dirty.size) e.preventDefault();
});

// --- boot -------------------------------------------------------------------------------------

async function boot(): Promise<void> {
  const response = await fetch("/api/cameras");
  const body = (await response.json()) as { catalog: string; cameras: Camera[] };
  state.cameras = body.cameras;
  for (const c of body.cameras) state.working.set(c.camera_id, clone(c.zones));
  $("catalog").textContent = body.catalog;
  if (!state.cameras.length) {
    say("The catalog has no cameras.", "error");
    return;
  }
  switchCamera(0);
}

boot().catch((err: Error) => say(`Could not load cameras: ${err.message}`, "error"));
