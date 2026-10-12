// Camera wall: the latest processed frame per camera with detection boxes and zones, polled.

import { fetchLive, type LiveCamera } from "./api";
import { h, s } from "./dom";
import { classColor, classSummary, formatAge } from "./view";

export interface WallOptions {
  intervalMs: number;
  showBoxes: () => boolean;
  showZones: () => boolean;
  onStatus: (text: string, ok: boolean) => void;
}

interface Tile {
  root: HTMLElement;
  img: HTMLImageElement;
  overlay: SVGSVGElement;
  meta: HTMLElement;
  frameId: string | null;
}

export function startWall(container: HTMLElement, options: WallOptions): () => void {
  const tiles = new Map<string, Tile>();
  let cameras: LiveCamera[] = [];
  let inFlight = false;

  const tileFor = (cam: LiveCamera): Tile => {
    let tile = tiles.get(cam.camera_id);
    if (tile === undefined) {
      const img = h("img", { alt: `latest frame from ${cam.name}`, decoding: "async" });
      const overlay = s("svg", { class: "overlay", preserveAspectRatio: "none" });
      const meta = h("figcaption", {});
      const root = h("figure", { class: "tile" }, h("div", { class: "frame" }, img, overlay), meta);
      tile = { root, img, overlay, meta, frameId: null };
      tiles.set(cam.camera_id, tile);
      container.append(root);
    }
    return tile;
  };

  const draw = (cam: LiveCamera, tile: Tile): void => {
    const frame = cam.frame;
    const w = frame?.width ?? 1280;
    const ht = frame?.height ?? 720;
    tile.overlay.setAttribute("viewBox", `0 0 ${w} ${ht}`);
    tile.overlay.replaceChildren();
    if (options.showZones()) {
      for (const zone of cam.zones) {
        const points = zone.polygon.map(([x, y]) => `${x * w},${y * ht}`).join(" ");
        tile.overlay.append(s("polygon", { points, class: "zone" }));
        const [lx, ly] = zone.polygon[0] ?? [0, 0];
        tile.overlay.append(s("text", { x: lx * w + 6, y: ly * ht + 22, class: "zone-label" }, zone.name));
      }
    }
    if (options.showBoxes() && frame !== null) {
      for (const d of frame.detections) {
        const [x1, y1, x2, y2] = d.box;
        const color = classColor(d.class_name);
        tile.overlay.append(
          s("rect", { x: x1, y: y1, width: x2 - x1, height: y2 - y1, class: "box", stroke: color }),
        );
        const id = d.track_id ? ` #${d.track_id.split(":").pop()}` : "";
        tile.overlay.append(
          s("text", { x: x1 + 3, y: Math.max(14, y1 - 5), class: "box-label", fill: color },
            `${d.class_name}${id}`),
        );
      }
    }
    const status = frame
      ? `${formatAge(frame.ts, Date.now())} · ${classSummary(frame.detections.map((d) => d.class_name)) || "no detections"}`
      : "waiting for frames";
    tile.meta.replaceChildren(
      h("strong", {}, cam.name),
      h("span", { class: "muted" }, ` ${cam.camera_id}`),
      h("span", { class: "tile-status" }, status),
    );
  };

  const swapImage = (tile: Tile, url: string, frameId: string): Promise<void> =>
    new Promise((resolve) => {
      // Load off-screen first, so the tile never flashes empty between frames.
      const next = new Image();
      next.onload = () => {
        tile.img.src = url;
        tile.frameId = frameId;
        resolve();
      };
      next.onerror = () => resolve();
      next.src = url;
    });

  const refresh = async (): Promise<void> => {
    if (inFlight) return;
    inFlight = true;
    try {
      cameras = await fetchLive();
      await Promise.all(
        cameras.map(async (cam) => {
          const tile = tileFor(cam);
          if (cam.frame !== null && cam.frame.frame_id !== tile.frameId) {
            await swapImage(tile, cam.frame.image_url, cam.frame.frame_id);
          }
          tile.root.classList.toggle("empty", cam.frame === null);
          draw(cam, tile);
        }),
      );
      options.onStatus(`${cameras.length} cameras · live`, true);
    } catch (err) {
      options.onStatus(`camera wall: ${(err as Error).message}`, false);
    } finally {
      inFlight = false;
    }
  };

  void refresh();
  const timer = window.setInterval(() => void refresh(), options.intervalMs);
  const redraw = (): void => cameras.forEach((cam) => draw(cam, tileFor(cam)));
  window.addEventListener("vms:overlay", redraw);
  return () => {
    window.clearInterval(timer);
    window.removeEventListener("vms:overlay", redraw);
  };
}
