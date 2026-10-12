import { defineConfig } from "vitest/config";

// `npm run dev` serves the UI on :5173 and forwards API calls to the query service
// (`make query`, :8070). In production the query service serves the built files itself.
const api = process.env.VMS_QUERY_URL ?? "http://127.0.0.1:8070";

export default defineConfig({
  // /api is the zone editor's server (make zones, :8075).
  server: { port: 5173, proxy: { "/v1": api, "/api": process.env.VMS_ZONES_URL ?? "http://127.0.0.1:8075" } },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    // Two pages: the demo UI (served by the query service) and the zone editor (vms.zones).
    rollupOptions: { input: { main: "index.html", zones: "zones.html" } },
  },
  test: { environment: "node", include: ["tests/**/*.test.ts"] },
});
