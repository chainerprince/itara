import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The gateway serves the build at /dashboard (gateway/server.py). In development, `npm run dev`
// proxies the API to a gateway running on :8000 (itara up). story.html is the animated demo,
// a page of its own that the Demo tab frames.
export default defineConfig({
  base: "/dashboard/",
  plugins: [react()],
  build: { rollupOptions: { input: { main: "index.html", story: "story.html" } } },
  server: { proxy: { "/v1": "http://localhost:8000" } },
});
