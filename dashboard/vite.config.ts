import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The gateway serves the build at /dashboard (gateway/server.py). In development, `npm run dev`
// proxies the API to a gateway running on :8000 (itara up).
export default defineConfig({
  base: "/dashboard/",
  plugins: [react()],
  server: { proxy: { "/v1": "http://localhost:8000" } },
});
