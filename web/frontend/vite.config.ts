import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev`, proxy API calls to the running pivotdb-server's
// dashboard port (its `--http-bind`), so the hot-reloading SPA hits the same
// API it will in production (where the server serves the built `dist/`).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8081",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
