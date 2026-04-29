import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config: dev server on :3001 with /api → Flask :3000, build → ./build/.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3001,
    strictPort: true,
    host: "localhost",
    proxy: {
      "/api": "http://localhost:3000",
    },
  },
  build: {
    outDir: "build",
    emptyOutDir: true,
  },
});
