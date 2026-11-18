import { readFileSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const workspaceManifest = readFileSync(new URL("../Cargo.toml", import.meta.url), "utf8");
const workspaceVersion = workspaceManifest.match(/\[workspace\.package\]\s*\nversion = "([^"]+)"/);
if (!workspaceVersion) {
  throw new Error("no [workspace.package] version in ../Cargo.toml");
}

export default defineConfig({
  plugins: [react()],
  define: {
    // The version of pivot the page was built beside, shown on the server card.
    __PIVOT_VERSION__: JSON.stringify(workspaceVersion[1]),
  },
  server: {
    port: 5174,
    // The backend in `server/`, which reads the pivot server for the page.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5175",
        changeOrigin: true,
      },
    },
  },
});
