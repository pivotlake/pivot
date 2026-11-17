import { readFileSync } from "node:fs";
import os from "node:os";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const workspaceManifest = readFileSync(new URL("../Cargo.toml", import.meta.url), "utf8");
const workspaceVersion = workspaceManifest.match(/\[workspace\.package\]\s*\nversion = "([^"]+)"/);
if (!workspaceVersion) {
  throw new Error("no [workspace.package] version in ../Cargo.toml");
}

// A pivot server does not report the machine it runs on, so the machine
// running Vite stands in for it on the server card.
const serverHost = {
  hostname: os.hostname(),
  version: workspaceVersion[1],
  cores: os.cpus().length,
  memoryBytes: os.totalmem(),
  architecture: os.arch(),
  platform: os.platform(),
};

export default defineConfig({
  plugins: [react()],
  define: {
    __PIVOT_SERVER_HOST__: JSON.stringify(serverHost),
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
