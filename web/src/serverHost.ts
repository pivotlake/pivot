// The machine the page is served from, which stands in for the pivot
// server's host until the server reports its own.
export type ServerHost = {
  hostname: string;
  version: string;
  cores: number;
  memoryBytes: number;
  architecture: string;
  platform: string;
};

// Filled in by `define` in vite.config.ts.
declare const __PIVOT_SERVER_HOST__: ServerHost;

export const serverHost: ServerHost = __PIVOT_SERVER_HOST__;
