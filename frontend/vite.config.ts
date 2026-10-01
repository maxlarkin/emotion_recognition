import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// React's production build embeds "https://react.dev/errors/<code>" in minified error messages.
// It is never fetched, but the offline bundle must not reference external origins at all,
// so the scheme is stripped (the error code stays readable as "react.dev/errors/<code>").
function stripExternalUrls(): Plugin {
  return {
    name: "strip-external-urls",
    apply: "build",
    renderChunk(code) {
      return code.includes("https://react.dev/errors/")
        ? { code: code.replaceAll("https://react.dev/errors/", "react.dev/errors/"), map: null }
        : null;
    },
  };
}

// Dev server proxies the backend the same way nginx does in Docker.
export default defineConfig({
  plugins: [react(), stripExternalUrls()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
