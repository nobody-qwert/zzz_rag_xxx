import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite config for local development. All API calls should use the /api prefix.
// This proxy forwards /api/* to the backend running on localhost:8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
