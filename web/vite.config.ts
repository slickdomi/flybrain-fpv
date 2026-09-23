import { defaultClientConditions, defineConfig } from "vite";

export default defineConfig({
  // Relative base: the build works at any path, e.g. https://<user>.github.io/<repo>/.
  base: "./",
  // Use the vendored flybrain sources straight from the workspace (no library build while developing).
  resolve: { conditions: ["flybrain-source", ...defaultClientConditions] },
  worker: { format: "es" },
});
