import { defineConfig } from "vite";

// Library build: one ES module entry, the web workers as separate chunks next to it,
// and the WGSL shaders inlined as strings.
export default defineConfig({
  base: "./",
  build: {
    lib: { entry: "src/index.ts", formats: ["es"], fileName: "index" },
    sourcemap: true,
    target: "es2022",
  },
  worker: { format: "es" },
});
